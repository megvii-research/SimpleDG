import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data as data
from pprint import pprint
from model import build_model
from dataset import build_dataset
from augment import FMixup
from transform import TrainTransform, ValTransform
from utils import (
    Clock,
    AverageMeter,
    Accuracy,
    load_config_from_file,
    ensure_dir,
    load_pretrained_backbone,
    save_checkpoint,
    load_checkpoint,
    get_logger,
)


def train(
    config,
    local_rank,
    train_loader,
    mixup_func,
    epoch,
    model,
    criterion,
    optimizer,
    scheduler,
    train_loss_meter,
    train_acc_meter,
):
    if local_rank == 0:
        logger = get_logger(config)
        logger.info("start training...")

    model.train()
    for step, (inputs, target) in enumerate(train_loader):
        iteration = epoch * len(train_loader) + step
        inputs = inputs.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)

        if config.mixup:
            inputs, target = mixup_func(inputs, target)

        predict = model(inputs)
        loss = criterion(predict, target)
        optimizer.zero_grad()
        loss.backward()
        current_lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()

        train_loss_meter.update(loss.item())
        if not config.mixup:
            train_acc_meter.update(predict, target)

        if (iteration + 1) % config.log_interval == 0:
            train_loss_meter.sync()
            train_loss = train_loss_meter.pop()
            if config.mixup:
                train_acc = 0.0
            else:
                train_acc_meter.sync()
                train_acc = train_acc_meter.pop()

            info_str = f"[train] "
            info_str += f"epoch: {epoch + 1}/{config.epochs}, "
            info_str += f"step: {step + 1}/{len(train_loader)}, "
            info_str += f"iter: {iteration + 1}/{config.epochs * len(train_loader)}, "
            info_str += f"lr: {current_lr:.7f}, "
            info_str += f"train_loss: {train_loss}, "
            info_str += f"train_acc: {train_acc}"

            if local_rank == 0:
                logger.parent.handlers.clear()
                logger.info(info_str)


@torch.no_grad()
def evaluate(
    config,
    local_rank,
    val_loader,
    epoch,
    model,
    criterion,
    val_loss_meter,
    val_acc_meter,
):
    if local_rank == 0:
        logger = get_logger(config)
        logger.info("start evaluating...")

    model.eval()
    for step, (inputs, target) in enumerate(val_loader):
        inputs = inputs.cuda(local_rank, non_blocking=True)
        target = target.cuda(local_rank, non_blocking=True)
        predict = model(inputs)
        loss = criterion(predict, target)

        val_loss_meter.update(loss.item())
        val_acc_meter.update(predict, target)

        if (step + 1) % config.log_interval == 0:
            if local_rank == 0:
                logger.info(f"[val] step: {step + 1}/{len(val_loader)}")

    val_loss_meter.sync()
    val_acc_meter.sync()
    val_loss = val_loss_meter.pop()
    val_acc = val_acc_meter.pop()
    info_str = f"[val] "
    info_str += f"epoch: {epoch + 1}/{config.epochs}, "
    info_str += f"val_loss: {val_loss}, val_acc: {val_acc}"

    if local_rank == 0:
        logger.info(info_str)
    return val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    cfg = load_config_from_file(args.config).instance()
    if local_rank == 0:
        pprint(cfg.to_dict())

    dist.init_process_group(
        backend=cfg.backend,
        init_method=cfg.dist_url,
        world_size=cfg.world_size,
        rank=local_rank,
    )
    torch.cuda.set_device(local_rank)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    if local_rank == 0:
        ensure_dir(cfg.local_output_dir)
        ensure_dir(cfg.local_model_dir)
        clock = Clock()

    dist.barrier()
    logger = get_logger(cfg)
    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    train_acc_meter = Accuracy()
    val_acc_meter = Accuracy()

    # dataset
    if local_rank == 0:
        logger.info("start load dataset")

    train_dataset, val_dataset = build_dataset(
        cfg.track,
        cfg.dataset_root,
        val_ratio=cfg.val_ratio,
        train_transform=TrainTransform(cfg),
        val_transform=ValTransform(cfg),
        seed=cfg.seed,
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size // cfg.world_size,
        sampler=data.DistributedSampler(train_dataset, shuffle=True, drop_last=True),
        num_workers=cfg.train_loader_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=cfg.val_batch_size // cfg.world_size,
        sampler=data.DistributedSampler(val_dataset, shuffle=False),
        num_workers=cfg.val_loader_workers,
        pin_memory=cfg.pin_memory,
    )

    mixup_func = FMixup(
        mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=cfg.cutmix_alpha,
        fmix_alpha=cfg.fmix_alpha,
        prob=cfg.mixup_prob,
        label_smoothing=cfg.label_smoothing,
        size=cfg.input_size,
        num_classes=cfg.num_classes,
    )

    if local_rank == 0:
        logger.info("finish load dataset")

    # model
    if local_rank == 0:
        logger.info("start load model")

    model = build_model(cfg.backbone, cfg.num_classes)
    model = model.cuda(local_rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # load pretrained backbone
    if cfg.pretrained_backbone:
        pretrained_weight = load_pretrained_backbone(cfg.pretrained_backbone)
        model.load_state_dict(pretrained_weight)

    if local_rank == 0:
        logger.info("finish load model")

    # optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )

    # scheduler
    def warmup_cosine_lr(current_iters):
        lr_max = 1.0
        lr_min = 0.0
        t_max = cfg.epochs * len(train_loader)
        warmup_iters = cfg.warmup_epochs * len(train_loader)
        if current_iters < warmup_iters:
            return current_iters / warmup_iters
        else:
            # fmt: off
            return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((current_iters - warmup_iters) / (t_max - warmup_iters) * math.pi))
            # fmt: on

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warmup_cosine_lr])

    # criterion
    criterion = nn.CrossEntropyLoss().cuda(local_rank)

    # load checkpoint
    best_val_acc = 0
    save_best = False
    start_epoch = 0
    if cfg.continue_training:
        checkpoint = load_checkpoint(cfg.local_model_dir, "latest")
        if checkpoint:
            if local_rank == 0:
                logger.info("resume from latest checkpoint")
            start_epoch = checkpoint["epoch"]
            best_val_acc = checkpoint["best_val_acc"]
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if local_rank == 0:
                logger.info(f"Successfully loaded the checkpoint from latest")
        else:
            if start_epoch == 0:
                pass
            else:
                if local_rank == 0:
                    logger.warning(f"Unable to load checkpoint from latest")
        del checkpoint

    if local_rank == 0:
        if start_epoch < cfg.epochs:
            logger.info(f"start from epoch {start_epoch + 1}")

    # training
    if local_rank == 0:
        clock.start()
    for epoch in range(start_epoch, cfg.epochs):
        train_loader.sampler.set_epoch(epoch)
        # fmt: off
        train(cfg, local_rank, train_loader, mixup_func, epoch, model, criterion, optimizer, scheduler, train_loss_meter, train_acc_meter)
        val_acc = evaluate(cfg, local_rank, val_loader, epoch, model, criterion, val_loss_meter, val_acc_meter)
        # fmt: on

        # save checkpoint
        if local_rank == 0:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_best = True
            epoch += 1
            checkpoint = {
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            if epoch % cfg.save_latest_checkpoint_interval == 0:
                if save_checkpoint(cfg.local_model_dir, "latest", checkpoint):
                    logger.info(f"Successfully saved the checkpoint to latest")
                else:
                    logger.warning(f"Unable to save checkpoint to latest")
            if epoch % cfg.save_checkpoint_interval == 0:
                if save_checkpoint(cfg.local_model_dir, f"epoch_{epoch}", checkpoint):
                    logger.info(f"Successfully saved the checkpoint to epoch_{epoch}")
                else:
                    logger.warning(f"Unable to save checkpoint to epoch_{epoch}")
            if save_best:
                if save_checkpoint(cfg.local_model_dir, f"best", checkpoint):
                    logger.info(f"Successfully saved the checkpoint to best")
                else:
                    logger.warning(f"Unable to save checkpoint to best")
            del checkpoint
            save_best = False

        if local_rank == 0:
            # fmt: off
            logger.info(f"total time spent on training and evaluation (epoch {epoch}): {clock.lap()}")
            # fmt: on
    if local_rank == 0:
        clock.stop()
        logger.info(f"finish training, total time: {clock.get()}")


if __name__ == "__main__":
    main()
