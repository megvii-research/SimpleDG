from random import random, randint
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as F
from timm.data import RandAugment, rand_augment_ops


def fourier_domain_adaptation(img, target_img, beta):
    img = np.squeeze(img)
    target_img = np.squeeze(target_img)

    if target_img.shape != img.shape:
        raise ValueError(
            "The source and target images must have the same shape,"
            " but got {} and {} respectively.".format(img.shape, target_img.shape)
        )

    # get fft of both source and target
    fft_src = np.fft.fft2(img.astype(np.float32), axes=(0, 1))
    fft_trg = np.fft.fft2(target_img.astype(np.float32), axes=(0, 1))

    # extract amplitude and phase of both fft-s
    amplitude_src, phase_src = np.abs(fft_src), np.angle(fft_src)
    amplitude_trg = np.abs(fft_trg)

    # mutate the amplitude part of source with target
    amplitude_src = np.fft.fftshift(amplitude_src, axes=(0, 1))
    amplitude_trg = np.fft.fftshift(amplitude_trg, axes=(0, 1))
    height, width = amplitude_src.shape[:2]
    border = np.floor(min(height, width) * beta).astype(int)
    center_y, center_x = np.floor([height / 2.0, width / 2.0]).astype(int)

    y1, y2 = center_y - border, center_y + border + 1
    x1, x2 = center_x - border, center_x + border + 1

    amplitude_src[y1:y2, x1:x2] = amplitude_trg[y1:y2, x1:x2]
    amplitude_src = np.fft.ifftshift(amplitude_src, axes=(0, 1))

    # get mutated image
    src_image_transformed = np.fft.ifft2(
        amplitude_src * np.exp(1j * phase_src), axes=(0, 1)
    )
    src_image_transformed = np.real(src_image_transformed)

    return src_image_transformed


class RandomFDA(object):
    def __init__(self, beta=[0, 0.01], p=0.5):
        self.beta = beta
        self.p = p
        self.last_image = None

    def __call__(self, img):
        auged_img = img
        if random() < self.p and self.last_image:
            beta = np.random.uniform(self.beta[0], self.beta[1])
            auged_img = Image.fromarray(
                fourier_domain_adaptation(
                    np.array(img), np.array(self.last_image), beta
                ).astype(np.uint8)
            )
        self.last_image = img
        return auged_img


class RandomPad(object):
    def __init__(self, pad_range=(0, 200), fill=0, padding_mode="constant", p=1.0):
        self.pad_range = pad_range
        self.fill = fill
        self.padding_mode = padding_mode
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            padding = randint(*self.pad_range)
            return F.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)
        else:
            return img


class RandomColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            return self.transform(img)
        else:
            return img


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random() < self.p:
            sigma = random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Transform(object):
    def __call__(self, image):
        return self.transform(image)


class TrainTransform(Transform):
    def __init__(self, config):
        ops = rand_augment_ops(magnitude=config.rand_aug_magnitude)
        ops.append(
            RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.1)
        )
        ops.append(transforms.RandomGrayscale(p=0.1))
        ops.append(RandomGaussianBlur(p=0.1))
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(config.input_size, scale=(0.7, 1.0)),
                RandAugment(ops, num_layers=config.rand_aug_num_layers),
                RandomFDA(beta=[0, 0.01], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.normalize_mean, std=config.normalize_std
                ),
            ]
        )


class ValTransform(Transform):
    def __init__(self, config):
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.input_size),
                transforms.CenterCrop(config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.normalize_mean, std=config.normalize_std
                ),
            ]
        )


class TestTransform(Transform):
    def __init__(self, config):
        self.transform = transforms.Compose(
            [
                transforms.Resize(config.input_size),
                transforms.CenterCrop(config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.normalize_mean, std=config.normalize_std
                ),
            ]
        )
