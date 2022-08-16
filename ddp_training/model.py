import torchvision
import timm


def build_model(model_name, num_classes):
    if model_name in (
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ):
        model = getattr(torchvision.models, model_name)(
            zero_init_residual=True, num_classes=num_classes, pretrained=False
        )
    elif model_name in (
        "legacy_seresnet18",
        "legacy_seresnet34",
        "legacy_seresnet50",
        "legacy_seresnet101",
        "legacy_seresnet152",
    ):
        model = timm.models.create_model(
            model_name, num_classes=num_classes, pretrained=False
        )
    return model
