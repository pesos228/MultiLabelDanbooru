"""
Model definitions for multi-label classification with GPU optimization
"""
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torchvision.models as models
import timm

logger = logging.getLogger(__name__)


def get_model(model_name: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Create model for multi-label classification

    Args:
        model_name: model name
        num_classes: number of classes (tags)
        **kwargs: additional model parameters

    Returns:
        Initialized PyTorch model
    """
    model_name = model_name.lower()

    logger.info(f"Creating model: {model_name}, num_classes: {num_classes}")

    if model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)

    elif model_name == "efficientnet_b2":
        model = timm.create_model('efficientnet_b2', pretrained=False, num_classes=num_classes)

    elif model_name == "efficientnet_b3":
        model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)

    elif model_name == "efficientnet_b4":
        model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)

    elif model_name == "convnext_tiny":
        model = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)

    elif model_name == "swin_t":
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)

    elif model_name == "vit_s_16":
        model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes)

    elif model_name == "mobilevit_s":
        model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: resnet34, resnet50, efficientnet_b0, efficientnet_b2, "
            f"efficientnet_b3, efficientnet_b4, convnext_tiny, swin_t, vit_s_16, mobilevit_s"
        )

    # Weight initialization
    _initialize_weights(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model created. Total parameters: {total_params:,}, "
                f"trainable: {trainable_params:,}")

    return model


def _initialize_weights(model: nn.Module):
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_available_models() -> list:
    """Return list of available models"""
    return [
        "resnet34", "resnet50", 
        "efficientnet_b0", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
        "convnext_tiny", "swin_t", "vit_s_16", "mobilevit_s"
    ]