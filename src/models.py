"""
Model definitions with LATEST versions from TIMM
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
    Create model with LATEST architectures from TIMM
    
    Args:
        model_name: model name
        num_classes: number of classes (tags)
        **kwargs: additional model parameters
    
    Returns:
        Initialized PyTorch model with latest architecture
    """
    model_name = model_name.lower()
    
    logger.info(f"Creating LATEST model: {model_name}, num_classes: {num_classes}")
    
    # ResNet - используем torchvision (они обновляются реже)
    if model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # EfficientNet - НОВЕЙШИЕ версии EfficientNet-V2
    elif model_name == "efficientnet_b0":
        # Используем EfficientNet-V2 - более новая архитектура!
        model = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=num_classes)
        logger.info("Using EfficientNet-V2 B0 (latest)")
        
    elif model_name == "efficientnet_b1":
        model = timm.create_model('tf_efficientnetv2_b1', pretrained=False, num_classes=num_classes)
        logger.info("Using EfficientNet-V2 B1 (latest)")
        
    elif model_name == "efficientnet_b2":
        model = timm.create_model('tf_efficientnetv2_b2', pretrained=False, num_classes=num_classes)
        logger.info("Using EfficientNet-V2 B2 (latest)")
        
    elif model_name == "efficientnet_b3":
        model = timm.create_model('tf_efficientnetv2_b3', pretrained=False, num_classes=num_classes)
        logger.info("Using EfficientNet-V2 B3 (latest)")

    elif model_name == "efficientnet_s":
        model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=num_classes)
        logger.info("Using EfficientNet-V2 S (latest)")
        
    elif model_name == "convnext_tiny":
        try:
            model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=False, num_classes=num_classes)
            print("✅ Using ConvNeXt-V2 Tiny")
        except:
            try:
                model = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=num_classes)
                print("✅ Using ConvNeXt-V2 Tiny (basic)")
            except:
                model = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
                print("⚠️ Fallback to ConvNeXt-V1 Tiny")
        
    elif model_name == "convnext_small":
        model = timm.create_model('convnextv2_small.fcmae_ft_in22k_in1k', pretrained=False, num_classes=num_classes)
        logger.info("Using ConvNeXt-V2 Small (latest)")
    
    # Swin Transformer - V2 версия
    elif model_name == "swin_t":
        model = timm.create_model('swinv2_tiny_window16_256', pretrained=False, num_classes=num_classes)
        logger.info("Using Swin-V2 Tiny (latest)")
        
    elif model_name == "swin_s":
        model = timm.create_model('swinv2_small_window16_256', pretrained=False, num_classes=num_classes)
        logger.info("Using Swin-V2 Small (latest)")
    
    # Vision Transformer - DeiT-III (самый новый!)  
    elif model_name == "vit_s_16":
        model = timm.create_model('deit3_small_patch16_224', pretrained=False, num_classes=num_classes)
        logger.info("Using DeiT-III Small (latest ViT)")
        
    elif model_name == "vit_b_16":
        model = timm.create_model('deit3_base_patch16_224', pretrained=False, num_classes=num_classes)
        logger.info("Using DeiT-III Base (latest ViT)")
    
    # MobileViT - V2 версия
    elif model_name == "mobilevit_s":
        try:
            model = timm.create_model('mobilevitv2_050', pretrained=False, num_classes=num_classes)
            logger.info("Using MobileViT-V2 0.5 (latest)")
        except:
            # Fallback к обычному MobileViT
            model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
            logger.info("Using MobileViT-V1 Small (fallback)")
            
    elif model_name == "mobilevit_xs":
        try:
            model = timm.create_model('mobilevitv2_075', pretrained=False, num_classes=num_classes)
            logger.info("Using MobileViT-V2 0.75 (latest)")
        except:
            model = timm.create_model('mobilevit_xs', pretrained=False, num_classes=num_classes)
            logger.info("Using MobileViT-V1 XS (fallback)")
    
    # Новые модели 2024
    elif model_name == "efficientvit_b0":
        try:
            model = timm.create_model('efficientvit_b0', pretrained=False, num_classes=num_classes)
            logger.info("Using EfficientViT B0 (2024 model)")
        except:
            model = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=num_classes)
            logger.info("Fallback to EfficientNet-V2 B0")
    
    elif model_name == "coatnet_0":
        try:
            model = timm.create_model('coatnet_0_rw_224', pretrained=False, num_classes=num_classes)
            logger.info("Using CoAtNet-0 (hybrid CNN+Transformer)")
        except:
            model = timm.create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=num_classes)
            logger.info("Fallback to EfficientNet-V2 B0")
    
    # MaxViT - новая архитектура 2022-2024
    elif model_name == "maxvit_tiny":
        try:
            model = timm.create_model('maxvit_tiny_tf_224', pretrained=False, num_classes=num_classes)
            logger.info("Using MaxViT Tiny (latest hybrid architecture)")
        except:
            model = timm.create_model('swinv2_tiny_window16_256', pretrained=False, num_classes=num_classes)
            logger.info("Fallback to Swin-V2 Tiny")
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: resnet34, resnet50, efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, "
            f"convnext_tiny, convnext_small, swin_t, swin_s, vit_s_16, vit_b_16, "
            f"mobilevit_s, mobilevit_xs, efficientvit_b0, coatnet_0, maxvit_tiny"
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
    """Return list of available latest models"""
    return [
        # Classic CNNs
        "resnet34", "resnet50", 
        
        # EfficientNet family (V2 - latest)
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_s",
        
        # ConvNeXt (V2 - latest)
        "convnext_tiny", "convnext_small",
        
        # Swin Transformer (V2 - latest)
        "swin_t", "swin_s",
        
        # Vision Transformer (DeiT-III - latest)
        "vit_s_16", "vit_b_16",
        
        # Mobile architectures (V2 - latest)
        "mobilevit_s", "mobilevit_xs",
        
        # New 2024 models
        "efficientvit_b0", "coatnet_0", "maxvit_tiny"
    ]


def print_model_info():
    """Print information about latest model versions"""
    print("🚀 LATEST MODEL VERSIONS USED:")
    print("="*50)
    print("EfficientNet → EfficientNet-V2 (2021, improved training)")
    print("ConvNeXt → ConvNeXt-V2 (2023, better performance)")  
    print("Swin → Swin-V2 (2022, improved architecture)")
    print("ViT → DeiT-III (2022, latest distillation)")
    print("MobileViT → MobileViT-V2 (2022, better efficiency)")
    print("+ New: EfficientViT, CoAtNet, MaxViT (2022-2024)")
    print("="*50)


if __name__ == "__main__":
    print_model_info()