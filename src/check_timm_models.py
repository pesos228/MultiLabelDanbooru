# check_timm_models.py
import timm

def check_latest_models():
    print("🔍 CHECKING LATEST MODEL VERSIONS IN TIMM")
    print("="*60)
    
    # EfficientNet модели
    print("📱 EFFICIENTNET MODELS:")
    efficientnet_models = [name for name in timm.list_models() if 'efficientnet' in name.lower()]
    efficientnet_models.sort()
    
    for model in efficientnet_models:
        if any(x in model for x in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']):
            print(f"  {model}")
    
    # MobileViT модели  
    print("\n📱 MOBILEVIT MODELS:")
    mobilevit_models = [name for name in timm.list_models() if 'mobilevit' in name.lower()]
    mobilevit_models.sort()
    
    for model in mobilevit_models:
        print(f"  {model}")
    
    # ConvNeXt модели
    print("\n🏗️ CONVNEXT MODELS:")
    convnext_models = [name for name in timm.list_models() if 'convnext' in name.lower()]
    convnext_models.sort()
    
    for model in convnext_models:
        if 'tiny' in model or 'small' in model or 'base' in model:
            print(f"  {model}")
    
    # Swin Transformer
    print("\n🪟 SWIN TRANSFORMER MODELS:")
    swin_models = [name for name in timm.list_models() if 'swin' in name.lower()]
    swin_models.sort()
    
    for model in swin_models:
        if 'tiny' in model or 'small' in model:
            print(f"  {model}")
    
    # Vision Transformer
    print("\n👁️ VISION TRANSFORMER MODELS:")
    vit_models = [name for name in timm.list_models() if 'vit' in name.lower() and 'deit' not in name.lower()]
    vit_models.sort()
    
    for model in vit_models:
        if 'small' in model and 'patch16' in model and '224' in model:
            print(f"  {model}")

if __name__ == "__main__":
    check_latest_models()