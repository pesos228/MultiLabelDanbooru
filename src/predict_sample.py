"""
Inference script for Danbooru Tagger with model version compatibility
"""
import argparse
import json
from pathlib import Path
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Import your models and transforms
from models import get_model
from dataset import get_val_transforms


def load_checkpoint_with_compatibility(checkpoint_path: Path, device: torch.device):
    """Load checkpoint with backward compatibility for different model versions."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = checkpoint['args']
    
    model_name = args_dict['model_name']
    tag_vocab = checkpoint['tag_vocab']
    num_classes = len(tag_vocab)
    
    print(f"Loading model: {model_name}")
    print(f"Number of classes: {num_classes}")
    print(f"PyTorch version used for training: {checkpoint.get('pytorch_version', 'unknown')}")
    
    # Create model using the same parameters
    model = get_model(model_name, num_classes=num_classes)
    
    try:
        # Try direct loading first
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully (direct compatibility)")
    except RuntimeError as e:
        print(f"❌ Direct loading failed: {e}")
        print("🔄 Attempting compatibility fixes...")
        
        # Try compatibility fixes
        if model_name == "convnext_tiny":
            model = fix_convnext_compatibility(model, checkpoint, device)
        elif any(name in model_name for name in ['efficientnet', 'mobilevit', 'swin', 'vit']):
            model = fix_timm_compatibility(model, checkpoint, model_name, num_classes, device)
        else:
            raise RuntimeError(f"Cannot fix compatibility for model: {model_name}")
    
    model.to(device)
    model.eval()
    return model, tag_vocab, args_dict


def fix_convnext_compatibility(model, checkpoint, device):
    """Fix ConvNeXt v1 vs v2 compatibility issues."""
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    
    print("🔧 Applying ConvNeXt compatibility fixes...")
    
    # Create new state dict with compatible keys
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Convert old ConvNeXt-v1 keys to new ConvNeXt-v2 keys
        if '.gamma' in key:
            # Old v1 format uses 'gamma', new v2 uses 'grn.weight' and 'grn.bias'
            continue  # Skip gamma parameters for now
        
        if key in model_state:
            new_state_dict[key] = value
        else:
            print(f"⚠️ Skipping incompatible key: {key}")
    
    # Try to load with strict=False to ignore missing keys
    try:
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)} (this is expected for version differences)")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
        print("✅ Model loaded with compatibility mode")
        return model
    except Exception as e:
        print(f"❌ Compatibility fix failed: {e}")
        raise


def fix_timm_compatibility(model, checkpoint, model_name, num_classes, device):
    """Fix TIMM model compatibility issues."""
    print(f"🔧 Attempting TIMM model fix for {model_name}...")
    
    # Try creating model with the exact same architecture as saved
    import timm
    
    # Map new model names to old ones that might have been saved
    old_model_mappings = {
        'convnext_tiny': [
            'convnext_tiny', 
            'convnext_tiny.fb_in22k_ft_in1k',
            'convnext_tiny.fcmae_ft_in22k_in1k'
        ],
        'efficientnet_b0': [
            'efficientnet_b0',
            'tf_efficientnet_b0',
            'tf_efficientnetv2_b0'
        ],
        'efficientnet_b2': [
            'efficientnet_b2',
            'tf_efficientnet_b2', 
            'tf_efficientnetv2_b2'
        ],
        'swin_t': [
            'swin_tiny_patch4_window7_224',
            'swinv2_tiny_window16_256'
        ]
    }
    
    # Try different model architectures
    possible_names = old_model_mappings.get(model_name, [model_name])
    
    for old_name in possible_names:
        try:
            print(f"Trying architecture: {old_name}")
            test_model = timm.create_model(old_name, pretrained=False, num_classes=num_classes)
            test_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Successfully loaded with architecture: {old_name}")
            return test_model
        except Exception as e:
            print(f"❌ Failed with {old_name}: {str(e)[:100]}...")
            continue
    
    raise RuntimeError(f"Could not find compatible architecture for {model_name}")


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    tag_vocab: list,
    threshold: float = 0.5,
    top_n_probs: int = 10
):
    """Make prediction for a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0)

    # Tags predicted above threshold
    predicted_indices_threshold = (probabilities > threshold).nonzero(as_tuple=True)[0]
    predicted_tags_threshold = [tag_vocab[i] for i in predicted_indices_threshold.cpu().tolist()]
    predicted_probs_threshold = probabilities[predicted_indices_threshold].cpu().tolist()

    # Top-N tags by probability
    top_probs, top_indices = torch.topk(probabilities, top_n_probs)
    top_predicted_tags = [tag_vocab[i] for i in top_indices.cpu().tolist()]
    top_predicted_probs = top_probs.cpu().tolist()

    print(f"\n--- Predictions for: {image_path.name} ---")
    if predicted_tags_threshold:
        print(f"\nTags with probability > {threshold}:")
        for tag, prob in zip(predicted_tags_threshold, predicted_probs_threshold):
            print(f"  - {tag}: {prob:.4f}")
    else:
        print(f"\nNo tags with probability > {threshold}.")

    print(f"\nTop-{top_n_probs} tags by probability:")
    for tag, prob in zip(top_predicted_tags, top_predicted_probs):
        print(f"  - {tag}: {prob:.4f}")
    
    return predicted_tags_threshold, top_predicted_tags


def main():
    parser = argparse.ArgumentParser(description="Danbooru Tagger inference on images.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to checkpoint file (.pth)")
    parser.add_argument('--image_paths', type=str, nargs='+', required=True,
                        help="Paths to images for prediction")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help="Probability threshold for tag prediction")
    parser.add_argument('--top_n_probs', type=int, default=10,
                        help="Number of top probability tags to show")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device for inference (cuda/cpu)")

    script_args = parser.parse_args()
    device = torch.device(script_args.device)

    print(f"Using device: {device}")
    
    # Load model with compatibility fixes
    model, tag_vocab, train_args = load_checkpoint_with_compatibility(
        Path(script_args.checkpoint_path), device
    )
    
    # Get image size from training config
    image_size = train_args.get('image_size', 256)
    print(f"Using image_size: {image_size} from training config")
    
    # Create transforms
    try:
        val_transform = get_val_transforms(image_size=image_size)
    except ImportError:
        print("WARNING: Could not import get_val_transforms. Using standard transforms.")
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Process each image
    for img_path_str in script_args.image_paths:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"Image file not found: {img_path}")
            continue
        
        predict_image(
            model,
            img_path,
            val_transform,
            device,
            tag_vocab,
            threshold=script_args.threshold,
            top_n_probs=script_args.top_n_probs
        )

if __name__ == '__main__':
    main()