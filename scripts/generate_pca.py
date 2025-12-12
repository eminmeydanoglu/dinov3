from __future__ import annotations
import os
import argparse
import glob
import torch
import time
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoImageProcessor, AutoModel, ViTImageProcessor

PATCH_SIZE = 14 


def get_args():
    parser = argparse.ArgumentParser(description="Run DINOv3 PCA visualization on a folder of images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save PCA visualizations.")
    parser.add_argument("--model", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m", help="HuggingFace model ID.")
    parser.add_argument("--img_size", type=int, default=770, help="Resize images to this size (should be divisible by patch size).")
    return parser.parse_args()

from huggingface_hub import hf_hub_download

def load_model(model_name):
    print(f"Loading model: {model_name}")
    
    # 1. Setup Processor (Robust fallback)
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception as e_proc:
        print(f"AutoImageProcessor failed ({e_proc}), falling back to ViTImageProcessor.")
        try:
            processor = ViTImageProcessor.from_pretrained(model_name)
        except Exception:
            print("ViTImageProcessor failed, using manual defaults.")
            processor = ViTImageProcessor(
                do_resize=True,
                size={"height": 224, "width": 224}, 
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                resample=3 
            )

    # 2. Setup Model
    try:
        # Try AutoModel first
        model = AutoModel.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    except Exception as e:
        print(f"AutoModel failed ({e}). Attempting manual load from dinov3 library...")
        try:
            # Import architecture from local code
            from dinov3.hub.backbones import dinov3_vits16, dinov3_vitl16
            
            if "vits16" in model_name:
                model_fn = dinov3_vits16
            elif "vitl16" in model_name:
                model_fn = dinov3_vitl16
            else:
                # Default fallback or error
                print("Could not infer arch from name, assuming vits16")
                model_fn = dinov3_vits16
            
            print(f"Instantiating {model_fn.__name__}...")
            model = model_fn(pretrained=False)
            
            print("Downloading weights from HF...")
            # Try safetensors first, then bin
            try:
                weights_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
            except Exception as e_safetensor:
                print(f"Safetensors load failed ({e_safetensor}), trying pytorch_model.bin")
                weights_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
                state_dict = torch.load(weights_path, map_location="cpu")
            
            # Load state dict
            # Note: HF models often prefix keys differently or might match exactly if converted faithfully.
            
            def remap_hf_to_dinov3(state_dict):
                new_dict = {}
                # Extract QKV keys to fuse later
                qkv_map = {} # {layer_idx: {'q': w, 'k': w, 'v': w, 'q_b': b, ...}}
                
                for k, v in state_dict.items():
                    # Patch Embeddings
                    if k == "embeddings.patch_embeddings.weight":
                        new_dict["patch_embed.proj.weight"] = v
                    elif k == "embeddings.patch_embeddings.bias":
                        new_dict["patch_embed.proj.bias"] = v
                    elif k == "embeddings.cls_token":
                        val = v
                        if val.ndim == 3 and val.shape[1] == 1:
                            val = val.squeeze(1)
                        new_dict["cls_token"] = val
                    elif k == "embeddings.mask_token":
                        val = v
                        if val.ndim == 3 and val.shape[1] == 1:
                            val = val.squeeze(1)
                        new_dict["mask_token"] = val
                    elif k == "embeddings.register_tokens":
                        # DINOv3 might explicitly have register tokens or part of variable length
                        # Native: likely 'register_tokens' if it exists, or handled elsewhere.
                        # Let's try direct mapping if missing
                        pass 
                    
                    # Blocks
                    elif k.startswith("layer."):
                        parts = k.split(".")
                        idx = parts[1]
                        block_prefix = f"blocks.{idx}"
                        suffix = ".".join(parts[2:]) # attention.q_proj.weight
                        
                        # Norms
                        if "norm1" in suffix:
                            new_dict[f"{block_prefix}.norm1.{parts[-1]}"] = v
                        elif "norm2" in suffix:
                            new_dict[f"{block_prefix}.norm2.{parts[-1]}"] = v
                        elif "layer_scale1" in suffix:
                             new_dict[f"{block_prefix}.ls1.gamma"] = v
                        elif "layer_scale2" in suffix:
                             new_dict[f"{block_prefix}.ls2.gamma"] = v
                        
                        # MLP
                        # HF: mlp.up_proj (fc1), mlp.down_proj (fc2)? 
                        # Native: mlp.fc1, mlp.fc2
                        elif "mlp.fc1" in suffix or "mlp.up_proj" in suffix: # HF often calls it fc1 too or up_proj
                             # Check specific naming in HF checkpoint from error log: 'layer.0.mlp.up_proj.weight'
                             if "up_proj" in suffix:
                                 new_dict[f"{block_prefix}.mlp.fc1.{parts[-1]}"] = v
                        elif "mlp.fc2" in suffix or "mlp.down_proj" in suffix:
                             if "down_proj" in suffix:
                                 new_dict[f"{block_prefix}.mlp.fc2.{parts[-1]}"] = v

                        # Attention (QKV Fusion)
                        elif "attention" in suffix:
                            # layer.0.attention.q_proj.weight
                            # Native: blocks.0.attn.qkv.weight
                            if "q_proj" in suffix:
                                qkv_map.setdefault(idx, {}).setdefault('q_' + parts[-1], v)
                            elif "k_proj" in suffix:
                                qkv_map.setdefault(idx, {}).setdefault('k_' + parts[-1], v)
                            elif "v_proj" in suffix:
                                qkv_map.setdefault(idx, {}).setdefault('v_' + parts[-1], v)
                            elif "o_proj" in suffix:
                                new_dict[f"{block_prefix}.attn.proj.{parts[-1]}"] = v

                # Fuse QKV
                for idx, vals in qkv_map.items():
                    # Weights
                    if 'q_weight' in vals and 'k_weight' in vals and 'v_weight' in vals:
                        # Concat along output dim (dim=0)
                        q = vals['q_weight']
                        k = vals['k_weight']
                        v = vals['v_weight']
                        fused_w = torch.cat([q, k, v], dim=0)
                        new_dict[f"blocks.{idx}.attn.qkv.weight"] = fused_w
                    
                    # Biases
                    if 'q_bias' in vals and 'k_bias' in vals and 'v_bias' in vals:
                        q_b = vals['q_bias']
                        k_b = vals['k_bias']
                        v_b = vals['v_bias']
                        fused_b = torch.cat([q_b, k_b, v_b], dim=0)
                        new_dict[f"blocks.{idx}.attn.qkv.bias"] = fused_b
                
                return new_dict

            state_dict = remap_hf_to_dinov3(state_dict)
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded. Missing/Unexpected keys: {msg}")
            
            model.cuda()
            model.eval()
            
        except Exception as manual_e:
            print(f"Manual loading failed: {manual_e}")
            raise e

    return model, processor

def run_pca(features, n_components=3):
    # features: (N_patches, Dim)
    pca = PCA(n_components=n_components)
    pca.fit(features)
    pca_features = pca.transform(features)
    return pca_features

def plot_pca(pca_features, h_patches, w_patches, save_path):
    # pca_features: (N_patches, 3)
    # Normalize to [0, 1] for RGB
    pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0))
    
    # Reshape to (H_patches, W_patches, 3)
    try:
        pca_img = pca_features.reshape(h_patches, w_patches, 3)
    except ValueError as e:
        print(f"Error reshaping PCA: {e}. Expected {h_patches}x{w_patches}={h_patches*w_patches}, got {pca_features.shape[0]}")
        return

    plt.figure(figsize=(10, 10))
    plt.imshow(pca_img)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    args = get_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    model, processor = load_model(args.model)
    
    # Get patch size from model config if possible
    # Determine patch size (DINOv3 default is 14)
    # Native model doesn't have a config attribute, so we check patch_embed or default to 14
    if hasattr(model, "patch_embed"):
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
             patch_size = patch_size[0]
    else:
        patch_size = 14
    print(f"Using patch size: {patch_size}")

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))
    
    print(f"Found {len(image_files)} images.")
    
    inference_times = []
    
    for img_path in image_files:
        print(f"Processing {img_path}...")
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Helper to calculate grid size
            # We resize manually to control aspect ratio and divisibility
            w, h = img.size
            if args.img_size > 0:
                scale = args.img_size / min(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Make divisible by patch_size (rounding down/up)
                new_w = (new_w // patch_size) * patch_size
                new_h = (new_h // patch_size) * patch_size
                
                # Ensure at least 1 patch
                new_w = max(new_w, patch_size)
                new_h = max(new_h, patch_size)
                
                img = img.resize((new_w, new_h), resample=Image.BICUBIC)
            
            w, h = img.size
            h_patches = h // patch_size
            w_patches = w // patch_size
            
            inputs = processor(images=img, return_tensors="pt", do_resize=False, do_center_crop=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            t_start = time.time()
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True)
                
                # Get last hidden state
                last_hidden_state = outputs.last_hidden_state # (B, Seq, Dim)
                
                # Remove CLS token (assuming index 0)
                # Check model type to be sure, but standard ViT uses index 0.
                # However, DINOv2 might have register tokens.
                # If registers are present, we need to skip them too.
                # DINOv3 might have them.
                
                # Let's verify sequence length vs patch count
                B, Seq, Dim = last_hidden_state.shape
                expected_patches = h_patches * w_patches
                
                # Usually: Seq = 1 (CLS) + [Registers] + Patches
                prefix_tokens = Seq - expected_patches
                
                if prefix_tokens < 0:
                    print(f"Warning: Sequence length {Seq} < expected patches {expected_patches}. Resizing logic mismatch.")
                    # Fallback: assume all are patches? No, that breaks.
                    continue
                
                # Take only the last 'expected_patches' tokens
                spatial_tokens = last_hidden_state[0, -expected_patches:, :] # (N_patches, Dim)
                
                feat = spatial_tokens.cpu().numpy()
            t_end = time.time()
            inference_times.append(t_end - t_start)
                
            # Run PCA
            pca_feat = run_pca(feat)
            
            # Save
            base_name = os.path.basename(img_path)
            name, _ = os.path.splitext(base_name)
            save_path = os.path.join(args.output_dir, f"{name}_pca.png")
            
            plot_pca(pca_feat, h_patches, w_patches, save_path)
            # print(f"Saved PCA to {save_path}")
            
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            import traceback
            traceback.print_exc()

    if inference_times:
        mean_time = np.mean(inference_times)
        mean_hz = 1.0 / mean_time
        print(f"\n--- Performance Summary ---")
        print(f"Model Used: {args.model}")
        print(f"Processed {len(image_files)} images.")
        print(f"Average Inference Time: {mean_time*1000:.2f} ms")
        print(f"Mean Inference Hz: {mean_hz:.2f} Hz")

if __name__ == "__main__":
    main()
