# Final submission script for the GlacierHack segmentation challenge.
# This script ensembles the top 4 simple U-Net models.

# Required pre-installed packages:
# pip install torch torchvision numpy tifffile scikit-learn pillow

import os
import argparse
import re
import numpy as np
import torch
import torch.nn as nn
from tifffile import imread, imwrite
from PIL import Image

# --- Helper function to get the tile ID ---
def get_tile_id(filename):
    match = re.search(r'(\d+)', filename)
    if match: return match.group(1)
    return None

# --- The Champion Model: Simple U-Net Architecture ---
class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super().__init__()
        def conv_block(i, o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU())
        self.e1=conv_block(in_channels,64); self.p1=nn.MaxPool2d(2,2); self.e2=conv_block(64,128); self.p2=nn.MaxPool2d(2,2); self.b=conv_block(128,256)
        self.u2=nn.ConvTranspose2d(256,128,2,2); self.d2=conv_block(256,128); self.u1=nn.ConvTranspose2d(128,64,2,2); self.d1=conv_block(128,64)
        self.out=nn.Conv2d(64,out_channels,1)
    def forward(self, x):
        s1=self.e1(x); p1=self.p1(s1); s2=self.e2(p1); p2=self.p2(s2); b=self.b(p2)
        u2=self.u2(b); m2=torch.cat([u2,s2],1); d2=self.d2(m2); u1=self.u1(d2); m1=torch.cat([u1,s1],1); d1=self.d1(m1); o=self.out(d1); return o

# --- Manual Preprocessing Function ---
def preprocess_image(image_stack_np):
    resized_channels = [];
    for i in range(image_stack_np.shape[2]):
        channel = image_stack_np[:, :, i]; pil_image = Image.fromarray(channel)
        resized_pil = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
        resized_channels.append(np.array(resized_pil))
    resized_stack = np.stack(resized_channels, axis=-1); normalized_stack = (resized_stack / 32767.5) - 1.0
    transposed_stack = np.transpose(normalized_stack, (2, 0, 1)); return torch.from_numpy(transposed_stack).float()

# --- Core Inference Function ---
def maskgeration(imagepath, model_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    actual_out_dir = os.path.dirname(model_path)
    os.makedirs(actual_out_dir, exist_ok=True)
    
    # ✅ --- ENSEMBLING LOGIC: LOAD MODELS --- ✅
    print(f"Loading ensemble model from {model_path}...")
    ensemble_state_dict = torch.load(model_path, map_location=DEVICE)
    
    models = []
    for i in range(len(ensemble_state_dict)):
        model = UNet().to(DEVICE)
        model.load_state_dict(ensemble_state_dict[f'model_{i}'])
        model.eval()
        models.append(model)
    print(f"Successfully loaded {len(models)} models for ensembling.")

    ref_band = 'Band1'
    if ref_band not in imagepath: ref_band = sorted(imagepath.keys())[0]
        
    image_files = sorted(os.listdir(imagepath[ref_band]))
    generated_masks = {}

    for filename in image_files:
        if not filename.endswith('.tif'): continue
        try:
            tile_id = get_tile_id(filename)
            if not tile_id: continue
            first_band_path = os.path.join(imagepath[ref_band], filename)
            with Image.open(first_band_path) as img:
                original_width, original_height = img.size

            band_layers = []; band_order = sorted(imagepath.keys())
            for band in band_order:
                file_path = os.path.join(imagepath[band], filename)
                band_layers.append(imread(file_path).astype(np.float32))

            image_np = np.stack(band_layers, axis=-1)
            input_tensor = preprocess_image(image_np).unsqueeze(0).to(DEVICE)

            # ✅ --- ENSEMBLING LOGIC: PREDICT AND AVERAGE --- ✅
            all_probs = []
            with torch.no_grad():
                for model in models:
                    logits = model(input_tensor)
                    probs = torch.sigmoid(logits)
                    all_probs.append(probs)
            
            # Average the probability maps from all models
            avg_probs = torch.mean(torch.stack(all_probs), dim=0)
            
            pred_mask_256 = (avg_probs > 0.5).cpu().squeeze().numpy().astype(np.uint8)
            pil_pred = Image.fromarray(pred_mask_256)
            resized_pred = pil_pred.resize((original_width, original_height), Image.Resampling.NEAREST)
            final_pred_mask = np.array(resized_pred)

            output_mask = (final_pred_mask * 255).astype(np.uint8)
            output_filepath = os.path.join(actual_out_dir, filename)
            imwrite(output_filepath, output_mask)
            
            generated_masks[tile_id] = output_mask
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return generated_masks

# --- Do not update this section ---
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True); parser.add_argument("--masks", required=True)
    parser.add_argument("--out", required=True); args = parser.parse_args()
    imagepath = {}
    for band in os.listdir(args.data):
        band_path = os.path.join(args.data, band);
        if os.path.isdir(band_path): imagepath[band] = band_path
    maskgeration(imagepath, args.out)

if __name__ == "__main__":
    main()