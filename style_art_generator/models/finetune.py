# style_art_generator/models/finetune.py
import os
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm
import glob
import json
import time
from torch.utils.data import Dataset, DataLoader
import random
import argparse
import numpy as np

class StyleImageDataset(Dataset):
    def __init__(self, image_paths, tokenizer, text_encoder, size=512, prompt=""):
        self.image_paths = image_paths
        self.size = size
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        
        # Use a generic prompt if none provided
        self.prompt = prompt if prompt else "an image in a specific artistic style"
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGBA")
        
        # Resize and center crop to maintain aspect ratio
        image = self._transform_image(image)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)  # Convert to C,H,W format
        
        # Encode text prompt
        text_inputs = self.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        
        return {
            "pixel_values": image,
            "text_embeddings": text_embeddings[0],
        }
    
    def _transform_image(self, image):
        # Resize to maintain aspect ratio
        ratio = min(self.size / image.width, self.size / image.height)
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        left = (new_width - self.size) // 2
        top = (new_height - self.size) // 2
        right = left + self.size
        bottom = top + self.size
        
        # Handle images smaller than target size
        if new_width < self.size or new_height < self.size:
            # Create a black canvas of the target size
            new_image = Image.new("RGBA", (self.size, self.size))
            # Paste the resized image in the center
            paste_left = (self.size - new_width) // 2
            paste_top = (self.size - new_height) // 2
            new_image.paste(image, (paste_left, paste_top))
            return new_image
        
        return image.crop((left, top, right, bottom))

def finetune_lora(style_folder, output_folder, base_model="runwayml/stable-diffusion-v1-5", num_epochs=50):
    print(f"Starting fine-tuning  with images from {style_folder}")
    
    # Get all image files
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(glob.glob(os.path.join(style_folder, f"*.{ext}")))
    
    if not image_paths:
        raise ValueError(f"No images found in {style_folder}")
    
    print(f"Found {len(image_paths)} images")
    
    # Load model components
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.train()
    
    # Training parameters
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    
    # Create dataset and dataloader
    dataset = StyleImageDataset(
        image_paths=image_paths,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt="an image in a specific artistic style"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, 
        beta_end=0.012, 
        beta_schedule="scaled_linear", 
        num_train_timesteps=1000
    )
    
    # Fine-tuning loop
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    device = "cpu"
    unet.to(device)
    text_encoder.to(device)
    
    print(f"Training on {device} for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Get image and text embeddings
            clean_images = batch["pixel_values"].to(device)
            text_embeddings = batch["text_embeddings"].to(device)
            
            # Add noise to the clean images
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Predict the noise
            noise_pred = unet(noisy_images, timesteps, text_embeddings).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch loss
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_folder, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(unet.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save the final LoRA model
    final_lora_path = os.path.join(output_folder, "lora_weights.safetensors")
    unet.save_pretrained(output_folder)
    
    print(f"LoRA fine-tuning complete. Model saved to {output_folder}")
    
    # Create a metadata file
    metadata = {
        "model_type": "LoRA",
        "base_model": base_model,
        "training_images": len(image_paths),
        "epochs": num_epochs,
        "timestamp": time.time()
    }
    
    with open(os.path.join(output_folder, "lora_metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    return os.path.join(output_folder, "adapter_model.bin")

if __name__ == "__main__":
    # For testing the module independently
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Fine-tune a LoRA model on a set of style images")
    parser.add_argument("--style_folder", type=str, required=True, help="Folder containing style images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the LoRA model")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    args = parser.parse_args()
    
    finetune_lora(args.style_folder, args.output_folder, args.base_model, args.epochs)