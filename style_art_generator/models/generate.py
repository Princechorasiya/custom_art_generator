# style_art_generator/models/generate.py
import os
import sys
import json
import requests
import time
import base64
from PIL import Image
import io
import subprocess
import tempfile

# Path to ComfyUI installation (adjust as needed)
COMFYUI_PATH = "../../ComfyUI"
COMFYUI_SERVER_ADDRESS = "http://127.0.0.1:8188"

def start_comfyui_server():
    """Start the ComfyUI server if it's not already running"""
    try:
        # Check if server is already running
        response = requests.get(f"{COMFYUI_SERVER_ADDRESS}/system_stats")
        if response.status_code == 200:
            print("ComfyUI server is already running")
            return True
    except requests.exceptions.ConnectionError:
        print("Starting ComfyUI server...")
        
        # Launch ComfyUI in a separate process
        comfyui_process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=COMFYUI_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start (up to 30 seconds)
        max_attempts = 30
        for i in range(max_attempts):
            try:
                response = requests.get(f"{COMFYUI_SERVER_ADDRESS}/system_stats")
                if response.status_code == 200:
                    print("ComfyUI server started successfully")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
            
        print("Failed to start ComfyUI server")
        return False

def create_comfyui_workflow(lora_path, prompt, negative_prompt, guidance_scale, steps):
    """Create a ComfyUI workflow JSON for image generation with LoRA"""
    
    # Adjust the relative path to be absolute for ComfyUI
    abs_lora_path = os.path.abspath(lora_path)
    
    # Create the workflow
    workflow = {
        "3": {
            "inputs": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "style": "artistic",
                "performance": "Quality",
                "empty_latent_width": 512,
                "empty_latent_height": 512,
                "batch_size": 1
            },
            "class_type": "KSampler (Efficient)",
            "_meta": {
                "title": "KSampler Efficient"
            }
        },
        "4": {
            "inputs": {
                "ckpt_name": "v1-5-pruned-emaonly.safetensors"
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": "Load Checkpoint"
            }
        },
        "5": {
            "inputs": {
                "lora_name": abs_lora_path,
                "strength_model": 1.0,
                "strength_clip": 1.0,
                "model": ["4", 0],
                "clip": ["4", 1]
            },
            "class_type": "LoraLoader",
            "_meta": {
                "title": "Load LoRA"
            }
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "Decode VAE"
            }
        },
        "9": {
            "inputs": {
                "filename_prefix": "styled_image",
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "Save Image"
            }
        }
    }
    
    # Update the workflow with user parameters
    workflow["3"]["inputs"]["seed"] = int(time.time()) % 1000000000  # Random seed
    workflow["3"]["inputs"]["steps"] = steps
    workflow["3"]["inputs"]["cfg"] = guidance_scale
    workflow["3"]["inputs"]["sampler_name"] = "euler_ancestral"
    workflow["3"]["inputs"]["scheduler"] = "normal"
    workflow["3"]["inputs"]["denoise"] = 1.0
    workflow["3"]["inputs"]["model"] = ["5", 0]
    workflow["3"]["inputs"]["positive"] = ["5", 1]
    workflow["3"]["inputs"]["negative"] = ["5", 2]
    
    return workflow

def queue_comfyui_prompt(workflow):
    """Submit a workflow to ComfyUI and wait for the result"""
    prompt_url = f"{COMFYUI_SERVER_ADDRESS}/prompt"
    
    p = {"prompt": workflow}
    response = requests.post(prompt_url, json=p)
    
    if response.status_code != 200:
        raise Exception(f"Failed to queue prompt: {response.text}")
    
    data = response.json()
    prompt_id = data.get("prompt_id")
    
    if not prompt_id:
        raise Exception("No prompt_id received from ComfyUI")
    
    return prompt_id

def wait_for_comfyui_execution(prompt_id):
    """Wait for ComfyUI to finish processing the prompt"""
    history_url = f"{COMFYUI_SERVER_ADDRESS}/history/{prompt_id}"
    
    max_attempts = 60  # Wait up to 60 seconds
    for i in range(max_attempts):
        response = requests.get(history_url)
        
        if response.status_code == 200:
            data = response.json()
            if data.get(prompt_id, {}).get("status", {}).get("completed", False):
                return data[prompt_id]
        
        time.sleep(1)
    
    raise Exception("Timed out waiting for ComfyUI to process the prompt")

def download_comfyui_result(history_data, output_path):
    """Download the generated image from ComfyUI"""
    # Get the node ID of the SaveImage node
    for node_id, node_data in history_data["outputs"].items():
        if "images" in node_data:
            image_data = node_data["images"][0]
            image_filename = image_data["filename"]
            image_subfolder = image_data.get("subfolder", "")
            
            # Construct the URL to download the image
            image_url = f"{COMFYUI_SERVER_ADDRESS}/view"
            params = {"filename": image_filename, "subfolder": image_subfolder, "type": "output"}
            
            response = requests.get(image_url, params=params, stream=True)
            
            if response.status_code == 200:
                # Save the image to the specified output path
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                return output_path
            else:
                raise Exception(f"Failed to download image: {response.text}")
    
    raise Exception("No image found in ComfyUI response")

def generate_with_comfyui(lora_path, prompt, negative_prompt, guidance_scale, steps, output_path):
    """Generate an image using ComfyUI with the specified LoRA model"""
    
    # Make sure ComfyUI server is running
    if not start_comfyui_server():
        raise Exception("Failed to start ComfyUI server")
    
    # Create the workflow
    workflow = create_comfyui_workflow(
        lora_path=lora_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        steps=steps
    )
    
    # Queue the prompt
    prompt_id = queue_comfyui_prompt(workflow)
    
    # Wait for execution to complete
    history_data = wait_for_comfyui_execution(prompt_id)
    
    # Download the result
    download_comfyui_result(history_data, output_path)
    
    return output_path

if __name__ == "__main__":
    # For testing the module independently
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate an image with a LoRA model using ComfyUI")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA model")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale (CFG)")
    parser.add_argument("--steps", type=int, default=30, help="Number of sampling steps")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated image")
    
    args = parser.parse_args()
    
    generate_with_comfyui(
        args.lora_path,
        args.prompt,
        args.negative_prompt,
        args.guidance_scale,
        args.steps,
        args.output_path
    )