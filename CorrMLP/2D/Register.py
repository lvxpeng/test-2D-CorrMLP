import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image # Using Pillow for image loading/saving
import torch
import torch.nn.functional as F

# Assuming your network module is accessible
import network2D

# --- Helper Function for Loading and Preprocessing ---

def load_and_preprocess_image(path, device):
    """
    Loads a PNG image, converts to grayscale, normalizes to [0, 1],
    adds batch/channel dims, and returns a tensor on the specified device.
    """
    try:
        img = Image.open(path).convert('L') # Load and ensure grayscale
        img_np = np.array(img)

        # Normalize to [0, 1] and convert to float tensor
        img_np = img_np.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np)

        # Add batch (B=1) and channel (C=1) dimensions -> (1, 1, H, W)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # Move to target device
        tensor = tensor.to(device)
        return tensor

    except FileNotFoundError:
        print(f"Error: Image file not found at {path}")
        return None
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

# --- Main Registration Function ---

def register_images(input_path, model_path, atlas_path, save_path, device_str):
    """
    Performs registration using a trained model.

    Args:
        input_path (str): Path to the directory containing moving images (.png).
        model_path (str): Path to the trained model (.pt file).
        atlas_path (str): Path to the fixed atlas image (.png).
        save_path (str): Path to the directory where registered images will be saved.
        device_str (str): Device to use ('cpu', 'gpu0', 'gpu1', etc.).
    """
    print("--- Starting Registration ---")

    # --- 1. Setup Device ---
    if 'gpu' in device_str and torch.cuda.is_available():
        gpu_id = device_str.split('gpu')[-1]
        if not gpu_id: gpu_id = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {device}")
    else:
        if 'gpu' in device_str:
            print("CUDA specified but not available. Using CPU.")
        device = torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Using CPU.")

    # --- 2. Load Fixed Atlas ---
    print(f"Loading fixed atlas from: {atlas_path}")
    fixed_atlas_tensor = load_and_preprocess_image(atlas_path, device)
    if fixed_atlas_tensor is None:
        print("Exiting due to atlas loading error.")
        return
    print(f"Atlas loaded successfully. Shape: {fixed_atlas_tensor.shape}")

    # --- 3. Load Model ---
    print(f"Loading model from: {model_path}")
    # Instantiate model (make sure parameters match the trained model)
    model = network2D.CorrMLP(in_channels=1) # Assuming 1 input channel
    try:
        state_dict = torch.load(model_path, map_location=device,weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Set model to evaluation mode (important!)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture matches the saved weights.")
        return

    # --- 4. Load Spatial Transformer ---
    # Use 'bilinear' for smoother intensity image warping
    spatial_transformer = network2D.SpatialTransformer_block(mode='bilinear').to(device)
    spatial_transformer.eval()

    # --- 5. Prepare Output Directory ---
    if not os.path.exists(save_path):
        print(f"Creating output directory: {save_path}")
        os.makedirs(save_path)
    else:
        print(f"Output directory exists: {save_path}")

    # --- 6. Find and Process Input Images ---
    image_paths = sorted(glob.glob(os.path.join(input_path, '*.png')))
    if not image_paths:
        print(f"Error: No .png images found in input directory: {input_path}")
        return

    print(f"Found {len(image_paths)} images to register in {input_path}")

    with torch.no_grad(): # Disable gradient calculations for inference
        for i, img_path in enumerate(image_paths):
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")

            # Load moving image
            moving_image_tensor = load_and_preprocess_image(img_path, device)
            if moving_image_tensor is None:
                print(f"Skipping image {img_path} due to loading error.")
                continue

            # --- Perform Registration ---
            # Model expects (fixed, moving) and returns (warped_moving, flow)
            # We need the flow to warp the original moving image
            _, flow_pred = model(fixed_atlas_tensor, moving_image_tensor)

            # Warp the original moving image using the predicted flow
            # Input to transformer: (source_image, flow)
            warped_moving_tensor = spatial_transformer(moving_image_tensor, flow_pred)

            # --- Post-process and Save ---
            # Remove batch and channel dimensions -> (H, W)
            warped_image_tensor_squeezed = warped_moving_tensor.squeeze()

            # Move to CPU and convert to NumPy array
            warped_image_np = warped_image_tensor_squeezed.cpu().numpy()

            # Rescale from [0, 1] back to [0, 255] and convert to uint8
            warped_image_np = (warped_image_np * 255.0).clip(0, 255).astype(np.uint8)

            # Create output filename
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            save_name = f"{name}_registered.png"
            output_filepath = os.path.join(save_path, save_name)

            # Save the warped image
            try:
                Image.fromarray(warped_image_np).save(output_filepath)
                print(f"Saved registered image to: {output_filepath}")
            except Exception as e:
                print(f"Error saving image {output_filepath}: {e}")

    print("\n--- Registration Finished ---")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register input images to a fixed atlas using a trained model.")

    # parser.add_argument("--input_path", type=str, required=True,
    #                     help="Path to the directory containing input .png images to register.")
    # parser.add_argument("--model_path", type=str,  required=True,
    #                     help="Path to the trained registration model (.pt file).")
    # parser.add_argument("--atlas_path", type=str,  required=True,
    #                     help="Path to the fixed atlas .png image.")
    # parser.add_argument("--save_path", type=str,  required=True,
    #                     help="Path to the directory where registered images will be saved.")
    # parser.add_argument("--device", type=str, default='gpu0',
    #                     help="Device to use ('cpu', 'gpu0', 'gpu1', etc.)")
    parser.add_argument("--input_path", type=str, default='./test/moved/',
                        help="Path to the directory containing input .png images to register.")
    parser.add_argument("--model_path", type=str, default='./models/epoch_081.pt',
                        help="Path to the trained registration model (.pt file).")
    parser.add_argument("--atlas_path", type=str, default='./data/fixed/atlas.png',
                        help="Path to the fixed atlas .png image.")
    parser.add_argument("--save_path", type=str, default='./test/saved/',
                        help="Path to the directory where registered images will be saved.")
    parser.add_argument("--device", type=str, default='gpu0',
                        help="Device to use ('cpu', 'gpu0', 'gpu1', etc.)")

    args = parser.parse_args()

    register_images(
        input_path=args.input_path,
        model_path=args.model_path,
        atlas_path=args.atlas_path,
        save_path=args.save_path,
        device_str=args.device
    )
    # register_images(**vars(args))