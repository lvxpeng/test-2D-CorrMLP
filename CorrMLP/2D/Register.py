# import os
# import sys
# import glob
# import argparse
# import numpy as np
# from PIL import Image # Using Pillow for image loading/saving
# import torch
# import torch.nn.functional as F
#
# # Assuming your network module is accessible
# import network2D
#
# # --- Helper Function for Loading and Preprocessing ---
#
# def load_and_preprocess_image(path, device):
#     """
#     Loads a PNG image, converts to grayscale, normalizes to [0, 1],
#     adds batch/channel dims, and returns a tensor on the specified device.
#     """
#     try:
#         img = Image.open(path).convert('L') # Load and ensure grayscale
#         img_np = np.array(img)
#
#         # Normalize to [0, 1] and convert to float tensor
#         img_np = img_np.astype(np.float32) / 255.0
#         tensor = torch.from_numpy(img_np)
#
#         # Add batch (B=1) and channel (C=1) dimensions -> (1, 1, H, W)
#         tensor = tensor.unsqueeze(0).unsqueeze(0)
#
#         # Move to target device
#         tensor = tensor.to(device)
#         return tensor
#
#     except FileNotFoundError:
#         print(f"Error: Image file not found at {path}")
#         return None
#     except Exception as e:
#         print(f"Error loading image {path}: {e}")
#         return None
#
# # --- Main Registration Function ---
#
# def register_images(input_path, model_path, atlas_path, save_path, device_str):
#     """
#     Performs registration using a trained model.
#
#     Args:
#         input_path (str): Path to the directory containing moving images (.png).
#         model_path (str): Path to the trained model (.pt file).
#         atlas_path (str): Path to the fixed atlas image (.png).
#         save_path (str): Path to the directory where registered images will be saved.
#         device_str (str): Device to use ('cpu', 'gpu0', 'gpu1', etc.).
#     """
#     print("--- Starting Registration ---")
#
#     # --- 1. Setup Device ---
#     if 'gpu' in device_str and torch.cuda.is_available():
#         gpu_id = device_str.split('gpu')[-1]
#         if not gpu_id: gpu_id = '0'
#         os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
#         device = torch.device(f'cuda:{gpu_id}')
#         print(f"Using GPU: {device}")
#     else:
#         if 'gpu' in device_str:
#             print("CUDA specified but not available. Using CPU.")
#         device = torch.device('cpu')
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#         print("Using CPU.")
#
#     # --- 2. Load Fixed Atlas ---
#     print(f"Loading fixed atlas from: {atlas_path}")
#     fixed_atlas_tensor = load_and_preprocess_image(atlas_path, device)
#     if fixed_atlas_tensor is None:
#         print("Exiting due to atlas loading error.")
#         return
#     print(f"Atlas loaded successfully. Shape: {fixed_atlas_tensor.shape}")
#
#     # --- 3. Load Model ---
#     print(f"Loading model from: {model_path}")
#     # Instantiate model (make sure parameters match the trained model)
#     model = network2D.CorrMLP(in_channels=1) # Assuming 1 input channel
#     try:
#         state_dict = torch.load(model_path, map_location=device,weights_only=True)
#         model.load_state_dict(state_dict)
#         model.to(device)
#         model.eval() # Set model to evaluation mode (important!)
#         print("Model loaded successfully.")
#     except FileNotFoundError:
#         print(f"Error: Model file not found at {model_path}")
#         return
#     except Exception as e:
#         print(f"Error loading model state_dict: {e}")
#         print("Ensure the model architecture matches the saved weights.")
#         return
#
#     # --- 4. Load Spatial Transformer ---
#     # Use 'bilinear' for smoother intensity image warping
#     spatial_transformer = network2D.SpatialTransformer_block(mode='bilinear').to(device)
#     spatial_transformer.eval()
#
#     # --- 5. Prepare Output Directory ---
#     if not os.path.exists(save_path):
#         print(f"Creating output directory: {save_path}")
#         os.makedirs(save_path)
#     else:
#         print(f"Output directory exists: {save_path}")
#
#     # --- 6. Find and Process Input Images ---
#     image_paths = sorted(glob.glob(os.path.join(input_path, '*.png')))
#     if not image_paths:
#         print(f"Error: No .png images found in input directory: {input_path}")
#         return
#
#     print(f"Found {len(image_paths)} images to register in {input_path}")
#
#     with torch.no_grad(): # Disable gradient calculations for inference
#         for i, img_path in enumerate(image_paths):
#             print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
#
#             # Load moving image
#             moving_image_tensor = load_and_preprocess_image(img_path, device)
#             if moving_image_tensor is None:
#                 print(f"Skipping image {img_path} due to loading error.")
#                 continue
#
#             # --- Perform Registration ---
#             # Model expects (fixed, moving) and returns (warped_moving, flow)
#             # We need the flow to warp the original moving image
#             _, flow_pred = model(fixed_atlas_tensor, moving_image_tensor)
#
#             # Warp the original moving image using the predicted flow
#             # Input to transformer: (source_image, flow)
#             warped_moving_tensor = spatial_transformer(moving_image_tensor, flow_pred)
#
#             # --- Post-process and Save ---
#             # Remove batch and channel dimensions -> (H, W)
#             warped_image_tensor_squeezed = warped_moving_tensor.squeeze()
#
#             # Move to CPU and convert to NumPy array
#             warped_image_np = warped_image_tensor_squeezed.cpu().numpy()
#
#             # Rescale from [0, 1] back to [0, 255] and convert to uint8
#             warped_image_np = (warped_image_np * 255.0).clip(0, 255).astype(np.uint8)
#
#             # Create output filename
#             base_name = os.path.basename(img_path)
#             name, ext = os.path.splitext(base_name)
#             save_name = f"{name}_registered.png"
#             output_filepath = os.path.join(save_path, save_name)
#
#             # Save the warped image
#             try:
#                 Image.fromarray(warped_image_np).save(output_filepath)
#                 print(f"Saved registered image to: {output_filepath}")
#             except Exception as e:
#                 print(f"Error saving image {output_filepath}: {e}")
#
#     print("\n--- Registration Finished ---")
#
# # --- Command Line Argument Parsing ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Register input images to a fixed atlas using a trained model.")
#
#     # parser.add_argument("--input_path", type=str, required=True,
#     #                     help="Path to the directory containing input .png images to register.")
#     # parser.add_argument("--model_path", type=str,  required=True,
#     #                     help="Path to the trained registration model (.pt file).")
#     # parser.add_argument("--atlas_path", type=str,  required=True,
#     #                     help="Path to the fixed atlas .png image.")
#     # parser.add_argument("--save_path", type=str,  required=True,
#     #                     help="Path to the directory where registered images will be saved.")
#     # parser.add_argument("--device", type=str, default='gpu0',
#     #                     help="Device to use ('cpu', 'gpu0', 'gpu1', etc.)")
#     parser.add_argument("--input_path", type=str, default='./test/moved/',
#                         help="Path to the directory containing input .png images to register.")
#     parser.add_argument("--model_path", type=str, default='./models/best_model.pt',
#                         help="Path to the trained registration model (.pt file).")
#     parser.add_argument("--atlas_path", type=str, default='./data/fixed/atlas.nii.gz',
#                         help="Path to the fixed atlas .png image.")
#     parser.add_argument("--save_path", type=str, default='./test/saved/',
#                         help="Path to the directory where registered images will be saved.")
#     parser.add_argument("--device", type=str, default='gpu0',
#                         help="Device to use ('cpu', 'gpu0', 'gpu1', etc.)")
#
#     args = parser.parse_args()
#
#     register_images(
#         input_path=args.input_path,
#         model_path=args.model_path,
#         atlas_path=args.atlas_path,
#         save_path=args.save_path,
#         device_str=args.device
#     )
#     # register_images(**vars(args))
# Register.py

# Register.py (using original datagenerators2D)

# Register.py (NIfTI-only version, no datagenerators2D)

import os
import sys
import argparse
import logging
import torch
import numpy as np
from PIL import Image
import nibabel as nib # Use nibabel directly

# Import your network module
import network2D

# --- Setup Logging ---
def setup_logging(log_dir):
    """Configures logging to file and console."""
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'register_nifti_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging to {log_file}")

# --- Helper Function to Load/Preprocess NIfTI ---
def load_nifti_as_tensor(path, is_label=False, device='cpu'):
    """Loads NIfTI, preprocesses, returns tensor and affine."""
    try:
        if not os.path.exists(path):
            logging.error(f"NIfTI file not found: {path}")
            return None, None

        img_nib = nib.load(path)
        affine = img_nib.affine
        data_np = img_nib.get_fdata() # Load as float initially

        # Ensure data is 2D
        if data_np.ndim > 2:
            logging.warning(f"Input NIfTI '{os.path.basename(path)}' has >2 dimensions ({data_np.shape}). Squeezing.")
            data_np = np.squeeze(data_np)
            if data_np.ndim != 2:
                logging.error(f"Could not squeeze NIfTI '{os.path.basename(path)}' to 2D. Final shape: {data_np.shape}")
                return None, None

        # Preprocessing
        if is_label:
            # Convert label to float32 for warping, add channel dim -> (1, H, W)
            data_np = data_np.astype(np.float32)[np.newaxis, :, :]
        else:
            # Normalize image to [0, 1], add channel dim -> (1, H, W)
            data_np = data_np.astype(np.float32)
            min_val, max_val = np.min(data_np), np.max(data_np)
            if max_val > min_val: data_np = (data_np - min_val) / (max_val - min_val)
            elif max_val > 0: data_np = data_np / max_val
            data_np = data_np[np.newaxis, :, :]

        # Convert to tensor, add batch dim -> (1, 1, H, W)
        tensor = torch.from_numpy(data_np).unsqueeze(0).to(device)

        return tensor, affine

    except Exception as e:
        logging.error(f"Failed to load or process NIfTI file {path}: {e}", exc_info=True)
        return None, None

# --- Helper Function to Save Tensor as NIfTI ---
# Using the robust version from previous iteration
def save_tensor_as_nifti(tensor, output_path, affine=None, force_dtype=None):
    """
    Saves a PyTorch tensor as a NIfTI file, enforcing the final dtype.
    """
    if tensor is None:
        logging.warning(f"Attempted to save None tensor to {output_path}. Skipping.")
        return

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 1. Prepare data
        initial_np_dtype = np.float32 if tensor.dtype == torch.float32 else tensor.numpy().dtype
        data_np = tensor.detach().cpu().numpy().astype(initial_np_dtype)
        # 2. Squeeze and Transpose
        original_shape = data_np.shape
        if data_np.ndim == 4:
            data_np = data_np.squeeze(0)
            if data_np.shape[0] > 1: data_np = np.transpose(data_np, (1, 2, 0))
            else: data_np = data_np.squeeze(0)
        elif data_np.ndim == 3:
             if data_np.shape[0] <= 4 and data_np.shape[0] > 1 : data_np = np.transpose(data_np, (1, 2, 0))
             elif data_np.shape[-1] <= 4 and data_np.shape[-1] > 1: pass
             elif data_np.shape[0] == 1: data_np = data_np.squeeze(0)
             elif data_np.shape[-1] == 1: data_np = data_np.squeeze(-1)
        if data_np.ndim < 2:
            logging.error(f"Cannot save tensor with final shape {data_np.shape} as NIfTI for {output_path} (original: {original_shape}).")
            return
        # 3. Determine Target Data Type
        target_dtype = force_dtype
        rescale_to_uint8 = False
        if target_dtype is None:
            if 'flow' in output_path.lower(): target_dtype = np.float32
            elif 'label' in output_path.lower(): target_dtype = np.int16
            elif 'warped' in output_path.lower() and 'label' not in output_path.lower():
                 if data_np.min() >= -0.01 and data_np.max() <= 1.01:
                     target_dtype = np.uint8; rescale_to_uint8 = True
                 else: target_dtype = np.float32
            else: target_dtype = np.float32
        # 4. Apply Rescaling
        if rescale_to_uint8: data_np = (data_np.clip(0, 1) * 255.0)
        # 5. Apply Final Data Type Conversion
        try: final_data = data_np.astype(target_dtype, casting='unsafe')
        except TypeError as te: logging.error(f"TypeError during final astype to {target_dtype} for {output_path}: {te}"); return
        # 6. Create default affine
        if affine is None: affine = np.eye(4); logging.warning(f"Using identity affine for {output_path}.")
        # 7. Create NIfTI image object
        nifti_img = nib.Nifti1Image(final_data, affine)
        try:
            nifti_img.header.set_data_dtype(target_dtype); nifti_img.header['scl_slope'] = 1.0; nifti_img.header['scl_inter'] = 0.0
        except Exception as hdr_e: logging.error(f"Error setting header fields for {output_path}: {hdr_e}")
        # 8. Save the image
        nib.save(nifti_img, output_path)
        logging.info(f"Attempted save: {output_path} (Shape: {final_data.shape}, Target Dtype: {target_dtype})")
        # 9. Verification Step (using dataobj)
        # ... (Verification logic can be kept or removed) ...

    except Exception as e:
        logging.error(f"Failed to save tensor to {output_path}: {e}", exc_info=True)


# --- Main Registration Function ---
def register_nifti_pair(
    fixed_nifti_path,
    moving_nifti_path,
    output_dir,
    model_path,
    moving_label_nifti_path=None, # Optional moving label
    device='gpu0',
    log_dir='./log/'
    ):
    """Performs registration for a single pair of NIfTI images."""

    setup_logging(log_dir)
    logging.info("--- Starting NIfTI Registration Process ---")
    logging.info(f"Fixed Image: {fixed_nifti_path}")
    logging.info(f"Moving Image: {moving_nifti_path}")
    logging.info(f"Output Dir: {output_dir}")
    logging.info(f"Model Path: {model_path}")
    if moving_label_nifti_path: logging.info(f"Moving Label: {moving_label_nifti_path}")

    # 1. Setup Device
    if 'gpu' in device and torch.cuda.is_available():
        gpu_id = device.split('gpu')[-1] or '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        dev = torch.device(f'cuda:{gpu_id}')
        logging.info(f"Using GPU: {dev}")
    else:
        if 'gpu' in device: logging.warning("CUDA not available. Using CPU.")
        dev = torch.device('cpu')
        logging.info("Using CPU.")

    # 2. Load Data using Nibabel Helper
    logging.info("Loading NIfTI images...")
    fixed_tensor, fixed_affine = load_nifti_as_tensor(fixed_nifti_path, is_label=False, device=dev)
    moving_tensor, _ = load_nifti_as_tensor(moving_nifti_path, is_label=False, device=dev) # Affine of moving not needed

    if fixed_tensor is None or moving_tensor is None:
        logging.error("Failed to load one or both NIfTI images. Exiting.")
        return
    if fixed_affine is None:
        logging.warning("Failed to get affine from fixed image. Using identity matrix for saving.")
        fixed_affine = np.eye(4)

    logging.info(f"Fixed image tensor shape: {fixed_tensor.shape}")
    logging.info(f"Moving image tensor shape: {moving_tensor.shape}")

    # Load moving label if path provided
    moving_label_tensor = None
    if moving_label_nifti_path:
        moving_label_tensor, _ = load_nifti_as_tensor(moving_label_nifti_path, is_label=True, device=dev)
        if moving_label_tensor is not None:
            logging.info(f"Moving label tensor shape: {moving_label_tensor.shape}")
        else: logging.warning(f"Could not load moving label: {moving_label_nifti_path}")

    # 3. Load Model
    logging.info(f"Loading model from {model_path}...")
    try:
        model = network2D.CorrMLP(in_channels=1) # Assuming grayscale input
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(dev)
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        return

    # 4. Perform Registration
    logging.info("Performing registration...")
    spatial_transformer = None
    if moving_label_tensor is not None:
         spatial_transformer = network2D.SpatialTransformer_block(mode='nearest').to(dev)
         spatial_transformer.eval()

    warped_moving_pred = None
    flow_pred = None
    warped_label_pred = None
    with torch.no_grad():
        try:
            warped_moving_pred, flow_pred = model(fixed_tensor, moving_tensor)
            logging.info("Registration complete.")
            if moving_label_tensor is not None and spatial_transformer is not None:
                logging.info("Warping moving label...")
                warped_label_pred = spatial_transformer(moving_label_tensor, flow_pred)
                logging.info("Label warping complete.")
        except Exception as e:
             logging.error(f"Error during model inference or warping: {e}", exc_info=True)
             return

    # 5. Save Results
    logging.info(f"Saving results to {output_dir}...")
    moving_basename = os.path.basename(moving_nifti_path)
    if moving_basename.lower().endswith('.nii.gz'):
        moving_basename = moving_basename[:-7] # Remove .nii.gz
    elif moving_basename.lower().endswith('.nii'):
         moving_basename = moving_basename[:-4] # Remove .nii

    logging.info(f"Saving results to {output_dir}...")

    # --- Determine Basename ---
    moving_basename = os.path.basename(moving_nifti_path)
    if moving_basename.lower().endswith('.nii.gz'):
        moving_basename = moving_basename[:-7] # Remove .nii.gz
    elif moving_basename.lower().endswith('.nii'):
         moving_basename = moving_basename[:-4] # Remove .nii
    else: # Fallback if extension wasn't .nii or .nii.gz
         moving_basename = os.path.splitext(moving_basename)[0]


    # --- Define Output Paths ---
    # Keep NIfTI for flow and potentially warped label (as they need affine)
    out_flow_path = os.path.join(output_dir, f"{moving_basename}_flow.nii.gz")
    out_warped_label_path = os.path.join(output_dir, f"{moving_basename}_label_warped.nii.gz")
    # Define PNG path for the warped image
    out_warped_img_path_png = os.path.join(output_dir, f"{moving_basename}_warped.png") # <-- PNG Extension


    # --- Save Warped Image as PNG ---
    if warped_moving_pred is not None:
        try:
            # 1. Detach, move to CPU, convert to NumPy
            warped_np = warped_moving_pred.detach().cpu().numpy()
            # 2. Squeeze batch and channel dims -> (H, W)
            if warped_np.ndim == 4: warped_np = warped_np.squeeze((0, 1))
            elif warped_np.ndim == 3: warped_np = warped_np.squeeze(0) # Assume (C=1, H, W)
            elif warped_np.ndim != 2: raise ValueError(f"Unexpected shape after squeezing: {warped_np.shape}")

            # 3. Rescale data from [0, 1] to [0, 255] for standard PNG
            if warped_np.min() >= -0.01 and warped_np.max() <= 1.01: # Check if normalized
                 img_data_uint8 = (warped_np.clip(0, 1) * 255.0).astype(np.uint8)
                 logging.info(f"Saving warped image as PNG (rescaled to uint8): {out_warped_img_path_png}")
            else: # If not normalized, save as is (might need viewer adjustment)
                 img_data_uint8 = warped_np.astype(np.uint8) # Or potentially save as 16-bit PNG if needed
                 logging.warning(f"Warped image data not in [0,1] range. Saving as PNG directly (dtype={img_data_uint8.dtype}). Visualization might require intensity adjustment.")

            # 4. Create Pillow Image object
            pil_img = Image.fromarray(img_data_uint8, mode='L') # 'L' mode for grayscale

            # 5. Ensure output directory exists
            os.makedirs(os.path.dirname(out_warped_img_path_png), exist_ok=True)

            # 6. Save the PNG image
            pil_img.save(out_warped_img_path_png)
            logging.info(f"Successfully saved warped image: {out_warped_img_path_png}")

        except Exception as e:
            logging.error(f"Failed to save warped image as PNG: {e}", exc_info=True)
    else:
        logging.warning("Warped image tensor is None. Skipping PNG save.")


    # --- Save Flow Field as NIfTI (still needs affine) ---
    # Keep using the robust save_tensor_as_nifti for flow
    save_tensor_as_nifti(flow_pred, out_flow_path, affine=fixed_affine)


    # --- Save Warped Label as NIfTI (optional, still needs affine) ---
    if warped_label_pred is not None:
         # Keep using the robust save_tensor_as_nifti for labels
         save_tensor_as_nifti(warped_label_pred, out_warped_label_path, affine=fixed_affine)

    logging.info("--- Registration Process Finished ---")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a pair of 2D NIfTI images using a trained CorrMLP model.")

    # Required arguments
    parser.add_argument("--fixed_nifti", type=str, default='./data/fixed/atlas.nii.gz', help="Path to the fixed NIfTI image file (.nii.gz or .nii).")
    parser.add_argument("--moving_nifti", type=str, default='./test/moved/12_slice_norm.nii.gz', help="Path to the moving NIfTI image file (.nii.gz or .nii).")
    parser.add_argument("--output_dir", type=str,default='./test/saved/',  help="Directory to save the registration results.")
    parser.add_argument("--model_path", type=str, default='./models/best_model.pt', help="Path to the trained model checkpoint (.pt file).")

    # Optional arguments
    parser.add_argument("--moving_label", type=str, default=None, help="Path to the moving label NIfTI file (optional, will be warped).")
    parser.add_argument("--device", type=str, default='gpu0', help="Device to use ('cpu' or 'gpuN'). Default: gpu0.")
    parser.add_argument("--log_dir", type=str, default='./log/', help="Directory for the registration log file.")

    args = parser.parse_args()

    # Basic input validation
    if not args.fixed_nifti.lower().endswith(('.nii.gz', '.nii')):
        print(f"Error: Fixed image '{args.fixed_nifti}' does not appear to be a NIfTI file.")
        sys.exit(1)
    if not args.moving_nifti.lower().endswith(('.nii.gz', '.nii')):
        print(f"Error: Moving image '{args.moving_nifti}' does not appear to be a NIfTI file.")
        sys.exit(1)
    if args.moving_label and not args.moving_label.lower().endswith(('.nii.gz', '.nii')):
         print(f"Error: Moving label '{args.moving_label}' does not appear to be a NIfTI file.")
         sys.exit(1)


    # Run the registration
    register_nifti_pair(
        fixed_nifti_path=args.fixed_nifti,
        moving_nifti_path=args.moving_nifti,
        output_dir=args.output_dir,
        model_path=args.model_path,
        moving_label_nifti_path=args.moving_label,
        device=args.device,
        log_dir=args.log_dir
    )