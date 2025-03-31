import os
import sys
import time
import glob
import random
import logging # <-- Import logging module
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser

# Assuming your network and loss files are structured like this
import network2D  # Contains CorrMLP, SpatialTransformer_block
import losses2D   # Contains NCC, Grad

# --- Helper Function for Loading and Preprocessing PNG ---
# ... (load_and_preprocess_image function remains the same) ...
def load_and_preprocess_image(path, device, is_segmentation=False):
    """Loads a PNG image, converts to grayscale, normalizes, and returns a tensor."""
    try:
        img = Image.open(path).convert('L') # Load and convert to grayscale
        img_np = np.array(img)

        if is_segmentation:
            # For segmentations, keep integer values, add batch/channel dims
            # Ensure segmentation labels are appropriate for 'nearest' interpolation
            tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device).float() # Use float for transformer
        else:
            # For intensity images, normalize to [0, 1], add batch/channel dims
            img_np = img_np.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device).float()

        return tensor
    except FileNotFoundError:
        logging.error(f"Image file not found at {path}") # <-- Log errors
        return None
    except Exception as e:
        logging.error(f"Error loading image {path}: {e}") # <-- Log errors
        return None

# --- Metrics (Copied and potentially refined from previous version) ---
# ... (Dice function remains the same) ...
def Dice(vol1, vol2, labels=None, nargout=1):
    """
    Calculate the Dice Similarity Coefficient for segmentation masks.
    Assumes vol1 and vol2 are numpy arrays of the same shape.
    """
    # Ensure inputs are numpy arrays
    if isinstance(vol1, torch.Tensor):
        vol1 = vol1.detach().cpu().numpy()
    if isinstance(vol2, torch.Tensor):
        vol2 = vol2.detach().cpu().numpy()

    if labels is None:
        # Consider all unique non-zero values as labels
        labels = np.unique(np.concatenate((vol1.flatten(), vol2.flatten())))
        labels = np.delete(labels, np.where(labels == 0)) # remove background

    if len(labels) == 0:
        # Handle case where only background is present
        # If both inputs are all zeros, Dice is 1, otherwise 0? Or return NaN? Let's return 1 if both empty bg.
         return 1.0 if np.all(vol1 == 0) and np.all(vol2 == 0) else 0.0

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = (vol1 == lab)
        vol2l = (vol2 == lab)
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)

        if bottom == 0:
            # If both segmentations are empty for this label, Dice is 1 if top is also 0, else 0
            dicem[idx] = 1.0 if top == 0 else 0.0
        else:
             dicem[idx] = top / bottom

    # Return the average Dice score across all foreground labels
    if nargout == 1:
        return np.mean(dicem)
    else:
        return (dicem, labels)

# ... (NJD function remains the same, added logging) ...
def NJD(displacement):
    """
    Calculate the number (or percentage) of points with non-positive Jacobian determinant.
    Assumes displacement is a numpy array of shape (H, W, 2), where channel 0 is dy, channel 1 is dx.
    """
    # Ensure displacement is numpy
    if isinstance(displacement, torch.Tensor):
        displacement = displacement.detach().cpu().numpy()

    if displacement.ndim != 3 or displacement.shape[-1] != 2:
        logging.warning(f"Unexpected displacement shape for NJD: {displacement.shape}. Expected (H, W, 2)") # <-- Log warnings
        return np.nan # Return NaN or 0

    H, W, _ = displacement.shape
    if H < 2 or W < 2:
        logging.warning(f"Displacement map too small for NJD calculation: {displacement.shape}") # <-- Log warnings
        return 0.0 # Or np.nan

    # Calculate spatial gradients using finite differences
    # Flow components: dy = displacement[..., 0], dx = displacement[..., 1]

    # Gradients in y-direction (change between rows)
    dy_dy = displacement[1:, :-1, 0] - displacement[:-1, :-1, 0]
    dx_dy = displacement[1:, :-1, 1] - displacement[:-1, :-1, 1]

    # Gradients in x-direction (change between columns)
    dy_dx = displacement[:-1, 1:, 0] - displacement[:-1, :-1, 0]
    dx_dx = displacement[:-1, 1:, 1] - displacement[:-1, :-1, 1]

    # Jacobian determinant: det(J) = (dx_dx + 1) * (dy_dy + 1) - dx_dy * dy_dx
    # Note the +1 terms account for the identity transformation baseline
    jacobian_det = (dx_dx + 1) * (dy_dy + 1) - dx_dy * dy_dx

    # Count points where the determinant is non-positive (<= 0)
    num_folding_points = np.sum(jacobian_det <= 0)

    # Optional: Calculate percentage
    # total_pixels = jacobian_det.size
    # percentage = (num_folding_points / total_pixels) * 100 if total_pixels > 0 else 0
    # return percentage

    return float(num_folding_points)


# --- Training Function ---

def train(
    # Data Paths
    data_base_dir, # Base directory containing 'fixed', 'moving', etc.
    atlas_name, # Filename of the fixed atlas image (e.g., 'atlas.png')
    atlas_seg_name, # Filename of the fixed atlas segmentation (e.g., 'atlas_seg.png')
    # Model Paths
    model_dir,
    load_model,
    # Logging
    log_dir, # <-- Add log directory argument
    # Training Params
    device,
    initial_epoch,
    epochs,
    steps_per_epoch,
    batch_size,
    lr,
    # Loss Weights
    ncc_weight,
    grad_weight,
    # Validation options
    run_validation,
    validation_freq # Run validation every N epochs
    ):

    # --- 0. Setup Logging ---
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'log.txt')

    # Configure logging to write to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'), # Append mode
            logging.StreamHandler(sys.stdout)       # Console output
        ]
    )
    logging.info("--- Starting Experiment ---")
    logging.info(f"Log file: {log_file}")


    # --- 1. Setup and Preparation ---
    logging.info("--- Initializing Training ---")

    # Create model directory if it doesn't exist
    if not os.path.isdir(model_dir):
        logging.info(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir)

    # Setup device
    if 'gpu' in device and torch.cuda.is_available():
        gpu_id = device.split('gpu')[-1]
        if not gpu_id: gpu_id = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        device = f'cuda:{gpu_id}'
        logging.info(f"Using GPU: {device}")
        torch.backends.cudnn.deterministic = True
    else:
        if 'gpu' in device:
            logging.warning("CUDA specified but not available. Falling back to CPU.")
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logging.info("Using CPU.")
    device = torch.device(device)
    logging.info(f"Selected device: {device}")


    # --- 2. Load Data ---
    logging.info("--- Loading Data ---")

    # Construct full paths
    fixed_atlas_path = os.path.join(data_base_dir, 'fixed', atlas_name)
    moving_img_dir = os.path.join(data_base_dir, 'moving')
    validation_moving_img_dir = os.path.join(data_base_dir, 'validation_moving')
    fixed_atlas_seg_path = os.path.join(data_base_dir, 'fixed_seg', atlas_seg_name)
    validation_moving_seg_dir = os.path.join(data_base_dir, 'validation_moving_seg')

    # Load Fixed Atlas Image (once)
    fixed_atlas_tensor = load_and_preprocess_image(fixed_atlas_path, device, is_segmentation=False)
    if fixed_atlas_tensor is None:
        logging.error("Failed to load fixed atlas image. Exiting.")
        return

    logging.info(f"Loaded Fixed Atlas: {fixed_atlas_path} | Shape: {fixed_atlas_tensor.shape}")

    # Load Fixed Atlas Segmentation (once, if running validation)
    fixed_atlas_seg_tensor = None
    if run_validation:
        fixed_atlas_seg_tensor = load_and_preprocess_image(fixed_atlas_seg_path, device, is_segmentation=True)
        if fixed_atlas_seg_tensor is None:
            logging.warning(f"Could not load fixed atlas segmentation from {fixed_atlas_seg_path}. Validation Dice disabled.")
            run_validation = False
        else:
            logging.info(f"Loaded Fixed Atlas Segmentation: {fixed_atlas_seg_path} | Shape: {fixed_atlas_seg_tensor.shape}")

    # Get list of training moving image paths
    train_moving_image_paths = sorted(glob.glob(os.path.join(moving_img_dir, '*.png')))
    if not train_moving_image_paths:
        logging.error(f"No training images found in {moving_img_dir}. Exiting.")
        return
    num_train_images = len(train_moving_image_paths)
    logging.info(f"Found {num_train_images} training moving images.")

    # Get list of validation moving image paths and corresponding segmentations (if running validation)
    validation_pairs = []
    if run_validation:
        validation_moving_image_paths = sorted(glob.glob(os.path.join(validation_moving_img_dir, '*.png')))
        if not validation_moving_image_paths:
            logging.warning(f"No validation images found in {validation_moving_img_dir}. Disabling validation.")
            run_validation = False
        else:
            for img_path in validation_moving_image_paths:
                img_filename = os.path.basename(img_path)
                seg_path = os.path.join(validation_moving_seg_dir, img_filename)
                if os.path.exists(seg_path):
                    validation_pairs.append((img_path, seg_path))
                else:
                    logging.warning(f"Missing validation segmentation for {img_filename}. Skipping this pair.")
            if not validation_pairs:
                logging.warning("No valid validation image/segmentation pairs found. Disabling validation.")
                run_validation = False
            else:
                 logging.info(f"Found {len(validation_pairs)} validation image/segmentation pairs.")


    # --- 3. Initialize Model, Optimizer, Losses ---
    logging.info("--- Initializing Model ---")

    # Model
    model = network2D.CorrMLP(in_channels=1)
    model.to(device)

    # Load pre-trained weights if specified
    if load_model and os.path.isfile(load_model):
        logging.info(f"Loading model weights from: {load_model}")
        try:
            state_dict = torch.load(load_model, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Error loading model weights: {e}. Training from scratch.")
            load_model = None # Prevent trying to load again if error
    elif load_model:
        logging.warning(f"Model file specified but not found at {load_model}. Training from scratch.")


    # Spatial Transformer for validation warping
    spatial_transformer = network2D.SpatialTransformer_block(mode='nearest').to(device)
    spatial_transformer.eval()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    logging.info(f"Optimizer: Adam, LR: {lr}")

    # Losses
    ncc_loss_fn = losses2D.NCC(win=9).loss
    grad_loss_fn = losses2D.Grad(penalty='l2').loss # Ensure Grad.loss takes only y_pred
    logging.info(f"Losses: NCC (Weight: {ncc_weight}), Grad L2 (Weight: {grad_weight})")


    # --- 4. Training Loop ---
    logging.info(f"--- Starting Training for {epochs} Epochs ---")
    logging.info(f"Steps/Epoch: {steps_per_epoch}, Batch Size: {batch_size}")
    if run_validation:
        logging.info(f"Validation will run every {validation_freq} epochs.")

    best_valid_dice = -1.0

    for epoch in range(initial_epoch, epochs):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        epoch_train_losses = []
        epoch_train_total_loss = []

        for step in range(steps_per_epoch):
            # Select batch of moving images
            batch_indices = random.sample(range(num_train_images), batch_size)
            batch_moving_paths = [train_moving_image_paths[i] for i in batch_indices]

            # Load and preprocess batch
            batch_moving_tensors = []
            valid_batch = True
            for path in batch_moving_paths:
                tensor = load_and_preprocess_image(path, device, is_segmentation=False)
                if tensor is None:
                    logging.warning(f"Skipping step due to image loading error: {path}")
                    valid_batch = False
                    break
                batch_moving_tensors.append(tensor)

            if not valid_batch: continue

            moving_batch_tensor = torch.cat(batch_moving_tensors, dim=0)
            fixed_batch_tensor = fixed_atlas_tensor.repeat(batch_size, 1, 1, 1)

            # Run model
            warped_moving_pred, flow_pred = model(fixed_batch_tensor, moving_batch_tensor)

            # Calculate losses
            loss_ncc = ncc_loss_fn(fixed_batch_tensor, warped_moving_pred)
            loss_grad = grad_loss_fn(flow_pred) # Assumes Grad.loss now takes only flow_pred
            total_loss = (ncc_weight * loss_ncc) + (grad_weight * loss_grad)

            # Store losses
            epoch_train_losses.append([loss_ncc.item(), loss_grad.item()])
            epoch_train_total_loss.append(total_loss.item())

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # --- Validation Phase (conditional) ---
        avg_valid_dice = 0.0
        avg_valid_njd = 0.0
        run_current_validation = run_validation and ((epoch + 1) % validation_freq == 0)

        if run_current_validation:
            model.eval()
            epoch_valid_dice = []
            epoch_valid_njd = []
            logging.info(f"--- Running Validation for Epoch {epoch+1} ---")

            with torch.no_grad():
                for moving_val_path, moving_val_seg_path in validation_pairs:
                    moving_val_tensor = load_and_preprocess_image(moving_val_path, device, is_segmentation=False)
                    moving_val_seg_tensor = load_and_preprocess_image(moving_val_seg_path, device, is_segmentation=True)

                    if moving_val_tensor is None or moving_val_seg_tensor is None:
                        logging.warning(f"Skipping validation pair due to loading error.")
                        continue

                    fixed_val_tensor = fixed_atlas_tensor
                    fixed_val_seg_tensor = fixed_atlas_seg_tensor

                    warped_val_pred, flow_val_pred = model(fixed_val_tensor, moving_val_tensor)
                    warped_val_seg_pred = spatial_transformer(moving_val_seg_tensor, flow_val_pred)

                    dice_val = Dice(warped_val_seg_pred.squeeze(), fixed_val_seg_tensor.squeeze())
                    epoch_valid_dice.append(dice_val)

                    flow_val_np = flow_val_pred.squeeze().permute(1, 2, 0).cpu().numpy()
                    njd_val = NJD(flow_val_np)
                    if not np.isnan(njd_val):
                        epoch_valid_njd.append(njd_val)

            avg_valid_dice = np.mean(epoch_valid_dice) if epoch_valid_dice else 0.0
            avg_valid_njd = np.mean(epoch_valid_njd) if epoch_valid_njd else 0.0
            logging.info(f"--- Validation Complete - Avg Dice: {avg_valid_dice:.4f}, Avg NJD: {avg_valid_njd:.2f} ---")


        # --- Epoch End Summary ---
        epoch_elapsed_time = time.time() - epoch_start_time
        avg_train_total_loss = np.mean(epoch_train_total_loss) if epoch_train_total_loss else 0
        avg_train_ncc_loss = np.mean([l[0] for l in epoch_train_losses]) if epoch_train_losses else 0
        avg_train_grad_loss = np.mean([l[1] for l in epoch_train_losses]) if epoch_train_losses else 0

        # Construct summary string for logging
        summary_str = f"Epoch {epoch + 1}/{epochs}"
        summary_str += f" - Time: {epoch_elapsed_time:.2f}s"
        summary_str += f" - Train Loss: {avg_train_total_loss:.4f}"
        summary_str += f" (NCC: {avg_train_ncc_loss:.4f}, Grad: {avg_train_grad_loss:.4f})"
        if run_current_validation:
            summary_str += f" - Valid Dice: {avg_valid_dice:.4f}"
            summary_str += f" - Valid NJD: {avg_valid_njd:.2f}"

        logging.info(summary_str) # <-- Log the summary string

        # --- Save Model Checkpoint ---
        if epoch % 20 == 0:
            logging.info(f"Saving model checkpoint at epoch {epoch+1}")
            save_path = os.path.join(model_dir, f'epoch_{epoch+1:03d}.pt')
            torch.save(model.state_dict(), save_path)

        # Save best model based on validation Dice
        if run_current_validation and avg_valid_dice > best_valid_dice:
            best_valid_dice = avg_valid_dice
            best_save_path = os.path.join(model_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_save_path)
            logging.info(f"Saved Best Model (Dice: {best_valid_dice:.4f}) to {best_save_path}")


    logging.info("--- Training Finished ---")


# --- Main Execution ---
if __name__ == "__main__":
    parser = ArgumentParser()

    # Paths
    parser.add_argument("--data_base_dir", type=str, default='./data/', help="Base directory for fixed/, moving/, etc.")
    parser.add_argument("--atlas_name", type=str, default='atlas.png', help="Filename of the fixed atlas image")
    parser.add_argument("--atlas_seg_name", type=str, default='atlas_seg.png', help="Filename of the fixed atlas segmentation")
    parser.add_argument("--model_dir", type=str, default='./models/', help="Directory to save models")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load pre-trained model weights")
    parser.add_argument("--log_dir", type=str, default='./log/', help="Directory to save log file") # <-- Add log dir argument

    # Training parameters
    parser.add_argument("--device", type=str, default='gpu0', help="Device: cpu or gpuN (e.g., gpu0)")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Starting epoch (for resuming)")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Number of training steps per epoch")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    # Loss weights
    parser.add_argument("--ncc_weight", type=float, default=1.0, help="Weight for NCC similarity loss")
    parser.add_argument("--grad_weight", type=float, default=1.0, help="Weight for Grad diffusion regularization loss")

    # Validation
    parser.add_argument("--run_validation", action='store_true', help="Run validation loop")
    parser.add_argument("--validation_freq", type=int, default=1, help="Run validation every N epochs")


    args = parser.parse_args()

    # Start training
    train(**vars(args))