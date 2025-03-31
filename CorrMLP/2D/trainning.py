import os
import sys
import time
import glob
import random
import logging
import numpy as np
from PIL import Image
import torch
import torch.optim as optim
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm # <-- Import tqdm


import network2D  # Contains CorrMLP, SpatialTransformer_block
import losses2D   # Contains NCC, Grad

# --- Helper Function for Loading and Preprocessing PNG ---
# Modified to return tensor directly, device handled later if batch loading
def load_png_to_numpy(path, is_segmentation=False):
    """Loads a PNG image, converts to grayscale numpy array."""
    try:
        img = Image.open(path).convert('L') # Load and convert to grayscale
        img_np = np.array(img)
        if not is_segmentation:
            # Normalize intensity images
            img_np = img_np.astype(np.float32) / 255.0
        return img_np
    except FileNotFoundError:
        logging.error(f"Image file not found at {path}")
        return None
    except Exception as e:
        logging.error(f"Error loading image {path}: {e}")
        return None


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

    # Ensure inputs are integer type for segmentation comparison
    vol1 = np.round(vol1).astype(int)
    vol2 = np.round(vol2).astype(int)


    if labels is None:
        # Consider all unique non-zero values as labels
        labels = np.unique(np.concatenate((vol1.flatten(), vol2.flatten())))
        labels = np.delete(labels, np.where(labels == 0)) # remove background

    if len(labels) == 0:
        # Handle case where only background is present
         return 1.0 if np.all(vol1 == 0) and np.all(vol2 == 0) else 0.0

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = (vol1 == lab)
        vol2l = (vol2 == lab)
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)

        if bottom == 0:
            dicem[idx] = 1.0 if top == 0 else 0.0
        else:
             dicem[idx] = top / bottom

    if nargout == 1:
        return np.mean(dicem) if len(dicem) > 0 else 1.0 # Return 1 if no foreground labels found
    else:
        return (dicem, labels)
# ... (NJD function remains the same) ...
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
    dy_dy = displacement[1:, :-1, 0] - displacement[:-1, :-1, 0]
    dx_dy = displacement[1:, :-1, 1] - displacement[:-1, :-1, 1]
    dy_dx = displacement[:-1, 1:, 0] - displacement[:-1, :-1, 0]
    dx_dx = displacement[:-1, 1:, 1] - displacement[:-1, :-1, 1]

    jacobian_det = (dx_dx + 1) * (dy_dy + 1) - dx_dy * dy_dx
    num_folding_points = np.sum(jacobian_det <= 0)

    return float(num_folding_points)


# --- Training Function ---

def train(
    # Data Paths
    data_base_dir,
    atlas_name,
    atlas_seg_name,
    # Model Paths
    model_dir,
    load_model,
    # Logging
    log_dir,
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
    validation_freq
    ):

    # --- 0. Setup Logging ---
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("--- Starting Experiment ---")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Arguments: {locals()}") # Log input arguments


    # --- 1. Setup and Preparation ---
    logging.info("--- Initializing Setup ---")

    if not os.path.isdir(model_dir):
        logging.info(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir)

    # Setup device
    if 'gpu' in device and torch.cuda.is_available():
        gpu_id = device.split('gpu')[-1]
        if not gpu_id: gpu_id = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        device = f'cuda:{gpu_id}'
        torch.backends.cudnn.deterministic = True
        # Consider torch.backends.cudnn.benchmark = True if input sizes are fixed
    else:
        if 'gpu' in device: logging.warning("CUDA specified but not available. Falling back to CPU.")
        device = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device(device)
    logging.info(f"Using device: {device}")


    # --- 2. Load and Pre-process Data ---
    logging.info("--- Loading and Pre-processing Data ---")
    loading_start_time = time.time()

    # Construct full paths
    fixed_atlas_path = os.path.join(data_base_dir, 'fixed', atlas_name)
    moving_img_dir = os.path.join(data_base_dir, 'moving')
    validation_moving_img_dir = os.path.join(data_base_dir, 'validation_moving')
    fixed_atlas_seg_path = os.path.join(data_base_dir, 'fixed_seg', atlas_seg_name)
    validation_moving_seg_dir = os.path.join(data_base_dir, 'validation_moving_seg')

    # Load Fixed Atlas Image
    fixed_atlas_np = load_png_to_numpy(fixed_atlas_path, is_segmentation=False)
    if fixed_atlas_np is None:
        logging.error("Failed to load fixed atlas image. Exiting.")
        return
    # Add batch/channel dim, convert to tensor, move to device
    fixed_atlas_tensor = torch.from_numpy(fixed_atlas_np).unsqueeze(0).unsqueeze(0).to(device).float()
    logging.info(f"Loaded Fixed Atlas: {fixed_atlas_path} | Shape: {fixed_atlas_tensor.shape}")

    # Load Fixed Atlas Segmentation (if running validation)
    fixed_atlas_seg_tensor = None
    if run_validation:
        fixed_atlas_seg_np = load_png_to_numpy(fixed_atlas_seg_path, is_segmentation=True)
        if fixed_atlas_seg_np is None:
            logging.warning(f"Could not load fixed atlas segmentation from {fixed_atlas_seg_path}. Validation Dice disabled.")
            run_validation = False
        else:
            # Use float for spatial transformer, keep values distinct
            fixed_atlas_seg_tensor = torch.from_numpy(fixed_atlas_seg_np).unsqueeze(0).unsqueeze(0).to(device).float()
            logging.info(f"Loaded Fixed Atlas Segmentation: {fixed_atlas_seg_path} | Shape: {fixed_atlas_seg_tensor.shape}")

    # Pre-load Training Moving Images
    train_moving_image_paths = sorted(glob.glob(os.path.join(moving_img_dir, '*.png')))
    if not train_moving_image_paths:
        logging.error(f"No training images found in {moving_img_dir}. Exiting.")
        return

    logging.info(f"Loading {len(train_moving_image_paths)} training moving images...")
    train_moving_tensors = []
    for path in tqdm(train_moving_image_paths, desc="Loading Train Images"):
        img_np = load_png_to_numpy(path, is_segmentation=False)
        if img_np is not None:
            # Add channel dim (batch dim added later) -> (1, H, W)
            tensor = torch.from_numpy(img_np).unsqueeze(0).to(device).float()
            train_moving_tensors.append(tensor)
        else:
            logging.warning(f"Skipping training image due to loading error: {path}")

    if not train_moving_tensors:
        logging.error("Failed to load any valid training moving images. Exiting.")
        return
    num_train_images = len(train_moving_tensors)
    logging.info(f"Successfully loaded {num_train_images} training moving tensors to device {device}.")

    # Pre-load Validation Data (if running validation)
    validation_data = [] # List of tuples: (moving_tensor, seg_tensor)
    if run_validation:
        validation_moving_image_paths = sorted(glob.glob(os.path.join(validation_moving_img_dir, '*.png')))
        if not validation_moving_image_paths:
            logging.warning(f"No validation images found in {validation_moving_img_dir}. Disabling validation.")
            run_validation = False
        else:
            logging.info(f"Loading {len(validation_moving_image_paths)} validation pairs...")
            for img_path in tqdm(validation_moving_image_paths, desc="Loading Validation Pairs"):
                img_filename = os.path.basename(img_path)
                img_filename = f"{os.path.splitext(img_filename)[0]}_seg.png"
                seg_path = os.path.join(validation_moving_seg_dir, img_filename)

                if not os.path.exists(seg_path):
                    logging.warning(f"Missing validation segmentation for {img_filename}. Skipping pair.")
                    continue

                moving_np = load_png_to_numpy(img_path, is_segmentation=False)
                seg_np = load_png_to_numpy(seg_path, is_segmentation=True)

                if moving_np is not None and seg_np is not None:
                    # Add channel dim -> (1, H, W)
                    moving_tensor = torch.from_numpy(moving_np).unsqueeze(0).to(device).float()
                    seg_tensor = torch.from_numpy(seg_np).unsqueeze(0).to(device).float()
                    validation_data.append((moving_tensor, seg_tensor))
                else:
                    logging.warning(f"Skipping validation pair due to loading error: {img_filename}")

            if not validation_data:
                logging.warning("No valid validation pairs loaded. Disabling validation.")
                run_validation = False
            else:
                logging.info(f"Successfully loaded {len(validation_data)} validation pairs to device {device}.")

    logging.info(f"Data loading and pre-processing finished in {time.time() - loading_start_time:.2f} seconds.")


    # --- 3. Initialize Model, Optimizer, Losses ---
    logging.info("--- Initializing Model ---")
    model = network2D.CorrMLP(in_channels=1).to(device)

    if load_model and os.path.isfile(load_model):
        logging.info(f"Loading model weights from: {load_model}")
        try:
            state_dict = torch.load(load_model, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Error loading model weights: {e}. Training from scratch.")
            load_model = None
    elif load_model:
        logging.warning(f"Model file specified but not found at {load_model}. Training from scratch.")

    spatial_transformer = network2D.SpatialTransformer_block(mode='nearest').to(device)
    spatial_transformer.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    logging.info(f"Optimizer: Adam, LR: {lr}")

    ncc_loss_fn = losses2D.NCC(win=9).loss
    grad_loss_fn = losses2D.Grad(penalty='l2').loss # Ensure Grad.loss takes only y_pred
    logging.info(f"Losses: NCC (Weight: {ncc_weight}), Grad L2 (Weight: {grad_weight})")


    # --- 4. Training Loop ---
    logging.info(f"--- Starting Training for {epochs} Epochs ---")
    logging.info(f"Steps/Epoch: {steps_per_epoch}, Batch Size: {batch_size}")
    if run_validation: logging.info(f"Validation will run every {validation_freq} epochs.")

    best_valid_dice = -1.0
    train_indices = list(range(num_train_images))

    for epoch in range(initial_epoch, epochs):
        epoch_start_time = time.time()
        logging.info(f"Starting Epoch {epoch + 1}/{epochs}")

        # --- Training Phase ---
        model.train()
        epoch_train_losses = []
        epoch_train_total_loss = []

        # Shuffle indices at the start of each epoch
        random.shuffle(train_indices)

        # Use tqdm for the steps loop
        train_pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs} Training", ncols=100)
        for step in train_pbar:
            # Select batch indices using shuffled list (with wrapping)
            batch_indices = [train_indices[(step * batch_size + i) % num_train_images] for i in range(batch_size)]

            # Retrieve pre-loaded tensors and add batch dimension
            batch_moving_tensors = [train_moving_tensors[i].unsqueeze(0) for i in batch_indices] # Add B=1 dim
            moving_batch_tensor = torch.cat(batch_moving_tensors, dim=0) # Cat -> (B, 1, H, W)

            # Repeat fixed atlas tensor for the batch (already on device)
            fixed_batch_tensor = fixed_atlas_tensor.expand(batch_size, -1, -1, -1) # More efficient than repeat

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

            # Update tqdm progress bar postfix
            train_pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                ncc=f"{loss_ncc.item():.4f}",
                grad=f"{loss_grad.item():.4f}"
            )
        train_pbar.close() # Close the training progress bar for the epoch

        # --- Validation Phase (conditional) ---
        avg_valid_dice = 0.0
        avg_valid_njd = 0.0
        run_current_validation = run_validation and ((epoch + 1) % validation_freq == 0)

        if run_current_validation:
            model.eval()
            epoch_valid_dice = []
            epoch_valid_njd = []
            logging.info(f"--- Running Validation ---")

            # Use tqdm for validation loop
            val_pbar = tqdm(validation_data, desc="Validation", ncols=100, leave=False)
            with torch.no_grad():
                for moving_val_tensor, moving_val_seg_tensor in val_pbar:
                    # Add Batch dim B=1 to pre-loaded tensors
                    moving_val_tensor = moving_val_tensor.unsqueeze(0)
                    moving_val_seg_tensor = moving_val_seg_tensor.unsqueeze(0)

                    # Fixed tensors already have B=1 dim
                    fixed_val_tensor = fixed_atlas_tensor
                    fixed_val_seg_tensor = fixed_atlas_seg_tensor

                    # Run model
                    warped_val_pred, flow_val_pred = model(fixed_val_tensor, moving_val_tensor)
                    # Warp validation segmentation
                    warped_val_seg_pred = spatial_transformer(moving_val_seg_tensor, flow_val_pred)

                    # Calculate metrics
                    dice_val = Dice(warped_val_seg_pred.squeeze(), fixed_val_seg_tensor.squeeze())
                    epoch_valid_dice.append(dice_val)

                    flow_val_np = flow_val_pred.squeeze().permute(1, 2, 0).cpu().numpy()
                    njd_val = NJD(flow_val_np)
                    if not np.isnan(njd_val): epoch_valid_njd.append(njd_val)

            avg_valid_dice = np.mean(epoch_valid_dice) if epoch_valid_dice else 0.0
            avg_valid_njd = np.mean(epoch_valid_njd) if epoch_valid_njd else 0.0
            val_pbar.close() # Close validation progress bar

        # --- Epoch End Summary ---
        epoch_elapsed_time = time.time() - epoch_start_time
        avg_train_total_loss = np.mean(epoch_train_total_loss) if epoch_train_total_loss else 0
        avg_train_ncc_loss = np.mean([l[0] for l in epoch_train_losses]) if epoch_train_losses else 0
        avg_train_grad_loss = np.mean([l[1] for l in epoch_train_losses]) if epoch_train_losses else 0

        summary_str = f"Epoch {epoch + 1}/{epochs}"
        summary_str += f" - Time: {epoch_elapsed_time:.2f}s"
        summary_str += f" - Train Loss: {avg_train_total_loss:.4f}"
        summary_str += f" (NCC: {avg_train_ncc_loss:.4f}, Grad: {avg_train_grad_loss:.4f})"
        if run_current_validation:
            summary_str += f" - Valid Dice: {avg_valid_dice:.4f}"
            summary_str += f" - Valid NJD: {avg_valid_njd:.2f}"
        logging.info(summary_str)

        # --- Save Model Checkpoint ---
        save_path = os.path.join(model_dir, f'epoch_{epoch + 1:03d}.pt')
        if epoch % 20 == 0:
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
    parser.add_argument("--log_dir", type=str, default='./log/', help="Directory to save log file")

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
    parser.add_argument("--run_validation", default=True,action='store_true', help="Run validation loop")
    parser.add_argument("--validation_freq", type=int, default=1, help="Run validation every N epochs")

    args = parser.parse_args()

    # Start training
    train(**vars(args))