
import os
import sys
import time
import glob
import random
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm # <--- Import tqdm

# Import your custom modules
import datagenerators2D
import network2D
import losses2D

# ... (Keep Dice and NJD functions) ...


# --- Metrics (Keep Dice and NJD functions as before) ---
def Dice(vol1, vol2, labels=None, nargout=1):
    # ... (Dice function code - ensure it handles numpy arrays) ...
    if isinstance(vol1, torch.Tensor): vol1 = vol1.detach().cpu().numpy()
    if isinstance(vol2, torch.Tensor): vol2 = vol2.detach().cpu().numpy()
    # ...(rest of Dice logic)...
    if labels is None:
        labels = np.unique(np.concatenate((vol1.flatten(), vol2.flatten())))
        labels = np.delete(labels, np.where(labels == 0))
    if len(labels) == 0:
        return 1.0 if np.all(vol1 == 0) and np.all(vol2 == 0) else 0.0
    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = (vol1 == lab);
        vol2l = (vol2 == lab)
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        dicem[idx] = 1.0 if bottom == 0 else top / bottom
    return np.mean(dicem) if nargout == 1 else (dicem, labels)


def NJD(displacement):
    # ... (NJD function code - ensure it handles numpy H, W, C arrays) ...
    if isinstance(displacement, torch.Tensor): displacement = displacement.detach().cpu().numpy()
    if displacement.ndim != 3 or displacement.shape[-1] != 2: return np.nan
    H, W, _ = displacement.shape
    if H < 2 or W < 2: return 0.0
    dy_dy = displacement[1:, :-1, 0] - displacement[:-1, :-1, 0]
    dx_dy = displacement[1:, :-1, 1] - displacement[:-1, :-1, 1]
    dy_dx = displacement[:-1, 1:, 0] - displacement[:-1, :-1, 0]
    dx_dx = displacement[:-1, 1:, 1] - displacement[:-1, :-1, 1]
    jacobian_det = (dx_dx + 1) * (dy_dy + 1) - dx_dy * dy_dx
    return float(np.sum(jacobian_det <= 0))


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
        steps_per_epoch,  # Note: Might not align perfectly with dataset size now
        batch_size,
        lr,
        num_workers,  # For DataLoader
        # Loss Weights
        ncc_weight,
        grad_weight,
        # Validation options
        run_validation,
        validation_freq,
        # Directories for training/validation data (relative to data_base_dir)
        train_moving_subdir='moving',
        valid_moving_subdir='validation_moving',
        fixed_seg_subdir='fixed_seg',  # Contains atlas segmentation
        valid_moving_seg_subdir='validation_moving_seg',
        # File patterns (useful if mixed types or specific naming)
        img_pattern='*',  # e.g., '*.png', '*.nii.gz'
        label_pattern='*',
        # NPZ keys (if using npz)
        img_npz_key='img',
        label_npz_key='label'
):
    # --- 0. Setup Logging ---
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)]
    )
    logging.info("--- Starting Experiment (Using Flexible Dataloader) ---")
    logging.info(f"Log file: {log_file}")

    # --- 1. Setup and Preparation ---
    logging.info("--- Initializing Training ---")
    if not os.path.isdir(model_dir):
        logging.info(f"Creating model directory: {model_dir}")
        os.makedirs(model_dir)

    # Setup device
    if 'gpu' in device and torch.cuda.is_available():
        gpu_id = device.split('gpu')[-1] or '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        device = torch.device(f'cuda:{gpu_id}')
        logging.info(f"Using GPU: {device}")
        torch.backends.cudnn.deterministic = True
    else:
        if 'gpu' in device: logging.warning("CUDA specified but not available. Falling back to CPU.")
        device = torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logging.info("Using CPU.")

    # --- 2. Prepare Data Lists and Loaders ---
    logging.info("--- Preparing Data ---")

    # Instantiate Loaders
    image_loader = datagenerators2D.ImageLoader(npz_key=img_npz_key, normalize=True)
    # Use float32 for labels if they will be warped by spatial transformer
    label_loader = datagenerators2D.LabelLoader(npz_key=label_npz_key, dtype=torch.float32)

    # --- Training Data (Atlas Mode) ---
    atlas_img_path = os.path.join(data_base_dir, 'fixed', atlas_name)
    atlas_label_path = os.path.join(data_base_dir, fixed_seg_subdir, atlas_seg_name)
    atlas_paths_dict = {'image': atlas_img_path}
    if run_validation and os.path.exists(atlas_label_path):
        atlas_paths_dict['label'] = atlas_label_path
    elif run_validation:
        logging.warning(f"Atlas segmentation {atlas_label_path} not found. Validation Dice may be inaccurate.")
        atlas_paths_dict['label'] = None  # Explicitly None

    train_moving_dir = os.path.join(data_base_dir, train_moving_subdir)
    train_moving_paths = sorted(glob.glob(os.path.join(train_moving_dir, img_pattern)))
    if not train_moving_paths:
        logging.error(f"No training images found in {train_moving_dir} matching pattern '{img_pattern}'. Exiting.")
        return
    logging.info(f"Found {len(train_moving_paths)} potential training moving images.")
    # For atlas mode, samples are just paths to moving images
    train_samples = train_moving_paths



    # Create Training Dataset and DataLoader
    try:
        train_dataset = datagenerators2D.RegistrationDataset(
            samples=train_samples,
            image_loader=image_loader,
            # label_loader needed only if moving labels are used in training (not typical for atlas)
            # label_loader=label_loader, # Uncomment if you have moving labels needed for training
            atlas_paths=atlas_paths_dict,
            mode='atlas'
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if device != torch.device('cpu') else False,
            drop_last=True  # Drop last incomplete batch
            # collate_fn= # Add custom collate if needed to handle None from dataset
        )
        logging.info(f"Training DataLoader created with {len(train_dataloader)} batches per epoch (approx).")
    except Exception as e:
        logging.error(f"Failed to create training dataset/dataloader: {e}", exc_info=True)
        return




    # --- Validation Data ---
    validation_samples = [] # List of (moving_img_path, moving_label_path) tuples
    if run_validation:
        valid_moving_dir = os.path.join(data_base_dir, valid_moving_subdir)
        valid_moving_seg_dir = os.path.join(data_base_dir, valid_moving_seg_subdir)
        valid_moving_img_paths = sorted(glob.glob(os.path.join(valid_moving_dir, img_pattern)))

        if not valid_moving_img_paths:
            logging.warning(f"No validation images found in {valid_moving_dir}. Disabling validation.")
            run_validation = False
        else:
            logging.info(f"Searching for validation pairs with '_seg' naming convention...")
            for img_path in valid_moving_img_paths:
                # Construct expected label filename
                img_filename_base, img_ext = os.path.splitext(os.path.basename(img_path))
                # Handle double extensions like .nii.gz
                if img_ext.lower() == '.gz':
                    img_filename_base_no_gz, img_ext_inner = os.path.splitext(img_filename_base)
                    if img_ext_inner.lower() == '.nii':
                         img_filename_base = img_filename_base_no_gz # Base is name before .nii.gz
                         img_ext = '.nii.gz' # Keep original double extension

                # --- Look for label with _seg suffix and ANY supported extension ---
                expected_label_base = f"{img_filename_base}_seg"
                # Search for the expected base name with various possible label extensions
                possible_label_paths = glob.glob(os.path.join(valid_moving_seg_dir, expected_label_base + ".*")) # More robust search

                if possible_label_paths:
                    # Use the first match found
                    label_path = possible_label_paths[0]
                    if len(possible_label_paths) > 1:
                         logging.warning(f"Multiple labels found for base '{expected_label_base}', using: {label_path}")
                    validation_samples.append((img_path, label_path))
                else:
                    logging.warning(f"Missing validation segmentation for {img_path} (expected base name '{expected_label_base}' in {valid_moving_seg_dir}). Skipping this pair.")

            if not validation_samples:
                logging.warning("No valid validation image/segmentation pairs found with '_seg' naming. Disabling validation.")
                run_validation = False
            else:
                 logging.info(f"Found {len(validation_samples)} validation image/segmentation pairs.")





    # --- 3. Initialize Model, Optimizer, Losses ---
    logging.info("--- Initializing Model ---")
    model = network2D.CorrMLP()  # Assuming 1 input channel (grayscale)
    model.to(device)

    if load_model and os.path.isfile(load_model):
        logging.info(f"Loading model weights from: {load_model}")
        try:
            state_dict = torch.load(load_model, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logging.error(f"Error loading model weights: {e}. Training from scratch.", exc_info=True)
            load_model = None
    elif load_model:
        logging.warning(f"Model file specified but not found at {load_model}. Training from scratch.")

    spatial_transformer = network2D.SpatialTransformer_block(mode='nearest').to(device)
    spatial_transformer.eval()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    logging.info(f"Optimizer: Adam, LR: {lr}")

    ncc_loss_fn = losses2D.NCC(win=9).loss
    grad_loss_fn = losses2D.Grad(penalty='l2').loss
    logging.info(f"Losses: NCC (Weight: {ncc_weight}), Grad L2 (Weight: {grad_weight})")

    # --- 4. Training Loop ---
    logging.info(f"--- Starting Training for {epochs} Epochs ---")
    if run_validation: logging.info(f"Validation will run every {validation_freq} epochs.")

    best_valid_dice = -1.0
    # steps_taken_in_epoch = 0 # No longer needed if tqdm shows iteration count

    for epoch in range(initial_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_train_losses = []
        epoch_train_total_loss = []
        # steps_taken_in_epoch = 0 # Reset counter if you still need steps_per_epoch limit

        # Wrap the dataloader with tqdm for a progress bar
        # Use total=len(train_dataloader) or steps_per_epoch if limiting steps
        num_steps_display = min(steps_per_epoch, len(train_dataloader)) if steps_per_epoch > 0 else len(train_dataloader)
        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} Training", total=num_steps_display, leave=False, dynamic_ncols=True)

        # Iterate over tqdm iterator
        for step, batch_data in enumerate(train_iterator):
            # Apply steps_per_epoch limit if specified and positive
            if steps_per_epoch > 0 and step >= steps_per_epoch:
                break

            # Check if batch loading failed
            if batch_data is None:
                logging.warning("Skipping step due to data loading error in batch.")
                continue

            # Move data to device
            fixed_batch = batch_data.get('fixed_image')
            moving_batch = batch_data.get('moving_image')
            if fixed_batch is None or moving_batch is None:
                logging.warning("Skipping step due to missing image data in batch.")
                continue
            fixed_batch = fixed_batch.to(device, non_blocking=True)
            moving_batch = moving_batch.to(device, non_blocking=True)

            # Run model and calculate loss (with error handling)
            try:
                warped_moving_pred, flow_pred = model(fixed_batch, moving_batch)
                loss_ncc = ncc_loss_fn(fixed_batch, warped_moving_pred)
                loss_grad = grad_loss_fn(flow_pred)
                total_loss = (ncc_weight * loss_ncc) + (grad_weight * loss_grad)
            except Exception as e:
                logging.error(f"Error during model forward/loss calculation: {e}", exc_info=True)
                continue

            # Store losses
            epoch_train_losses.append([loss_ncc.item(), loss_grad.item()])
            epoch_train_total_loss.append(total_loss.item())

            # Backpropagation (with error handling)
            optimizer.zero_grad()
            try:
                total_loss.backward()
                optimizer.step()
            except Exception as e:
                 logging.error(f"Error during backward/optimizer step: {e}", exc_info=True)
                 continue

            # Update tqdm progress bar description with current loss (optional)
            train_iterator.set_postfix(loss=f"{total_loss.item():.4f}")
            # steps_taken_in_epoch += 1 # Increment if limiting steps

        train_iterator.close() # Close the training progress bar

        # --- Validation Phase --- (Add tqdm here too)
        avg_valid_dice = 0.0
        avg_valid_njd = 0.0
        run_current_validation = run_validation and ((epoch + 1) % validation_freq == 0)

        if run_current_validation:
            model.eval()
            epoch_valid_dice = []
            epoch_valid_njd = []
            logging.info(f"--- Running Validation for Epoch {epoch+1} ---")

            # Wrap validation sample iteration with tqdm
            validation_iterator = tqdm(validation_samples, desc="Validation", leave=False, dynamic_ncols=True)

            with torch.no_grad():
                for i, (moving_val_path, moving_val_seg_path) in enumerate(validation_iterator):
                    val_data = datagenerators2D.load_validation_pair(
                        fixed_img_path=atlas_paths_dict['image'],
                        moving_img_path=moving_val_path,
                        fixed_label_path=atlas_paths_dict.get('label'),
                        moving_label_path=moving_val_seg_path,
                        image_loader=image_loader,
                        label_loader=label_loader,
                        device=device
                    )
                    if val_data is None: continue # Skip if loading failed

                    fixed_val = val_data['fixed_image']
                    moving_val = val_data['moving_image']
                    fixed_label_val = val_data.get('fixed_label')
                    moving_label_val = val_data.get('moving_label')

                    if fixed_label_val is None or moving_label_val is None:
                        logging.warning(f"Missing fixed or moving label for val pair {i+1}. Cannot calc Dice.")
                        continue

                    try:
                        _, flow_val_pred = model(fixed_val, moving_val)
                        warped_val_seg_pred = spatial_transformer(moving_label_val, flow_val_pred)

                        dice_val = Dice(warped_val_seg_pred.squeeze(0), fixed_label_val.squeeze(0))
                        epoch_valid_dice.append(dice_val)

                        flow_val_np = flow_val_pred.squeeze().permute(1, 2, 0).cpu().numpy()
                        njd_val = NJD(flow_val_np)
                        if not np.isnan(njd_val): epoch_valid_njd.append(njd_val)

                        # Update validation progress bar postfix (optional)
                        validation_iterator.set_postfix(dice=f"{dice_val:.4f}")

                    except Exception as e:
                        logging.error(f"Error during validation processing for pair {i+1}: {e}", exc_info=True)
                        continue

            validation_iterator.close() # Close the validation progress bar

            # Calculate average validation metrics
            avg_valid_dice = np.mean(epoch_valid_dice) if epoch_valid_dice else 0.0
            avg_valid_njd = np.mean(epoch_valid_njd) if epoch_valid_njd else 0.0
            # Logging info about validation results is now handled in the epoch summary

        # --- Epoch End Summary ---
        epoch_elapsed_time = time.time() - epoch_start_time
        # Calculate average training losses... (code remains the same)
        avg_train_total_loss = np.mean(epoch_train_total_loss) if epoch_train_total_loss else 0
        avg_train_ncc_loss = np.mean([l[0] for l in epoch_train_losses]) if epoch_train_losses else 0
        avg_train_grad_loss = np.mean([l[1] for l in epoch_train_losses]) if epoch_train_losses else 0

        summary_str = f"Epoch {epoch + 1}/{epochs}"
        summary_str += f" - Time: {epoch_elapsed_time:.2f}s"
        # summary_str += f" - Train Steps: {steps_taken_in_epoch}" # Use tqdm's count instead
        summary_str += f" - Train Loss: {avg_train_total_loss:.4f}"
        summary_str += f" (NCC: {avg_train_ncc_loss:.4f}, Grad: {avg_train_grad_loss:.4f})"
        if run_current_validation:
            summary_str += f" - Valid Dice: {avg_valid_dice:.4f}"
            summary_str += f" - Valid NJD: {avg_valid_njd:.2f}"
        logging.info(summary_str) # Log the summary

        # --- Save Model Checkpoint --- (code remains the same)
        # ...

        # --- Save Model Checkpoint ---
        if epoch % 20 == 0:
          save_path = os.path.join(model_dir, f'epoch_{epoch + 1:03d}.pt')
          torch.save(model.state_dict(), save_path)

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
    parser.add_argument("--data_base_dir", type=str, default='./data/', help="Base directory for data")
    parser.add_argument("--atlas_name", type=str, default='atlas.nii.gz',
                        help="Filename of fixed atlas image (in data_base_dir/fixed/)")
    parser.add_argument("--atlas_seg_name", type=str, default='atlas_seg.nii.gz',
                        help="Filename of fixed atlas segmentation (in data_base_dir/fixed_seg/)")
    parser.add_argument("--train_moving_subdir", type=str, default='moving',
                        help="Subdirectory for training moving images")
    parser.add_argument("--valid_moving_subdir", type=str, default='validation_moving',
                        help="Subdirectory for validation moving images")
    parser.add_argument("--fixed_seg_subdir", type=str, default='fixed_seg',
                        help="Subdirectory for fixed atlas segmentation")
    parser.add_argument("--valid_moving_seg_subdir", type=str, default='validation_moving_seg',
                        help="Subdirectory for validation moving segmentations")
    parser.add_argument("--img_pattern", type=str, default='*.nii.gz',
                        help="Glob pattern for image files (e.g., '*.png', '*.nii.gz')")
    parser.add_argument("--label_pattern", type=str, default='*.nii.gz', help="Glob pattern for label files")
    parser.add_argument("--img_npz_key", type=str, default='img', help="Key for image data in NPZ files")
    parser.add_argument("--label_npz_key", type=str, default='label', help="Key for label data in NPZ files")
    parser.add_argument("--model_dir", type=str, default='./models/', help="Directory to save models")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load pre-trained model weights")
    parser.add_argument("--log_dir", type=str, default='./log/', help="Directory to save log file")

    # Training parameters
    parser.add_argument("--device", type=str, default='gpu0', help="Device: cpu or gpuN")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Starting epoch")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=200, help="Max steps per epoch (data loader might yield fewer)")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Num workers for DataLoader (0 for main thread)")

    # Loss weights
    parser.add_argument("--ncc_weight", type=float, default=1.0, help="Weight for NCC loss")
    parser.add_argument("--grad_weight", type=float, default=1.0, help="Weight for Grad L2 loss")

    # Validation
    parser.add_argument("--run_validation", default=True, action='store_true', help="Run validation loop")
    parser.add_argument("--validation_freq", type=int, default=1, help="Run validation every N epochs")

    args = parser.parse_args()

    # Start training
    train(**vars(args))