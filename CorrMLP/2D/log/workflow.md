
**Assumed Goal:** Train a network to learn a spatial deformation (flow field) that warps a "moving" image to align with a fixed "atlas" image.

**Phase 1: Data Preparation**

1.  **Gather Raw Data:**
    *   **Fixed Atlas Image:** Obtain the single reference image (e.g., an average brain template, a specific patient scan). This can be in formats like `.png`, `.nii.gz`, `.npy`, etc.
    *   **Moving Images (Training):** Collect a diverse set of images that you want to align *to* the atlas. These should represent the variability your model needs to handle (different patients, anatomy variations, acquisition differences). Format can be `.png`, `.nii.gz`, `.npy`, `.npz`, etc.
    *   **Fixed Atlas Segmentation (Optional, for Validation):** Obtain a segmentation mask corresponding *exactly* to the fixed atlas image, defining anatomical structures. Format can be `.png`, `.nii.gz`, etc.
    *   **Moving Images (Validation):** Select a separate set of moving images, **not** used for training, for unbiased evaluation.
    *   **Moving Segmentations (Validation, Optional):** Obtain segmentation masks corresponding *exactly* to each validation moving image.

2.  **Organize Data into Directories:** Structure your data on disk according to the expectations of `trainning.py`:
    *   `--data_base_dir` (e.g., `./data/`)
        *   `fixed/`
            *   `atlas.png` (or whatever `--atlas_name` is)
        *   `moving/` (or `--train_moving_subdir`)
            *   `subject01.png`
            *   `subject02.nii.gz`
            *   `scan_abc.npy`
            *   ... (all training moving images)
        *   `fixed_seg/` (or `--fixed_seg_subdir`)
            *   `atlas_seg.png` (or whatever `--atlas_seg_name` is - Needed if `run_validation=True`)
        *   `validation_moving/` (or `--valid_moving_subdir`)
            *   `val_subject99.png`
            *   `val_scan_xyz.nii.gz`
            *   ... (all validation moving images - Needed if `run_validation=True`)
        *   `validation_moving_seg/` (or `--valid_moving_seg_subdir`)
            *   `val_subject99_seg.png` (Note the `_seg` suffix convention)
            *   `val_scan_xyz_seg.nii.gz`
            *   ... (segmentations for validation images - Needed if `run_validation=True`)

3.  **Preprocessing (Handled by `datagenerators2D.py`):**
    *   No manual preprocessing like resizing or intensity clipping is done *before* training in this setup. The dataloader handles loading and basic normalization.
    *   **Normalization:** Intensity images (`.png`, `.nii.gz`, etc.) are automatically normalized to the `[0.0, 1.0]` range by `ImageLoader`.
    *   **Channel Dimension:** Both `ImageLoader` and `LabelLoader` add a channel dimension (`C=1`) to the 2D data (`H, W` -> `1, H, W`).
    *   **Data Types:** Images are loaded as `float32`. Labels are loaded based on the `dtype` specified for `LabelLoader` (default `float32`, suitable for warping).

**Phase 2: Training Initialization (`trainning.py`)**

1.  **Parse Arguments:** Reads command-line arguments (paths, hyperparameters, flags like `--run_validation`).
2.  **Setup Logging:** Configures the `logging` module to save output to `./log/log.txt` and print to the console.
3.  **Setup Device:** Detects CUDA availability and sets the appropriate `torch.device` (`cuda:N` or `cpu`).
4.  **Instantiate Loaders (`datagenerators2D`):** Creates instances of `ImageLoader` and `LabelLoader`, configuring them with NPZ keys and desired label data types.
5.  **Prepare Data Lists:**
    *   Identifies the atlas image path and (optionally) the atlas label path.
    *   Uses `glob` to find all training moving image files matching the pattern in the specified directory.
    *   If validation is enabled, uses `glob` to find validation moving images and attempts to pair them with corresponding segmentation files (expecting the `_seg` naming convention). Warns if pairs are incomplete.
6.  **Create `Dataset` (`datagenerators2D`):**
    *   Instantiates `RegistrationDataset` in `'atlas'` mode.
    *   Passes the list of training moving image paths (`train_samples`), the loaders, and the `atlas_paths_dict`.
    *   The Dataset pre-loads the fixed atlas image and (optional) label into memory upon initialization.
7.  **Create `DataLoader` (`torch.utils.data`):**
    *   Wraps the `train_dataset`.
    *   Handles shuffling, batching (`batch_size`), parallel loading (`num_workers`), and potentially moving data to pinned memory (`pin_memory`).
    *   `drop_last=True` ensures all batches have the same size.
8.  **Initialize Model (`network2D`):**
    *   Creates an instance of `network2D.CorrMLP` (assuming 1 input channel).
    *   Moves the model to the selected `device`.
    *   Optionally loads pre-trained weights from `--load_model`.
9.  **Initialize Optimizer (`torch.optim`):** Creates an Adam optimizer targeting the model's parameters with the specified learning rate (`lr`).
10. **Initialize Loss Functions (`losses2D`):** Creates instances of `NCC` and `Grad` loss functions.
11. **Initialize Spatial Transformer (`network2D`):** Creates an instance of `SpatialTransformer_block` (in 'nearest' mode for warping segmentations) for use during validation. Moves it to the `device`.

**Phase 3: Training Loop (`trainning.py`)**

The script iterates for the specified number of `epochs`. Each epoch consists of a training phase and an optional validation phase.

**Epoch - Training Phase:**

1.  **Set Mode:** `model.train()` enables training-specific layers like Dropout (if any).
2.  **Initialize Epoch Metrics:** Resets lists for tracking losses within the epoch.
3.  **Iterate Batches:** Loops through the `train_dataloader`, showing a `tqdm` progress bar. The loop runs for `len(train_dataloader)` iterations or up to `steps_per_epoch`, whichever is smaller.
4.  **Load Batch:** The `DataLoader` yields a `batch_data` dictionary. This dictionary contains:
    *   `'fixed_image'`: A tensor of shape `(B, 1, H, W)` containing the *same* pre-loaded atlas image, repeated `B` times.
    *   `'moving_image'`: A tensor of shape `(B, 1, H, W)` containing the batch of loaded and preprocessed moving images.
    *   *(Potentially `'fixed_label'`, `'moving_label'` if configured and available, but typically not used for atlas training loss).*
5.  **Move to Device:** Tensors are moved to the designated `device` (GPU or CPU).
6.  **Forward Pass:** The model performs a forward pass: `warped_moving_pred, flow_pred = model(fixed_batch, moving_batch)`.
    *   `fixed_batch` and `moving_batch` are the inputs.
    *   `warped_moving_pred`: The model's attempt to warp the `moving_batch` to match the `fixed_batch`. Shape `(B, 1, H, W)`.
    *   `flow_pred`: The predicted deformation field. Shape `(B, 2, H, W)` (2 channels for dy, dx).
7.  **Calculate Losses:**
    *   `loss_ncc = ncc_loss_fn(fixed_batch, warped_moving_pred)`: Calculates similarity between the atlas and the warped moving image. (Expected to be negative).
    *   `loss_grad = grad_loss_fn(flow_pred)`: Calculates the spatial smoothness penalty of the predicted flow field. (Expected to be positive).
    *   `total_loss = (ncc_weight * loss_ncc) + (grad_weight * loss_grad)`: Computes the weighted sum.
8.  **Backward Pass & Optimization:**
    *   `optimizer.zero_grad()`: Clears gradients from the previous step.
    *   `total_loss.backward()`: Computes gradients of the `total_loss` with respect to model parameters.
    *   `optimizer.step()`: Updates model parameters based on the computed gradients and the optimizer's logic (Adam).
9.  **Record Losses:** Appends the individual and total loss values for the step to the epoch's tracking lists.
10. **Update Progress Bar:** `tqdm` updates the display, possibly showing the current step's loss.

**Epoch - Validation Phase (Conditional):**

This phase runs only if `run_validation=True` and the current `epoch + 1` is a multiple of `validation_freq`.

1.  **Set Mode:** `model.eval()` disables training-specific layers (like Dropout) and sets BatchNorm layers (if any) to use running statistics.
2.  **Disable Gradients:** `with torch.no_grad():` ensures no gradients are computed during validation, saving memory and computation.
3.  **Iterate Validation Samples:** Loops through the `validation_samples` list (pairs of moving image/label paths), showing a `tqdm` progress bar.
4.  **Load Validation Pair:** Uses `datagenerators2D.load_validation_pair` to load the fixed atlas image, the current validation moving image, the fixed atlas label, and the current validation moving label. Tensors are loaded with `B=1` and moved to the `device`.
5.  **Forward Pass (Model):** `_, flow_val_pred = model(fixed_val, moving_val)` predicts the flow field for the validation pair. Only the flow is needed for metrics.
6.  **Forward Pass (Transformer):** `warped_val_seg_pred = spatial_transformer(moving_label_val, flow_val_pred)` uses the predicted flow to warp the *moving validation segmentation*.
7.  **Calculate Metrics:**
    *   `dice_val = Dice(warped_val_seg_pred, fixed_label_val)`: Computes the Dice score between the *warped moving segmentation* and the *fixed atlas segmentation*.
    *   `njd_val = NJD(flow_val_pred)`: Computes the number of non-positive Jacobian determinant points in the predicted flow field (converting flow to NumPy `H, W, C` format first).
8.  **Record Metrics:** Appends the calculated `dice_val` and `njd_val` to the epoch's validation metric lists.
9.  **Update Progress Bar:** `tqdm` updates, possibly showing the current pair's Dice score.

**Epoch - End:**

1.  **Calculate Averages:** Computes the average training losses (total, NCC, Grad) and average validation metrics (Dice, NJD) for the completed epoch.
2.  **Log Summary:** Records a summary string containing epoch number, time taken, average losses, and average validation metrics to both the console and the log file.
3.  **Save Checkpoint:** Saves the model's current state (`model.state_dict()`) to a file named like `epoch_XXX.pt`.
4.  **Save Best Model:** If validation ran and the average validation Dice score improved compared to the previous best, saves the current model state as `best_model.pt`.

**Phase 4: Training Completion**

After the loop finishes all epochs, a "Training Finished" message is logged. You can then use the saved checkpoints (`epoch_XXX.pt` or `best_model.pt`) for inference or further analysis.