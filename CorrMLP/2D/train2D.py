# import os
# import glob
# import sys
# import random
# import time
# import torch
# import numpy as np
# import scipy.ndimage
# from argparse import ArgumentParser
#
# # project imports
# import datagenerators2D
# import network2D
# import losses2D
#
#
# def Dice(vol1, vol2, labels=None, nargout=1):
#
#     if labels is None:
#         labels = np.unique(np.concatenate((vol1, vol2)))
#         labels = np.delete(labels, np.where(labels == 0))  # remove background
#
#     dicem = np.zeros(len(labels))
#     for idx, lab in enumerate(labels):
#         vol1l = vol1 == lab
#         vol2l = vol2 == lab
#         top = 2 * np.sum(np.logical_and(vol1l, vol2l))
#         bottom = np.sum(vol1l) + np.sum(vol2l)
#         bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
#         dicem[idx] = top / bottom
#
#     if nargout == 1:
#         return dicem
#     else:
#         return (dicem, labels)
#
#
# def NJD(displacement):
#
#     D_y = (displacement[1:, :-1, :] - displacement[:-1, :-1, :])
#     D_x = (displacement[:-1, 1:, :] - displacement[:-1, :-1, :])
#
#     D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1) - D_y[..., 0] * D_x[..., 1]
#     Ja_value = D1
#
#     return np.sum(Ja_value < 0)
#
#
#
# def train(train_dir,
#           train_pairs,
#           valid_dir,
#           valid_pairs,
#           model_dir,
#           load_model,
#           device,
#           initial_epoch,
#           epochs,
#           steps_per_epoch,
#           batch_size):
#
#     # preparation
#     train_pairs = np.load(train_dir + train_pairs, allow_pickle=True)
#     valid_pairs = np.load(valid_dir + valid_pairs, allow_pickle=True)
#
#     # prepare model folder
#     if not os.path.isdir(model_dir):
#         os.mkdir(model_dir)
#
#     # device handling
#     if 'gpu' in device:
#         os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
#         device = 'cuda'
#         torch.backends.cudnn.deterministic = True
#     else:
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#         device = 'cpu'
#
#     # prepare the model
#     model = network2D.CorrMLP()
#     model.to(device)
#     if load_model != './':
#         print('loading', load_model)
#         state_dict = torch.load(load_model, map_location=device)
#         model.load_state_dict(state_dict)
#
#     # transfer model
#     SpatialTransformer = network2D.SpatialTransformer_block(mode='nearest')
#     SpatialTransformer.to(device)
#     SpatialTransformer.eval()
#
#     # set optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#
#     # prepare losses
#     Losses = [losses2D.NCC(win=9).loss, losses2D.Grad('l2').loss]
#     Weights = [1.0, 1.0]
#
#     # data generator
#     train_gen_pairs = datagenerators2D.gen_pairs(train_dir, train_pairs, batch_size=batch_size)
#     train_gen = datagenerators2D.gen_s2s(train_gen_pairs, batch_size=batch_size)
#
#     # training/validate loops
#     for epoch in range(initial_epoch, epochs):
#         start_time = time.time()
#
#         # training
#         model.train()
#         train_losses = []
#         train_total_loss = []
#         for step in range(steps_per_epoch):
#
#             # generate inputs (and true outputs) and convert them to tensors
#             inputs, labels = next(train_gen)
#             print(f"原始 inputs 的形状: {[d.shape for d in inputs]}")
#             # inputs = [torch.from_numpy(d).unsqueeze(1).to(device).float().permute(0,1,2,3) for d in inputs]
#             # labels = [torch.from_numpy(d).unsqueeze(1).to(device).float().permute(0,1,2,3) for d in labels]
#
#
#             # 统一输入张量的维度
#             inputs = [
#                 torch.from_numpy(d).unsqueeze(-1) if len(d.shape) == 3 else torch.from_numpy(d)
#                 for d in inputs
#             ]
#             # 调整转换管道
#             inputs = [
#                 t # 在位置 1 添加新维度
#                 .to(device)  # 移动到指定设备
#                 .float()  # 确保张量是浮点类型
#                 .permute(0, 3, 1, 2)  # 调整 permute 以匹配张量的维度
#                 for t in inputs
#             ]
#             print(f"最终 inputs 的形状: {[t.shape for t in inputs]}")
#
#
#             print(f"原始 labels 的形状: {[d.shape for d in labels]}")
#             # labels = [
#             #     torch.from_numpy(d).unsqueeze(-1) if len(d.shape) == 3 else torch.from_numpy(d)
#             #     for d in labels
#             # ]
#             # 处理 labels
#             labels = [
#                 torch.from_numpy(d).unsqueeze(-1) if len(d.shape) == 3 else torch.from_numpy(d)
#                 for d in labels
#             ]
#             labels = [
#                 t.to(device).float().permute(0, 3, 1, 2)  # 调整 permute 以匹配张量的维度
#                 for t in labels
#             ]
#             print(f"最终 labels 的形状: {[t.shape for t in labels]}")
#             # for a in inputs:
#             #     tensor = torch.from_numpy(a).to(device).float()
#             #     print(f"Tensor shape: {tensor.shape}")
#             #     inputs = [tensor.permute(0,1,2,3)]
#             #     print(f'inputs shape: {tensor.shape}')
#             # for b in labels:
#             #     tensor = torch.from_numpy(b).unsqueeze(1).to(device).float()
#             #     print(f"Tensor shape: {tensor.shape}")
#             #     labels = [tensor.permute(0, 1, 2, 3, 4)]
#
#                     # run inputs through the model to produce a warped image and flow field
#             pred = model(*inputs)
#
#             # calculate total loss
#             loss = 0
#             loss_list = []
#             for i, Loss in enumerate(Losses):
#                 curr_loss = Loss(labels[i], pred[i]) * Weights[i]
#                 loss_list.append(curr_loss.item())
#                 loss += curr_loss
#             train_losses.append(loss_list)
#             train_total_loss.append(loss.item())
#
#             # backpropagate and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         # validation
#         model.eval()
#         valid_Dice = []
#         valid_NJD = []
#         for valid_pair in valid_pairs:
#
#             # generate inputs (and true outputs) and convert them to tensors
#             fixed_vol, fixed_seg = datagenerators2D.load_by_name(valid_dir, valid_pair[0])
#             fixed_vol = torch.from_numpy(fixed_vol).to(device).float()
#             fixed_seg = torch.from_numpy(fixed_seg).to(device).float()
#
#             moving_vol, moving_seg = datagenerators2D.load_by_name(valid_dir, valid_pair[1])
#             moving_vol = torch.from_numpy(moving_vol).to(device).float()
#             moving_seg = torch.from_numpy(moving_seg).to(device).float()
#
#             # run inputs through the model to produce a warped image and flow field
#             with torch.no_grad():
#                 pred = model(fixed_vol, moving_vol)
#                 warped_seg = SpatialTransformer(moving_seg, pred[1])
#
#             warped_seg = warped_seg.detach().cpu().numpy().squeeze()
#             fixed_seg = fixed_seg.detach().cpu().numpy().squeeze()
#             Dice_val = Dice(warped_seg, fixed_seg)
#             valid_Dice.append(Dice_val)
#
#             flow = pred[1].detach().cpu().permute(0, 2, 3, 1).numpy().squeeze()
#             NJD_val = NJD(flow)
#             valid_NJD.append(NJD_val)
#
#         # print epoch info
#         epoch_info = 'Epoch %d/%d' % (epoch + 1, epochs)
#         time_info = 'Total %.2f sec' % (time.time() - start_time)
#         train_losses = ', '.join(['%.4f' % f for f in np.mean(train_losses, axis=0)])
#         train_loss_info = 'Train loss: %.4f  (%s)' % (np.mean(train_total_loss), train_losses)
#         valid_Dice_info = 'Valid DSC: %.4f' % (np.mean(valid_Dice))
#         valid_NJD_info = 'Valid NJD: %.2f' % (np.mean(valid_NJD))
#         print(' - '.join((epoch_info, time_info, train_loss_info, valid_Dice_info, valid_NJD_info)), flush=True)
#
#         # save model checkpoint
#         torch.save(model.state_dict(), os.path.join(model_dir, '%02d.pt' % (epoch+1)))
#
#
#
# if __name__ == "__main__":
#     parser = ArgumentParser()
#
#     parser.add_argument("--train_dir", type=str,
#                         dest="train_dir", default='./data/pairs/',
#                         help="training folder")
#     parser.add_argument("--train_pairs", type=str,
#                         dest="train_pairs", default='train_pairs.npy',
#                         help="training pairs(.npy)")
#     parser.add_argument("--valid_dir", type=str,
#                         dest="valid_dir", default='./data/pairs/',
#                         help="validation folder")
#     parser.add_argument("--valid_pairs", type=str,
#                         dest="valid_pairs", default='valid_pairs.npy',
#                         help="validation pairs(.npy)")
#     parser.add_argument("--model_dir", type=str,
#                         dest="model_dir", default='./models/',
#                         help="models folder")
#     parser.add_argument("--load_model", type=str,
#                         dest="load_model", default='./',
#                         help="load model file to initialize with")
#     parser.add_argument("--device", type=str, default='gpu0',
#                         dest="device", help="cpu or gpuN")
#     parser.add_argument("--initial_epoch", type=int,
#                         dest="initial_epoch", default=0,
#                         help="initial epoch")
#     parser.add_argument("--epochs", type=int,
#                         dest="epochs", default=100,
#                         help="number of epoch")
#     parser.add_argument("--steps_per_epoch", type=int,
#                         dest="steps_per_epoch", default=1000,
#                         help="iterations of each epoch")
#     parser.add_argument("--batch_size", type=int,
#                         dest="batch_size", default=1,
#                         help="batch size")
#
#     args = parser.parse_args()
#     train(**vars(args))


import os
import time
import numpy as np
import torch
import torch.optim as optim
from argparse import ArgumentParser

# Assuming these are your modules (make sure they are importable)
import network2D
import losses2D
import datagenerators2D

# --- Keep your Dice and NJD functions here ---
def Dice(vol1, vol2, labels=None, nargout=1):
    # ... (your Dice function code) ...
    if labels is None:
        labels = np.unique(np.concatenate((vol1.flatten(), vol2.flatten()))) # Flatten needed for unique
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    if len(labels) == 0:
        # Handle cases where only background is present or inputs are empty
         return np.array([1.0]) if np.sum(vol1 == 0) > 0 and np.sum(vol2 == 0) > 0 else np.array([0.0])


    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        # bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        # Avoid division by zero if both segmentations are empty for this label
        if bottom == 0:
            dicem[idx] = 1.0 if top == 0 else 0.0 # If top is also 0, perfect match (empty); otherwise mismatch
        else:
             dicem[idx] = top / bottom

    if nargout == 1:
        return np.mean(dicem) # Return mean Dice over labels by default
    else:
        return (dicem, labels)

def NJD(displacement):
    # Ensure displacement is 3D (H, W, C)
    if displacement.ndim != 3 or displacement.shape[-1] != 2:
         print(f"Warning: Unexpected displacement shape for NJD: {displacement.shape}. Expected (H, W, 2)")
         # Decide how to handle: return NaN, 0, or raise error
         return np.nan # Or 0, or raise ValueError

    # Check if displacement map is large enough
    if displacement.shape[0] < 2 or displacement.shape[1] < 2:
        print(f"Warning: Displacement map too small for NJD calculation: {displacement.shape}")
        return 0.0 # Or np.nan

    # dy(y+1, x) - dy(y, x)
    D_y_y = displacement[1:, :-1, 0] - displacement[:-1, :-1, 0]
    # dx(y+1, x) - dx(y, x)
    D_x_y = displacement[1:, :-1, 1] - displacement[:-1, :-1, 1]

    # dy(y, x+1) - dy(y, x)
    D_y_x = displacement[:-1, 1:, 0] - displacement[:-1, :-1, 0]
    # dx(y, x+1) - dx(y, x)
    D_x_x = displacement[:-1, 1:, 1] - displacement[:-1, :-1, 1]

    Ja_det = (D_x_x + 1) * (D_y_y + 1) - D_x_y * D_y_x

    num_folding = np.sum(Ja_det <= 0) # Count non-positive values


    return num_folding # Return the count of folding points


def train(train_dir,
          train_pairs,
          valid_dir,
          valid_pairs,
          model_dir,
          load_model,
          device,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size):

    # preparation
    train_pairs = np.load(os.path.join(train_dir, train_pairs), allow_pickle=True)
    valid_pairs = np.load(os.path.join(valid_dir, valid_pairs), allow_pickle=True)

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir) # Use makedirs to create parent dirs if needed

    # device handling
    if 'gpu' in device:
        gpu_id = device.split('gpu')[-1]
        if not gpu_id: # Handle 'gpu' case
            gpu_id = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        if torch.cuda.is_available():
            print(f"Using GPU: {gpu_id}")
            device = 'cuda'
            torch.backends.cudnn.deterministic = True # May impact performance
            # torch.backends.cudnn.benchmark = True # Usually good for performance if input sizes don't vary much
        else:
            print(f"CUDA not available, falling back to CPU.")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            device = 'cpu'

    else:
        print("Using CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    model = network2D.CorrMLP(in_channels=1) # Specify in_channels if needed by your constructor
    model.to(device)
    if load_model and os.path.isfile(load_model): # Check if it's a valid file
        print('Loading model weights from:', load_model)
        state_dict = torch.load(load_model, map_location=device)
        # Consider strict=False if loading partial weights or from a slightly different architecture
        model.load_state_dict(state_dict, strict=True)
    elif load_model and load_model != './': # Handle case where path is specified but not found
        print(f"Warning: Model file not found at {load_model}. Starting from scratch.")

    SpatialTransformer = network2D.SpatialTransformer_block(mode='nearest')
    SpatialTransformer.to(device)
    SpatialTransformer.eval() # Transformer usually doesn't have dropout/BN, but good practice

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Consider lr schedulers for longer runs

    Losses = [losses2D.NCC(win=9).loss, losses2D.Grad('l2').loss] # Use appropriate loss modules
    Weights = [1.0, 1.0] # Adjust weights as needed

    train_gen_pairs = datagenerators2D.gen_pairs(train_dir, train_pairs, batch_size=batch_size)
    train_gen = datagenerators2D.gen_s2s(train_gen_pairs, batch_size=batch_size) # This should yield batches

    print(f"--- Starting Training ---")
    print(f"Epochs: {epochs}, Steps/Epoch: {steps_per_epoch}, Batch Size: {batch_size}")
    print(f"Device: {device}")
    # training/validate loops
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()

        # --- Training Phase ---
        model.train()
        epoch_train_losses = [] # Stores loss values for each step in the epoch
        # epoch_train_total_loss = [] # Stores the combined loss for each step

        for step in range(steps_per_epoch):

            try:
                # Assuming generator yields ([np_fixed_batch, np_moving_batch], [np_fixed_batch, np_moving_batch])
                inputs_np, labels_np = next(train_gen)
            except StopIteration:
                # Handle generator exhaustion if steps_per_epoch is larger than dataset size / batch_size
                print("Warning: Training data generator exhausted. Restarting generator.")
                train_gen_pairs = datagenerators2D.gen_pairs(train_dir, train_pairs, batch_size=batch_size)
                train_gen = datagenerators2D.gen_s2s(train_gen_pairs, batch_size=batch_size)
                inputs_np, labels_np = next(train_gen)

            inputs = [torch.from_numpy(img).unsqueeze(1).to(device).float() for img in inputs_np]
            # labels = [torch.from_numpy(img).unsqueeze(1).to(device).float() for img in labels_np] # If labels are also images for loss calc

            fixed_img_tensor = inputs[0]
            moving_img_tensor = inputs[1]

            # run inputs through the model
            # Model call: model(fixed, moving)
            warped_moving_pred, flow_pred = model(fixed_img_tensor, moving_img_tensor)

            loss_inputs_pred = [warped_moving_pred, flow_pred]
            loss_inputs_target = [fixed_img_tensor, flow_pred] # Target for NCC is fixed, Grad loss operates on flow itself

            # calculate total loss
            step_loss = 0
            step_loss_list = []
            for i, loss_func in enumerate(Losses):
                 # Adjust loss call based on what each function expects
                 # e.g., Loss(y_true, y_pred) or Loss(y_pred)
                 if i == 0: # NCC Loss - expects (target, prediction)
                     curr_loss = loss_func(loss_inputs_target[i], loss_inputs_pred[i]) * Weights[i]
                 elif i == 1: # Grad Loss - likely expects (prediction)
                     curr_loss = loss_func(loss_inputs_pred[i]) * Weights[i]
                 else: # Handle other potential losses
                     # You might need different targets/preds here
                     curr_loss = loss_func(loss_inputs_target[i], loss_inputs_pred[i]) * Weights[i]

                 step_loss_list.append(curr_loss.item())
                 step_loss += curr_loss

            epoch_train_losses.append(step_loss_list) # Append list of individual losses for this step
            # epoch_train_total_loss.append(step_loss.item()) # Append total loss for this step

            # backpropagate and optimize
            optimizer.zero_grad()
            step_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Optional: Gradient clipping
            optimizer.step()
        # --- Validation Phase ---
        model.eval()
        epoch_valid_Dice = []
        epoch_valid_NJD = []
        with torch.no_grad(): # Disable gradient calculations for validation
            for valid_pair in valid_pairs:

                # Load single validation pair (ensure numpy, handle channels if necessary)
                # Assuming load_by_name returns (H, W) numpy arrays
                fixed_vol_np, fixed_seg_np = datagenerators2D.load_by_name(valid_dir, valid_pair[0])
                moving_vol_np, moving_seg_np = datagenerators2D.load_by_name(valid_dir, valid_pair[1])

                # Add batch and channel dims, convert to tensor, move to device
                fixed_vol = torch.from_numpy(fixed_vol_np).unsqueeze(0).unsqueeze(0).to(device).float()
                moving_vol = torch.from_numpy(moving_vol_np).unsqueeze(0).unsqueeze(0).to(device).float()
                # Segmentation might need different handling depending on loss/warping
                # If warping seg: add batch/channel, move to device
                # Make sure channel dim is added correctly (1 for grayscale/single-label seg)
                moving_seg = torch.from_numpy(moving_seg_np).unsqueeze(0).unsqueeze(0).to(device).float() # Or long() if labels


                # Run model: model(fixed, moving) -> warped_moving, flow
                warped_vol_pred, flow_pred = model(fixed_vol, moving_vol)

                # Warp moving segmentation using the predicted flow
                # Input to transformer: (moving_seg_tensor, flow_pred)
                warped_seg_pred = SpatialTransformer(moving_seg, flow_pred) # Ensure moving_seg has correct shape (B, C, H, W)


                # --- Calculate Metrics ---
                # Move predictions and ground truth to CPU, convert to numpy
                warped_seg_np = warped_seg_pred.squeeze().cpu().numpy() # Remove batch/channel dims
                fixed_seg_np = fixed_seg_np # Already numpy (H, W)
                flow_np = flow_pred.squeeze().permute(1, 2, 0).cpu().numpy() # B C H W -> H W C


                # Calculate Dice (ensure inputs are numpy arrays, handle labels if multi-class)
                # Dice function expects (vol1, vol2)
                dice_val = Dice(warped_seg_np, fixed_seg_np) # Use the fixed seg as ground truth
                epoch_valid_Dice.append(dice_val)


                # Calculate NJD (ensure flow is numpy H, W, 2)
                njd_val = NJD(flow_np)
                if not np.isnan(njd_val): # Append only if valid number
                    epoch_valid_NJD.append(njd_val)


        # --- Epoch End Summary ---
        elapsed_time = time.time() - start_time

        # Calculate average losses and metrics for the epoch
        avg_train_loss_total = np.mean([sum(step_losses) for step_losses in epoch_train_losses]) if epoch_train_losses else 0
        avg_valid_dice = np.mean(epoch_valid_Dice) if epoch_valid_Dice else 0
        avg_valid_njd = np.mean(epoch_valid_NJD) if epoch_valid_NJD else 0 # Handle case with no valid NJD values
        # Construct the desired output string
        progress_string = f"Epoch {epoch + 1}/{epochs}"
        progress_string += f" - Time: {elapsed_time:.2f}s"
        progress_string += f" - Train Loss: {avg_train_loss_total:.4f}"

        # Optionally add breakdown of average individual training losses
        if epoch_train_losses:
             avg_individual_losses = np.mean(epoch_train_losses, axis=0)
             loss_breakdown = ', '.join([f"{l:.4f}" for l in avg_individual_losses])
             progress_string += f" ({loss_breakdown})" # e.g., (NCC_loss, Grad_loss)

        progress_string += f" - Valid Dice: {avg_valid_dice:.4f}"
        progress_string += f" - Valid NJD: {avg_valid_njd:.2f}" # Changed format specifier for NJD
        print(progress_string, flush=True) # Print the summary

        save_path = os.path.join(model_dir, f'epoch_{epoch+1:03d}.pt') # Use formatted epoch number
        torch.save(model.state_dict(), save_path)
    print("--- Training Finished ---")


if __name__ == "__main__":
    parser = ArgumentParser()

    # --- Default paths adjusted assuming standard project structure ---
    parser.add_argument("--train_dir", type=str,
                        dest="train_dir", default='./data/pairs/', # More specific path
                        help="training folder (containing images/segmentations)")
    parser.add_argument("--train_pairs", type=str,
                        dest="train_pairs", default='train_pairs.npy', # Name of the pairs file inside train_dir
                        help="training pairs file (.npy)")
    parser.add_argument("--valid_dir", type=str,
                        dest="valid_dir", default='./data/pairs/', # More specific path
                        help="validation folder")
    parser.add_argument("--valid_pairs", type=str,
                        dest="valid_pairs", default='valid_pairs.npy', # Name of the pairs file inside valid_dir
                        help="validation pairs file (.npy)")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./models/',
                        help="models folder")
    # Changed load_model default to None or empty string to signify no loading by default
    parser.add_argument("--load_model", type=str,
                        dest="load_model", default=None,
                        help="load model file to initialize with (full path)")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="cpu or gpuN (e.g., gpu0)")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial epoch (if resuming training)")
    parser.add_argument("--epochs", type=int,
                        dest="epochs", default=100,
                        help="number of epoch")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100, # Reduced for faster testing, increase for real runs
                        help="iterations of each epoch")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1, # Adjust based on GPU memory
                        help="batch size")

    args = parser.parse_args()
    train(**vars(args))