# import os, sys
# import numpy as np
# import scipy.ndimage
# import nibabel as nib
#
#
# def gen_s2s(gen, batch_size=1):
#     """
#     生成训练样本，固定图像以及移动图像的一个组合
#     Generates samples for training a model that transforms one image into another.
#
#     This function is a generator function that continuously generates training samples. Each sample consists of two images:
#     a fixed image and a moving image. The moving image is what the model learns to transform into the fixed image.
#
#     Parameters:
#         gen: A generator that yields batches of input data, which includes both fixed and moving images.
#         batch_size: The number of samples per batch (default is 1).
#
#     Yields:
#         A tuple where the first element is a list containing the fixed and moving images, and the second element is a list
#         containing the fixed image again and a zero tensor. This design is to meet the input requirements of certain models,
#         where the zero tensor may serve as a placeholder for specific training objectives.
#     """
#     while True:
#         # Get the next batch of data from the generator
#         X = next(gen)
#         # Extract the fixed and moving images from the batch
#         fixed = X[0]
#         moving = X[1]
#
#         # generate a zero tensor
#         Zero = np.zeros_like((fixed))
#         print(f"Fixed shape: {fixed.shape}, Moving shape: {moving.shape}")
#
#         # Yield the processed batch of data, including the fixed and moving images, as well as the fixed image and a zero tensor for training
#         yield ([fixed, moving], [fixed, Zero])
#
#
# def gen_pairs(path, pairs, batch_size=1):
#     """
#     生成配对图像数据的生成器函数。
#
#     参数:
#     path (str): 图像数据的路径。
#     pairs (list): 包含配对图像名称的列表，每个元素是一个包含两个图像名称的元组。
#     batch_size (int): 每次生成的图像对的数量，默认为1。
#
#     生成:
#     一个包含配对图像数据的元组，元组中的第一个元素是固定图像的批量数据，第二个元素是移动图像的批量数据。
#     """
#     # 获取图像对的数量
#     pairs_num = len(pairs)
#     while True:
#         # 随机选择batch_size数量的图像对索引
#         idxes = np.random.randint(pairs_num, size=batch_size)
#
#         # load fixed images
#         X_data = []
#         for idx in idxes:
#             # 解码图像对中固定图像的名称，并加载图像数据
#             fixed = bytes.decode(pairs[idx][0])
#             X = load_volfile(path + fixed, np_var='img')
#             # 为图像数据添加新轴以匹配网络输入格式
#             X = X[np.newaxis, ...]
#             X_data.append(X)
#         # 根据batch_size调整返回的图像数据格式
#         if batch_size > 1:
#             return_vals = [np.concatenate(X_data, 0)]
#         else:
#             return_vals = [X_data[0]]
#
#         # load moving images
#         X_data = []
#         for idx in idxes:
#             # 解码图像对中移动图像的名称，并加载图像数据
#             moving = bytes.decode(pairs[idx][1])
#             X = load_volfile(path + moving, np_var='img')
#             # 为图像数据添加新轴以匹配网络输入格式
#             X = X[np.newaxis, ...]
#             X_data.append(X)
#         # 根据batch_size调整返回的图像数据格式
#         if batch_size > 1:
#             return_vals.append(np.concatenate(X_data, 0))
#         else:
#             return_vals.append(X_data[0])
#
#         # 生成并返回配对图像数据
#         yield tuple(return_vals)
#
#
# def load_by_name(path, name):
#     """
#     根据文件路径和名称加载医学图像数据及其标签。
#
#     该函数通过拼接路径和文件名来加载指定的npz文件，该文件应包含'img'和'label'两个numpy数组，
#     分别代表图像数据和标签数据。函数会将这两个数据分别进行预处理，然后以元组的形式返回。
#
#     参数:
#     - path: 医学图像数据文件的路径，类型为字符串。
#     - name: 文件名，类型为字节串，需要解码为字符串以便于拼接路径。
#
#     返回:
#     - 一个包含两个元素的元组，第一个元素是图像数据数组，第二个元素是标签数据数组。
#       每个数据数组的形状为[1, 1, *original_shape]，其中*original_shape表示原始数据的形状。
#     """
#     # 加载指定路径的npz文件，该文件包含了图像数据和标签数据
#     npz_data = load_volfile(path + bytes.decode(name), np_var='all')
#
#     # 提取图像数据，并通过添加两个维度以适应后续处理或模型输入的要求
#     X = npz_data['img']
#     X = X[np.newaxis, ...]
#     return_vals = [X]
#
#     # 提取标签数据，并进行与图像数据相同的维度处理
#     X = npz_data['label']
#     X = X[np.newaxis, ...]
#     return_vals.append(X)
#
#     # 返回包含图像数据和标签数据的元组
#     return tuple(return_vals)
#
#
# def load_volfile(datafile, np_var):
#     """
#     load volume file
#     formats: nii, nii.gz, mgz, npz
#     if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
#     """
#     assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'
#
#     if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
#         # import nibabel
#         # if 'nibabel' not in sys.modules:
#         #     try:
#         #         import nibabel as nib
#         #     except:
#         #         print('Failed to import nibabel. need nibabel library for these data file types.')
#
#         X = nib.load(datafile).get_fdata()
#
#     else:  # npz
#         if np_var == 'all':
#             X = X = np.load(datafile)
#         else:
#             X = np.load(datafile)[np_var]
#
#     return X
# datagenerators2D.py
# datagenerators2D.py

# datagenerators2D.py

import os
import glob
import json
import random
import logging
import numpy as np
from PIL import Image
import nibabel as nib
import torch
from torch.utils.data import Dataset

# --- Configure Logging ---
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

# --- Data Loading Classes ---

class BaseLoader:
    """Base class for loading data items with error handling."""
    def _load_or_log_error(self, path, load_func, *args, **kwargs):
        # Use os.path.normpath for better cross-platform path handling
        norm_path = os.path.normpath(path)
        if not os.path.exists(norm_path):
            log.error(f"File not found: {norm_path}")
            return None
        try:
            return load_func(norm_path, *args, **kwargs)
        except FileNotFoundError:
            log.error(f"File not found during loading: {norm_path}")
            return None
        except Exception as e:
            log.error(f"Failed to load or process file {norm_path}: {e}", exc_info=True) # Log traceback
            return None

class ImageLoader(BaseLoader):
    """Loads 2D images from various formats."""
    def __init__(self, npz_key='img', normalize=True):
        self.npz_key = npz_key
        self.normalize = normalize
        log.debug(f"ImageLoader initialized (npz_key='{npz_key}', normalize={normalize})") # Changed to debug

    def _get_extension(self, path):
        """Extracts the normalized file extension, handling composite ones like .nii.gz."""
        norm_path = os.path.normpath(path).lower()
        # Special case for .nii.gz
        if norm_path.endswith('.nii.gz'):
            return 'niigz'
        # General case: get the part after the last dot
        base, ext = os.path.splitext(norm_path)
        return ext.lstrip('.') # Remove the leading dot

    def _load_png(self, path):
        img = Image.open(path).convert('L')
        return np.array(img)

    def _load_npy(self, path):
        return np.load(path)

    def _load_npz(self, path):
        with np.load(path) as data:
            if self.npz_key not in data:
                log.error(f"Key '{self.npz_key}' not found in npz file: {path}. Available keys: {list(data.keys())}")
                return None
            return data[self.npz_key]

    def _load_nifti(self, path):
        img = nib.load(path)
        data = img.get_fdata()
        return np.squeeze(data).astype(np.float32)

    def _preprocess(self, img_np):
        if img_np.ndim != 2:
            log.warning(f"Image data expected to be 2D, got shape {img_np.shape}. Squeezing.")
            img_np = np.squeeze(img_np)
            if img_np.ndim != 2:
                 log.error(f"Could not reduce image to 2D. Final shape: {img_np.shape}")
                 return None
        if self.normalize:
            img_np = img_np.astype(np.float32)
            min_val, max_val = np.min(img_np), np.max(img_np)
            if max_val > min_val:
                img_np = (img_np - min_val) / (max_val - min_val)
            elif max_val > 0:
                 img_np = img_np / max_val
        return img_np[np.newaxis, :, :] # Add channel dim

    def load(self, path):
        """Loads and preprocesses an image based on its extension."""
        ext = self._get_extension(path) # <-- Use corrected extension getter
        img_np = None
        log.debug(f"Attempting to load image '{path}' with detected extension '{ext}'") # Debug log

        if ext == 'png':
            img_np = self._load_or_log_error(path, self._load_png)
        elif ext == 'npy':
            img_np = self._load_or_log_error(path, self._load_npy)
        elif ext == 'npz':
            img_np = self._load_or_log_error(path, self._load_npz)
        elif ext in ('niigz', 'nii'):
             img_np = self._load_or_log_error(path, self._load_nifti)
        else:
            log.error(f"Unsupported image file extension: '{ext}' for path {path}")
            return None

        if img_np is None:
            log.warning(f"Loading returned None for image: {path}")
            return None

        return self._preprocess(img_np)


class LabelLoader(BaseLoader):
    """Loads 2D segmentation labels/masks."""
    def __init__(self, npz_key='label', dtype=torch.float32):
        self.npz_key = npz_key
        self.dtype = dtype
        log.debug(f"LabelLoader initialized (npz_key='{npz_key}', dtype={dtype})") # Changed to debug

    # --- Use the same corrected extension getter ---
    def _get_extension(self, path):
        """Extracts the normalized file extension, handling composite ones like .nii.gz."""
        norm_path = os.path.normpath(path).lower()
        if norm_path.endswith('.nii.gz'):
            return 'niigz'
        base, ext = os.path.splitext(norm_path)
        return ext.lstrip('.')
    # --------------------------------------------

    def _load_png(self, path):
        img = Image.open(path).convert('L')
        return np.array(img)

    def _load_npy(self, path):
        return np.load(path)

    def _load_npz(self, path):
        with np.load(path) as data:
            if self.npz_key not in data:
                log.error(f"Key '{self.npz_key}' not found in npz file: {path}. Available keys: {list(data.keys())}")
                return None
            return data[self.npz_key]

    def _load_nifti(self, path):
        img = nib.load(path)
        data = img.get_fdata(dtype=np.int32)
        return np.squeeze(data)

    def _load_json(self, path):
        log.warning(f"Loading JSON label from {path}. Returning parsed data, not a mask array.")
        with open(path, 'r') as f:
            return json.load(f)

    def _preprocess(self, label_data):
        if not isinstance(label_data, np.ndarray):
            log.error(f"Label data loaded is not a NumPy array (type: {type(label_data)}). Cannot preprocess into mask tensor.")
            return None
        if label_data.ndim != 2:
            log.warning(f"Label data expected to be 2D, got shape {label_data.shape}. Squeezing.")
            label_data = np.squeeze(label_data)
            if label_data.ndim != 2:
                 log.error(f"Could not reduce label to 2D. Final shape: {label_data.shape}")
                 return None
        label_np = label_data[np.newaxis, :, :]
        return torch.tensor(label_np, dtype=self.dtype)

    def load(self, path):
        """Loads and preprocesses a label based on its extension."""
        ext = self._get_extension(path) # <-- Use corrected extension getter
        label_data = None
        log.debug(f"Attempting to load label '{path}' with detected extension '{ext}'") # Debug log

        if ext == 'png':
            label_data = self._load_or_log_error(path, self._load_png)
        elif ext == 'npy':
            label_data = self._load_or_log_error(path, self._load_npy)
        elif ext == 'npz':
            label_data = self._load_or_log_error(path, self._load_npz)
        elif ext in ('niigz', 'nii'):
            label_data = self._load_or_log_error(path, self._load_nifti)
        elif ext == 'json':
            label_data = self._load_or_log_error(path, self._load_json)
            if not isinstance(label_data, np.ndarray):
                log.warning(f"JSON label loaded from {path} is not directly usable as a mask array.")
                return None
        else:
            log.error(f"Unsupported label file extension: '{ext}' for path {path}")
            return None

        if label_data is None:
            log.warning(f"Loading returned None for label: {path}")
            return None

        return self._preprocess(label_data)

# --- RegistrationDataset and load_validation_pair remain unchanged below ---
# ... (Rest of datagenerators2D.py) ...

class RegistrationDataset(Dataset):
    """
    PyTorch Dataset for 2D registration tasks. Supports atlas-based and pair-based loading.

    Args:
        samples (list): A list defining the dataset samples.
            - Atlas Mode: List of paths to moving images `[mov_img_path1, mov_img_path2, ...]`.
                        Optionally, can be `[(mov_img_path1, mov_seg_path1), ...]`.
            - Pair Mode: List of tuples, each containing paths for a pair:
                         `(fixed_img_path, moving_img_path)` or
                         `(fixed_img_path, moving_img_path, fixed_seg_path, moving_seg_path)`.
        atlas_paths (dict, optional): Dictionary with paths for the fixed atlas. Required for Atlas Mode.
                                      Example: {'image': 'path/to/atlas.nii.gz', 'label': 'path/to/atlas_seg.nii.gz'}
                                      Set label to None if not available.
        image_loader (ImageLoader): An instance of ImageLoader.
        label_loader (LabelLoader, optional): An instance of LabelLoader. Required if segmentations are used.
        mode (str): 'atlas' or 'pair'. Determined automatically if not provided.
        transforms (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, samples, image_loader, label_loader=None, atlas_paths=None, mode=None, transforms=None):
        super().__init__()
        self.samples = samples
        self.image_loader = image_loader
        self.label_loader = label_loader
        self.transforms = transforms
        self.atlas_paths = atlas_paths

        # Determine mode
        if mode:
            self.mode = mode
        else:
            if atlas_paths:
                self.mode = 'atlas'
            elif isinstance(samples[0], (tuple, list)) and len(samples[0]) >= 2:
                self.mode = 'pair'
            else:
                raise ValueError("Cannot automatically determine mode. Provide 'mode' or 'atlas_paths'.")

        log.info(f"RegistrationDataset initialized in '{self.mode}' mode with {len(samples)} samples.")

        # Pre-load atlas in atlas mode
        self.atlas_image = None
        self.atlas_label = None
        if self.mode == 'atlas':
            if not atlas_paths or 'image' not in atlas_paths:
                raise ValueError("Atlas paths (including 'image') required for atlas mode.")
            # --- Load Atlas Image ---
            log.info(f"Loading atlas image from: {atlas_paths['image']}")
            self.atlas_image = self.image_loader.load(atlas_paths['image'])
            if self.atlas_image is None:
                 raise RuntimeError(f"Failed to load atlas image: {atlas_paths['image']}")
            log.info(f"Atlas image loaded successfully. Shape: {self.atlas_image.shape}")

            # --- Load Atlas Label (Optional) ---
            if atlas_paths.get('label') and self.label_loader:
                log.info(f"Loading atlas label from: {atlas_paths['label']}")
                self.atlas_label = self.label_loader.load(atlas_paths['label'])
                if self.atlas_label is None:
                    log.warning(f"Failed to load atlas label: {atlas_paths['label']}, proceeding without it.")
                else:
                    log.info(f"Atlas label loaded successfully. Shape: {self.atlas_label.shape}")
            elif atlas_paths.get('label'):
                 log.warning("Atlas label path provided, but no LabelLoader initialized.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a sample dictionary containing torch tensors:
        'fixed_image', 'moving_image'
        Optional: 'fixed_label', 'moving_label'
        """
        sample_info = self.samples[idx]
        data = {}

        if self.mode == 'atlas':
            data['fixed_image'] = self.atlas_image # Use pre-loaded atlas
            if self.atlas_label is not None:
                data['fixed_label'] = self.atlas_label # Use pre-loaded atlas label

            moving_img_path = sample_info[0] if isinstance(sample_info, (tuple, list)) else sample_info
            moving_label_path = sample_info[1] if isinstance(sample_info, (tuple, list)) and len(sample_info) > 1 else None

            data['moving_image'] = self.image_loader.load(moving_img_path)
            if data['moving_image'] is None:
                 log.error(f"Failed to load moving image for sample {idx}: {moving_img_path}")
                 return None # Let DataLoader handle this (e.g., skip)

            if moving_label_path and self.label_loader:
                moving_label_tensor = self.label_loader.load(moving_label_path)
                if moving_label_tensor is not None:
                    data['moving_label'] = moving_label_tensor
                else:
                    log.warning(f"Failed to load moving label for sample {idx}: {moving_label_path}")
            elif moving_label_path:
                log.warning(f"Moving label path {moving_label_path} provided for sample {idx}, but no LabelLoader.")


        elif self.mode == 'pair':
            fixed_img_path, moving_img_path = sample_info[0], sample_info[1]
            fixed_label_path = sample_info[2] if len(sample_info) > 2 else None
            moving_label_path = sample_info[3] if len(sample_info) > 3 else None

            data['fixed_image'] = self.image_loader.load(fixed_img_path)
            if data['fixed_image'] is None: return None
            data['moving_image'] = self.image_loader.load(moving_img_path)
            if data['moving_image'] is None: return None

            if fixed_label_path and self.label_loader:
                fixed_label_tensor = self.label_loader.load(fixed_label_path)
                if fixed_label_tensor is not None: data['fixed_label'] = fixed_label_tensor
                else: log.warning(f"Failed to load fixed label for sample {idx}: {fixed_label_path}")
            elif fixed_label_path: log.warning(f"Fixed label path {fixed_label_path} provided but no LabelLoader.")

            if moving_label_path and self.label_loader:
                moving_label_tensor = self.label_loader.load(moving_label_path)
                if moving_label_tensor is not None: data['moving_label'] = moving_label_tensor
                else: log.warning(f"Failed to load moving label for sample {idx}: {moving_label_path}")
            elif moving_label_path: log.warning(f"Moving label path {moving_label_path} provided but no LabelLoader.")

        else:
            raise ValueError(f"Unknown dataset mode: {self.mode}")

        if self.transforms:
            data = self.transforms(data)

        # Convert numpy arrays to tensors before returning (if not already tensors)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.from_numpy(value)
            # Ensure labels have the correct dtype if they are tensors
            elif isinstance(value, torch.Tensor) and 'label' in key and self.label_loader:
                 data[key] = value.to(self.label_loader.dtype)


        return data


def load_validation_pair(fixed_img_path, moving_img_path,
                         fixed_label_path=None, moving_label_path=None,
                         image_loader=None, label_loader=None, device='cpu'):
    """Loads single validation pair as tensors on device with batch dim."""
    # ... (load_validation_pair function remains unchanged) ...
    if image_loader is None: image_loader = ImageLoader()
    if label_loader is None and (fixed_label_path or moving_label_path):
        label_loader = LabelLoader()

    data = {}
    log.debug(f"Loading validation pair: Fixed='{fixed_img_path}', Moving='{moving_img_path}'")

    fixed_img_np = image_loader.load(fixed_img_path)
    if fixed_img_np is None: return None
    data['fixed_image'] = torch.from_numpy(fixed_img_np).unsqueeze(0).to(device)

    moving_img_np = image_loader.load(moving_img_path)
    if moving_img_np is None: return None
    data['moving_image'] = torch.from_numpy(moving_img_np).unsqueeze(0).to(device)

    if fixed_label_path and label_loader:
        fixed_label_tensor = label_loader.load(fixed_label_path)
        if fixed_label_tensor is not None:
            data['fixed_label'] = fixed_label_tensor.unsqueeze(0).to(device)
        else:
            log.warning(f"Failed to load validation fixed label: {fixed_label_path}")

    if moving_label_path and label_loader:
        moving_label_tensor = label_loader.load(moving_label_path)
        if moving_label_tensor is not None:
            data['moving_label'] = moving_label_tensor.unsqueeze(0).to(device)
        else:
            log.warning(f"Failed to load validation moving label: {moving_label_path}")

    log.debug("Validation pair loaded successfully.")
    return data






