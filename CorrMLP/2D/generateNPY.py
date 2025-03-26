# generateNPY.py
import os
import sys
import numpy as np
from PIL import Image

def load_volfile(datafile, np_var='vol_data'):
    """
    Load volume file
    formats: nii, nii.gz, mgz, npz, png
    if it's a npz (compressed numpy), variable names in np_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try:
                import nibabel as nib
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')
                return None

        X = nib.load(datafile).get_fdata()

    elif datafile.endswith('.npz'):
        if np_var == 'all':
            X = np.load(datafile)
        else:
            X = np.load(datafile)[np_var]

    elif datafile.endswith('.png'):
        X = np.array(Image.open(datafile).convert('L'))  # Convert to grayscale

    # Normalize the image data to the range [0, 1]
    X = X / np.max(X)

    return X

# 修改create_pairs函数中的保存部分
def create_pairs(fixed_dir, moved_dir, output_file, split_ratio=0.8):
    """
    Create pairs of fixed and moving images and split them into training and validation sets.

    Parameters:
    - fixed_dir: Directory containing the fixed image (atlas).
    - moved_dir: Directory containing the moving images.
    - output_file: Base name for the output .npy files (without extension).
    - split_ratio: Ratio of data to be used for training (default is 0.8).
    """
    # List all image files in the moved directory
    moved_files = [f for f in os.listdir(moved_dir) if f.endswith('.nii.gz')]

    # Sort image files to ensure consistent pairing
    moved_files.sort()

    # Load the fixed image (atlas)
    fixed_image = os.listdir(fixed_dir)[0]
    fixed_image_path = os.path.join(fixed_dir, fixed_image)

    # Create pairs of fixed and moving images
    pairs = [(fixed_image.encode('utf-8'), moved_image.encode('utf-8')) for moved_image in moved_files]

    # Shuffle pairs to ensure randomness
    np.random.shuffle(pairs)

    # Split pairs into training and validation sets
    split_index = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_index]
    valid_pairs = pairs[split_index:]

    train_file = output_file + 'train_pairs.npy'
    valid_file = output_file + 'valid_pairs.npy'

    # Save pairs to .npy files
    np.save(train_file, train_pairs, allow_pickle=True)
    np.save(valid_file, valid_pairs, allow_pickle=True)

    print(f"Training pairs saved to {train_file}")
    print(f"Validation pairs saved to {valid_file}")

    # Load and print the contents of the .npy files
    train_data = np.load(train_file, allow_pickle=True)
    valid_data = np.load(valid_file, allow_pickle=True)

    print("\nTraining pairs:")
    print(train_data)

    print("\nValidation pairs:")
    print(valid_data)


# Example usage
fixed_directory = './data/fixed'  # Path to the fixed image directory
moved_directory = './data/moved'  # Path to the moved images directory
output_base_name = './data/pairs/'  # Base name for the output .npy files
create_pairs(fixed_directory, moved_directory, output_base_name)
