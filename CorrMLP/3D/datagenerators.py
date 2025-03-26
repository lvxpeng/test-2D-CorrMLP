import os, sys
import numpy as np
import scipy.ndimage
        
        
def gen_s2s(gen, batch_size=1):
    """
    Generates samples for training a model that transforms one image into another.

    This function is a generator function that continuously generates training samples. Each sample consists of two images:
    a fixed image and a moving image. The moving image is what the model learns to transform into the fixed image.

    Parameters:
        gen: A generator that yields batches of input data, which includes both fixed and moving images.
        batch_size: The number of samples per batch (default is 1).

    Yields:
        A tuple where the first element is a list containing the fixed and moving images, and the second element is a list
        containing the fixed image again and a zero tensor. This design is to meet the input requirements of certain models,
        where the zero tensor may serve as a placeholder for specific training objectives.
    """
    while True:
        # Get the next batch of data from the generator
        X = next(gen)
        # Extract the fixed and moving images from the batch
        fixed = X[0]
        moving = X[1]

        # generate a zero tensor
        Zero = np.zeros((1))

        # Yield the processed batch of data, including the fixed and moving images, as well as the fixed image and a zero tensor for training
        yield ([fixed, moving], [fixed, Zero])

        

def gen_pairs(path, pairs, batch_size=1):
    """
    生成配对图像数据的生成器函数。

    参数:
    path (str): 图像数据的路径。
    pairs (list): 包含配对图像名称的列表，每个元素是一个包含两个图像名称的元组。
    batch_size (int): 每次生成的图像对的数量，默认为1。

    生成:
    一个包含配对图像数据的元组，元组中的第一个元素是固定图像的批量数据，第二个元素是移动图像的批量数据。
    """
    # 获取图像对的数量
    pairs_num = len(pairs)
    while True:
        # 随机选择batch_size数量的图像对索引
        idxes = np.random.randint(pairs_num, size=batch_size)

        # load fixed images
        X_data = []
        for idx in idxes:
            # 解码图像对中固定图像的名称，并加载图像数据
            fixed = bytes.decode(pairs[idx][0])
            X = load_volfile(path+fixed, np_var='img')
            # 为图像数据添加新轴以匹配网络输入格式
            X = X[np.newaxis, np.newaxis, ...]
            X_data.append(X)
        # 根据batch_size调整返回的图像数据格式
        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # load moving images
        X_data = []
        for idx in idxes:
            # 解码图像对中移动图像的名称，并加载图像数据
            moving = bytes.decode(pairs[idx][1])
            X = load_volfile(path+moving, np_var='img')
            # 为图像数据添加新轴以匹配网络输入格式
            X = X[np.newaxis, np.newaxis, ...]
            X_data.append(X)
        # 根据batch_size调整返回的图像数据格式
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])

        # 生成并返回配对图像数据
        yield tuple(return_vals)


        
def load_by_name(path, name):
    """
    根据文件路径和名称加载医学图像数据及其标签。

    该函数通过拼接路径和文件名来加载指定的npz文件，该文件应包含'img'和'label'两个numpy数组，
    分别代表图像数据和标签数据。函数会将这两个数据分别进行预处理，然后以元组的形式返回。

    参数:
    - path: 医学图像数据文件的路径，类型为字符串。
    - name: 文件名，类型为字节串，需要解码为字符串以便于拼接路径。

    返回:
    - 一个包含两个元素的元组，第一个元素是图像数据数组，第二个元素是标签数据数组。
      每个数据数组的形状为[1, 1, *original_shape]，其中*original_shape表示原始数据的形状。
    """
    # 加载指定路径的npz文件，该文件包含了图像数据和标签数据
    npz_data = load_volfile(path+bytes.decode(name), np_var='all')

    # 提取图像数据，并通过添加两个维度以适应后续处理或模型输入的要求
    X = npz_data['img']
    X = X[np.newaxis, np.newaxis, ...]
    return_vals = [X]

    # 提取标签数据，并进行与图像数据相同的维度处理
    X = npz_data['label']
    X = X[np.newaxis, np.newaxis, ...]
    return_vals.append(X)

    # 返回包含图像数据和标签数据的元组
    return tuple(return_vals)



def load_volfile(datafile, np_var):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        if np_var == 'all':
            X = X = np.load(datafile)
        else:
            X = np.load(datafile)[np_var]

    return X
