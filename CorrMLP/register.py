import os.path

import nibabel
import numpy as np
import torch
import datagenerators as dg
import argparse

from networks import  CorrMLP


def register(moving_img,fixed_img,model_path):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # 加载固定图像
    fixed_img = dg.load_volfile(fixed_img)
    fixed_img_size = fixed_img.shape
    fixed_data = torch.from_numpy(fixed_img).to(device).float()[np.newaxis,np.newaxis,...]

    # 加载移动图像
    moving_img = dg.load_volfile(moving_img)
    moving_data = torch.from_numpy(moving_img).to(device).float()[np.newaxis,np.newaxis,...]

    # 加载模型
    model = CorrMLP()
    model.to(device)
    model.load_state_dict(torch.load(model_path,map_location=lambda storage,loc:storage))
    move_image,fixed_image =model(moving_data,fixed_data)
    fixed_vol = nibabel.load(fixed_img)
    save_path = os.path.join('./','saved/moved')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_moved_img(move_image,fixed_vol,save_path)


def save_moved_img(I_img,fixd_vol_file, savename):
    """
    I_img：需要保存的图像
    fixd_vol_file：固定图像 保存配准图像时需要使用固定图像的空间信息
    savename：保存文件名 也就是文件全路径
    """
    I_img = I_img[0, 0, ...].cpu().detach().numpy()
    # 使用固定图像的affine矩阵，描述了图像在物理空间中的位置和方向
    affine = fixd_vol_file.affine
    new_img = nibabel.nifti1.Nifti1Image(I_img, affine, header=None)
    nibabel.save(new_img, savename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving_img", type=str,default='',help='移动图像')
    parser.add_argument("--fixed_img", type=str,default='',help='固定图像')
    parser.add_argument("--model_path", type=str,default='',help='模型路径')
