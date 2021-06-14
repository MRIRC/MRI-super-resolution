""" This script automatically runs the super-resolution method for the test 
images and saves them as dicom files"""
from nn_mri import ImageFitting_set, Siren, get_mgrid, cases, calculate_contrast, save_dicom
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from skimage.color import rgb2gray, gray2rgb

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
from skimage.transform import rescale, resize, downscale_local_mean

import os

out_folder = '../output_images/'

# TODO: Make the following as arguments to be inputted
seg = 50

total_steps = 3000
gland_start = 40
focus_size = 50
scale = 3
sigma_est = 2
hidden_layers = 2
hidden_features = 256
weighted = False
patch_kw = dict(patch_size=3, patch_distance=3)
method_name = 'sr1'
exp_no = 4



for case in cases:
    _slice = case.cancer_slice

    predicted_XYZ = []
    original_XYZ = []
    directions = ['x', 'y', 'z']
    for direction in range(3):  # gradient directions x, y, z
        ends = np.cumsum(case.acquisitions)
        starts = ends - case.acquisitions

        # Create a dataset for training SIREN
        img_dataset = []
        for acq in range(starts[direction], ends[direction]):
            img = case.dwi[gland_start : gland_start + focus_size,
                           gland_start : gland_start + focus_size,
                           _slice,
                           acq]
            img_dataset.append(Image.fromarray(img))

        dataset = ImageFitting_set(img_dataset)
        orig = dataset.mean
        pt_no = case.pt_id.split('-')[-1]

        original_XYZ.append(orig)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
        img_siren = Siren(in_features=2, out_features=1, hidden_features=hidden_features, 
                     hidden_layers=hidden_layers)
        img_siren.cuda()
        torch.cuda.empty_cache()
        optim = torch.optim.Adam(lr=0.0003, params=img_siren.parameters())
        ctr = 0
        for step in range(total_steps):
            size = dataset.shape
            for sample in range(len(dataset)):                    
                ground_truth, model_input  = dataset.pixels[sample], dataset.coords[sample]
                ground_truth, model_input = ground_truth.cuda(), model_input.cuda()
                model_output, coords = img_siren(model_input)
                if weighted:
                    weights = ground_truth/ground_truth.sum()
                    weights -= weights.min()
                    weights += 0.000001
                else:
                    weights = 1
                loss = weights*(model_output - ground_truth)**2
                loss = loss.mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
            if not step % seg:
                coords2 = get_mgrid(size[0]*scale, 2).cuda()
                superres, _ = img_siren(coords2)
                pr = superres.cpu().view(scale*size[0], scale* size[1]).detach().numpy()
                if ctr < 50:
                    predicted = superres.cpu().view(scale*size[0], scale* size[1]).detach().numpy()
                    out_img = predicted
                else:
                    predicted += superres.cpu().view(scale*size[0], scale* size[1]).detach().numpy()
                    out_img = predicted/float(ctr-49)
                ctr += 1
                nlm = denoise_nl_means(out_img, h=1.15 * sigma_est, fast_mode=True, **patch_kw)


        predicted_XYZ.append(out_img)
        im_name = method_name + '_exp_' + str(exp_no) + '_' + pt_no + '_mean_' + directions[direction] + '.dcm'
        filename = os.path.join(out_folder, im_name)
        save_dicom(orig, filename)
        im_name = method_name + '_exp_' + str(exp_no) + '_' + pt_no + '_super_' + directions[direction] + '.dcm'
        filename = os.path.join(out_folder, im_name)
        save_dicom(out_img, filename)

    
    predicted = sum(predicted_XYZ)/len(predicted_XYZ)
    orig = sum(original_XYZ)/len(original_XYZ)
    
    nlm = denoise_nl_means(predicted, h=1.15 * sigma_est, fast_mode=False,**patch_kw)

    filename = os.path.join(out_folder, method_name + '_exp_' + str(exp_no) + '_' + pt_no + '_mean.dcm')
    save_dicom(orig, filename)
    filename = os.path.join(out_folder, method_name + '_exp_' + str(exp_no) + '_' + pt_no + '_super.dcm')
    save_dicom(predicted, filename)
    filename = os.path.join(out_folder, method_name + '_exp_' + str(exp_no) + '_' + pt_no + '_NLM.dcm')
    save_dicom(nlm, filename)