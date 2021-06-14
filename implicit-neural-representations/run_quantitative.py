from nn_mri import ImageFitting_set, Siren, get_mgrid, cases, calculate_contrast
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

out_folder = '../experiments/'

# TODO: Make the following as arguments to be inputted
total_steps = 3000
seg = 50
gland_start = 40
focus_size = 50
weighted = True
sigma_est = 2
hidden_layers = 2
hidden_features = 256
scale = 1
patch_kw = dict(patch_size=3, patch_distance=3)

metrics = ['C', 'CNR']

method_name = 'sr1'
exp_no = 4
filename = os.path.join(out_folder, method_name + '_exp_' + str(exp_no) + '.csv')
with open(filename, 'w') as f:
    f.write('seed,patient,direction,epoch,image,metric,performance\n')

for seed in range(5):
    torch.manual_seed(seed)
    for case in cases:
        print(case.pt_id)
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
            img_siren = Siren(in_features=2, out_features=1, hidden_layers=hidden_layers, 
                         hidden_features=hidden_features)
            img_siren.cuda()
            torch.cuda.empty_cache()
            optim = torch.optim.Adam(lr=0.0003, params=img_siren.parameters())
            ctr = 0
            for step in tqdm(range(total_steps)):
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
                        out_img = predicted/(ctr-49)
                    ctr += 1
                    nlm = denoise_nl_means(out_img, h=1.15 * sigma_est, fast_mode=True, **patch_kw)
                    orig2 = rescale(orig, scale, anti_aliasing=False)
                    images = {'mean':orig2, 'reconst':pr, 'superres':out_img, 'NLM':nlm}

                    with open(filename, 'a') as f:
                        for image in images.keys():
                            for inx, metric in enumerate(metrics):
                                f.write('{},{},{},{},{},{},{}\n'.format(seed, pt_no, directions[direction], step,
                                                                        image, metric,
                                                                        calculate_contrast(case, 
                                                                                           scale,
                                                                                           images[image],
                                                                                           gland_start)[inx]))

            predicted_XYZ.append(out_img)
            
        predicted = sum(predicted_XYZ)/len(predicted_XYZ)
        orig = sum(original_XYZ)/len(original_XYZ)    
        noisy = predicted
        denoise = denoise_nl_means(noisy, h=1.15 * sigma_est, fast_mode=False,
                                       **patch_kw)
        nlm = denoise
        out_img = noisy
        with open(filename, 'a') as f:
            for image in images.keys():
                for inx, metric in enumerate(metrics):
                    f.write('{},{},{},{},{},{},{}\n'.format(seed, pt_no, 'x+y+z', ((total_steps-1)//seg)*seg,
                                                            image, metric,
                                                            calculate_contrast(case,
                                                                               scale,
                                                                               images[image],
                                                                               gland_start)[inx]))