from nn_mri import ImageFitting_set, Siren, get_mgrid, cases, calculate_contrast, save_dicom
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from skimage.color import rgb2gray, gray2rgb

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import argparse


parser = argparse.ArgumentParser(description='Superresolution of DWI/ADC maps enhanced with AutoERD')
parser.add_argument('--out_folder', default='../experiments/', help='directory to save the quantitative results')
parser.add_argument('--out_img_folder', default='../output_images/', help='directory to save the images')
parser.add_argument('--total_steps', type=int, default=3000, help='total steps for training')
parser.add_argument('--seg', type=int, default=150, help='the epochs to wait until ensamble calculation')
parser.add_argument('--hidden_layers', type=int, default=6, help='depth of the network')
parser.add_argument('--hidden_features', type=int, default=64, help='number of neurons on each layer')
parser.add_argument('--learning_rate', type=float, default=0.0003, help='learning rate')
parser.add_argument('--scale', type=int, default=3, help='scaling factor super-resolution')
parser.add_argument('--exp_name', default='sr2', help='name of the experiment')
parser.add_argument('--repeat_time', type=int, default=1, help='run the experiment multiple times to account for randomness')
parser.add_argument('--erd', action='store_true', help='conduct AutoERD with agglomerative clustering before training')


args = parser.parse_args()
metrics = ['C', 'CNR']
eps = 1e-7
mag = 1000

def minmax_normalize(img, ref):

    return ((img - img.min())/(img.max() - img.min()))*(ref.max() - ref.min()) + ref.min()

def calc_adc(dwi, b0, b):
    adc = -np.log((dwi/(b0 + eps)) + eps)/b 
    return adc*mag*mag

def main():
    
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    cvs_filename = os.path.join(args.out_folder, args.exp_name + '.csv')
    
    with open(cvs_filename, 'w') as f:
        f.write('seed,patient,direction,image,metric,performance\n')
    
    for seed in range(args.repeat_time):
        torch.manual_seed(seed)
        for case in cases:
            print(case.pt_id)            
            directions = ['x', 'y', 'z']
            _slice = case.cancer_slice
            b = case.b
            b0 = case.b0[:, :, _slice]
            img = case.dwi[:, :, _slice, :] #TODO: This will be conducted on all slides later
            inx = np.arange(case.dwi.shape[3]) #acquisition axis
            if args.erd:
                print('Conducting Auto-ERD with Agglomerative Clustering...')
                for i in tqdm(range(case.dwi.shape[0])):
                    for j in range(case.dwi.shape[1]):
                        acq = img[i, j, :].reshape(-1,1)
                        db = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit(acq)
                        sample_means = [acq[db.labels_== x].mean() for x in set(db.labels_)]
                        sample_lens = [len(acq[db.labels_== x]) for x in set(db.labels_)]
                        for k in range(2):
                            if (sample_lens[k] >= (2/3)*case.dwi.shape[3]):# and sample_means[k] > sample_means[1-k] ):
                                case.accept[i, j, _slice, inx[db.labels_== (1-k)]] = 0 
                                            
            for direction in range(3):  # gradient directions x, y, z
                print(f'Training for {directions[direction]} direction...')
                ends = np.cumsum(case.acquisitions)
                starts = ends - case.acquisitions
                img_dataset = []
                accept_weights = []
                sum_image = np.zeros((case.dwi.shape[0], case.dwi.shape[1]))
                sum_accepted = np.zeros((case.dwi.shape[0], case.dwi.shape[1]))
                sum_accepts = np.zeros((case.dwi.shape[0], case.dwi.shape[1]))
                ctr = 0
                for acq in range(starts[direction], ends[direction]):
                    img = case.dwi[:, :, _slice, acq]    
                    accept = case.accept[:, :, _slice, acq]
                    sum_image += img
                    sum_accepted += img*accept
                    sum_accepts += accept
                    ctr += 1	
                accepted_mean = sum_accepted/sum_accepts
                direction_mean = sum_image/ctr

                for acq in range(starts[direction], ends[direction]):
                    img = case.dwi[:, :, _slice, acq]
                    accept = case.accept[:, :, _slice, acq]
                    img_dataset.append(Image.fromarray(img))
                    accept_weights.append(accept)
                dataset = ImageFitting_set(img_dataset)
                _accept_weights = torch.empty((len(img_dataset), dataset.shape[0]**2, 1))
                for ctr, accept in enumerate(accept_weights):
                    transform = Compose([ToTensor()])
                    accept = transform(accept)
                    _accept_weights [ctr] = accept.permute(1, 2, 0).view(-1, 1)
                orig = dataset.mean
                pt_no = case.pt_id.split('-')[-1]

                dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
                img_siren = Siren(in_features=2, out_features=1, 
                                  hidden_features=args.hidden_features,
                                  hidden_layers=args.hidden_layers)
                img_siren.cuda()
                torch.cuda.empty_cache()
                optim = torch.optim.Adam(lr=args.learning_rate, params=img_siren.parameters())
                ctr = 1
                for step in tqdm(range(args.total_steps)):
                    size = dataset.shape
                    for sample in range(len(dataset)):
                        ground_truth, model_input  = dataset.pixels[sample], dataset.coords[sample]
                        ground_truth, model_input = ground_truth.cuda(), model_input.cuda()
                        model_output, coords = img_siren(model_input)
                        weights = _accept_weights[sample].cuda()
                        loss = weights*(model_output - ground_truth)**2
                        loss = loss.mean()
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                    if step == args.total_steps-args.seg:
                        coords2 = get_mgrid(size[0], 2).cuda()
                        superres, _ = img_siren(coords2)
                        coords_large = get_mgrid(size[0]*args.scale, 2).cuda()
                        superres_large, _ = img_siren(coords_large)
                        predicted = superres.cpu().view(size[0], size[1]).detach().numpy()
                        large = superres_large.cpu().view(size[0]*args.scale, size[1]*args.scale).detach().numpy()
                    elif step > args.total_steps-args.seg:
                        superres, _ = img_siren(coords2)
                        superres_large, _ = img_siren(coords_large)
                        predicted += superres.cpu().view(size[0], size[1]).detach().numpy()
                        large += superres_large.cpu().view(size[0]*args.scale, size[1]*args.scale).detach().numpy()

                erd_img = accepted_mean
                out_img = predicted/args.seg
                large_out = large/args.seg
                out_img -= out_img.min()
                large_out -= large_out.min()  
                              
                norm_out_img = minmax_normalize(out_img, direction_mean)
                norm_large_out = minmax_normalize(large_out, direction_mean)               

                b0_scaled = rescale(b0, args.scale, anti_aliasing=False)
                
                adc_erd = calc_adc(erd_img, b0, b)
                adc_orig = calc_adc(orig, b0, b) 
                adc_large =  calc_adc(large_out, b0_scaled, b)                 
                adc_superres = calc_adc(out_img, b0, b)
                adc_norm = calc_adc(norm_out_img, b0, b)
                adc_large_norm = calc_adc(norm_large_out, b0_scaled, b)

                
                images = {'mean':orig, 
                          'erd':erd_img,
                          'ADC_ERD':adc_erd,
                          'superres_n':norm_out_img,
                          'superres':out_img, 
                          'ADC_orig': adc_orig, 
                	      'ADC_super':adc_superres}

                with open(cvs_filename, 'a') as f:
                    for image in images.keys():
                        for inx, metric in enumerate(metrics):
                            f.write('{},{},{},{},{},{}\n'.format(seed, pt_no, directions[direction],image, metric,  
                                                                        calculate_contrast(case, 1, images[image], 0)[inx]))
                
                if direction:
                	out_img += out_img
                	erd_img += erd_img
                	adc_erd += adc_erd
                	large_out += large_out
                	adc_superres += adc_superres
                	adc_large += adc_large
                	adc_orig += adc_orig
                	orig += orig

                
            out_img = out_img/len(directions)
            orig = orig/len(directions)
            erd_img = erd_img/len(directions)
            adc_erd = adc_erd/len(directions)
            large = large_out/len(directions)
            adc_superres = adc_superres/len(directions)
            adc_large = adc_large/len(directions)
            adc_orig = adc_orig/len(directions)
            
            
            filename = os.path.join(args.out_img_folder, args.exp_name, pt_no, 'DWI', 'mean.dcm')
            save_dicom(orig*mag, filename)
            filename = os.path.join(args.out_img_folder, args.exp_name, pt_no, 'DWI', 'erd.dcm')
            save_dicom(erd_img*mag, filename)
            filename = os.path.join(args.out_img_folder, args.exp_name, pt_no, 'DWI', 'super.dcm')
            save_dicom(large*mag, filename)
            filename = os.path.join(args.out_img_folder, args.exp_name, pt_no, 'ADC', 'mean.dcm')
            save_dicom(adc_orig, filename)
            filename = os.path.join(args.out_img_folder, args.exp_name, pt_no, 'ADC', 'erd.dcm')
            save_dicom(adc_erd, filename)
            filename = os.path.join(args.out_img_folder, args.exp_name, pt_no, 'ADC', 'super.dcm')
            save_dicom(adc_superres, filename)
            filename = os.path.join(args.out_img_folder, args.exp_name, pt_no, 'ADC', 'large.dcm')
            save_dicom(adc_large, filename)
                        
                                
images = {'mean':orig,
          'erd':erd_img,
          'ADC_ERD':adc_erd,
          'superres_n':norm_out_img,
          'superres':out_img, 
          'ADC_orig': adc_orig, 
          'ADC_super':adc_superres}

with open(cvs_filename, 'a') as f:
    for image in images.keys():
        for inx, metric in enumerate(metrics):
            f.write('{},{},{},{},{},{}\n'.format(seed, pt_no, 'mean', image, metric,  
                                                                        calculate_contrast(case, 1, images[image], 0)[inx]))
    
if __name__ == "__main__":
    main()


