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

def minmax_normalize(img, ref):

    return ((img - img.min())/(img.max() - img.min()))*(ref.max() - ref.min()) + ref.min()



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
            predicted_XYZ = []
            original_XYZ = []
            large_xyz = []
            pred_ADC_XYZ = []
            large_ADC_xyz = []
            ADC_xyz = []
            directions = ['x', 'y', 'z']
            _slice = case.cancer_slice
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
                # TODO: Calculate and print the mean-max-min of the remaining signals per each direction
                sum_image = np.zeros((128, 128)) #TODO: make the sizes dynamic
                sum_accepted = np.zeros((128, 128))
                sum_accepts = np.zeros((128, 128))
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

                print(directions[direction], direction_mean.min(), direction_mean.max(), direction_mean.mean())
                print(directions[direction], accepted_mean.min(), accepted_mean.max(), accepted_mean.mean())
                print(directions[direction], direction_mean[40:90, 40:90].min(), direction_mean[40:90, 40:90].max(), direction_mean[40:90, 40:90].mean())
                print(directions[direction], accepted_mean[40:90, 40:90].min(), accepted_mean[40:90, 40:90].max(), accepted_mean[40:90, 40:90].mean())
                # TODO: This value is to be calculated over the prostate region
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
                original_XYZ.append(orig)
                dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)
                img_siren = Siren(in_features=2, out_features=1, 
                                  hidden_features=args.hidden_features,
                                  hidden_layers=args.hidden_layers)
                img_siren.cuda()
                torch.cuda.empty_cache()
                optim = torch.optim.Adam(lr=0.0003, params=img_siren.parameters())
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
                        out_img = predicted
                        large_out = large
                    elif step > args.total_steps-args.seg:
                        superres, _ = img_siren(coords2)
                        superres_large, _ = img_siren(coords_large)
                        predicted += superres.cpu().view(size[0], size[1]).detach().numpy()
                        large += superres_large.cpu().view(size[0]*args.scale, size[1]*args.scale).detach().numpy()
                out_img = predicted/100
                large_out = large/100
                
                # TODO: Normalize predicted output to match the range of the original image
                out_img = minmax_normalize(out_img, accepted_mean)
                large_out = minmax_normalize(large_out, accepted_mean)               
                # TODO: Calculate ADC for each direction
                b = case.b
                b0 = case.b0[:, :, _slice]
                b0_scaled = rescale(b0, args.scale, anti_aliasing=False)
                adc_orig = -np.log((orig/(b0 + eps)) + eps)/b 
                adc_orig *= 1000000
                adc_large = -np.log((large/(b0_scaled + eps)) + eps)/b 
                adc_large *= 1000000
                adc_superres = -np.log((predicted/(b0 + eps)) + eps)/b
                adc_superres *= 1000000
                
                
                
                # TODO: Save ADC images for each direction
                img_name = args.experiment_name + '_' + directions[direction]+ '_' + pt_no + '_mean.dcm'
                filename = os.path.join(args.out_img_folder, img_name)
                save_dicom(orig, filename)
                img_name = args.experiment_name + '_' + directions[direction]+ '_' + pt_no + '_super.dcm'
                filename = os.path.join(args.out_img_folder, img_name)
                save_dicom(large, filename)
                img_name = args.experiment_name + '_' + directions[direction]+ '_' + pt_no + '_mean_adc.dcm'
                filename = os.path.join(args.out_img_folder, img_name)
                save_dicom(adc_orig, filename)
                img_name = args.experiment_name + '_' + directions[direction]+ '_' + pt_no + '_super_adc.dcm'
                filename = os.path.join(args.out_img_folder, img_name)
                save_dicom(adc_large, filename)
                
                images = {'mean':orig, 'superres':out_img, 'ADC_orig': adc_orig, 'ADC_new':adc_superres}

                with open(cvs_filename, 'a') as f:
                    for image in images.keys():
                        for inx, metric in enumerate(metrics):
                            f.write('{},{},{},{},{},{}\n'.format(seed, pt_no, directions[direction],image, metric,  
                                                                        calculate_contrast(case, 1, images[image], 0)[inx]))
                
                
                        
                predicted_XYZ.append(out_img)
                large_xyz.append(large_out)
                pred_ADC_XYZ.append(adc_superres)
                large_ADC_xyz.append(adc_large)
                ADC_xyz.append(adc_orig)

                
                
            predicted = sum(predicted_XYZ)/len(predicted_XYZ)
            orig = sum(original_XYZ)/len(original_XYZ)
            large = sum(large_xyz)/len(large_xyz)
            adc_superres = sum(pred_ADC_XYZ)/len(pred_ADC_XYZ)
            adc_large = sum(large_ADC_xyz)/len(large_ADC_xyz)
            adc_orig = sum(ADC_xyz)/len(ADC_xyz)

            
            filename = os.path.join(args.out_img_folder, args.experiment_name + '_' + pt_no + '_mean.dcm')
            save_dicom(orig, filename)
            filename = os.path.join(args.out_img_folder, args.experiment_name + '_' + pt_no + '_super.dcm')
            save_dicom(large, filename)
            filename = os.path.join(args.out_img_folder, args.experiment_name + '_' + pt_no + '_mean_adc.dcm')
            save_dicom(adc_orig, filename)
            filename = os.path.join(args.out_img_folder, args.experiment_name + '_' + pt_no + '_super_adc.dcm')
            save_dicom(adc_large, filename)
                        
                                
            images = {'mean':orig, 'superres':predicted, 'ADC_orig': adc_orig, 'ADC_new':adc_superres}
			# TODO: We may consider putting direction in the file again
            with open(cvs_filename, 'a') as f:
                for image in images.keys():
                    for inx, metric in enumerate(metrics):
                        f.write('{},{},{},{},{},{}\n'.format(seed, pt_no, 'mean', image, metric,  
                                                                        calculate_contrast(case, 1, images[image], 0)[inx]))
    
if __name__ == "__main__":
    main()
