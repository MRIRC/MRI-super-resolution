from nn_mri import cases, calculate_contrast
import numpy as np
import torch
from PIL import Image
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

parser = argparse.ArgumentParser(description='DAVID')
parser.add_argument('--out_folder', default='../experiments/', help='directory to save the quantitative results')
parser.add_argument('--experiment_name', default='david', help='name of the experiment')

args = parser.parse_args()
metrics = ['C', 'CNR']
eps = 1e-7


def main():
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    cvs_filename = os.path.join(args.out_folder, args.experiment_name + '.csv')

    with open(cvs_filename, 'w') as f:
        f.write('patient,image,direction,acquisition,metric,performance\n')
    for case in cases:
        pt_no = case.pt_id.split('-')[-1]
        print(case.pt_id)
        original_XYZ = []
        directions = ['x', 'y', 'z']
        _slice = case.cancer_slice
        img = case.dwi[:, :, _slice, :] #TODO: This will be conducted on all slides later	
        inx = np.arange(case.dwi.shape[3]) #acquisition axis
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
        b0 = case.b0[:, :, _slice]
        b = case.b					
        for direction in range(3):  # gradient directions x, y, z
            print(f'calculating for {directions[direction]} direction...')
            ends = np.cumsum(case.acquisitions)
            starts = ends - case.acquisitions
            sum_image = np.zeros((128, 128))
            sum_accepted = np.zeros((128, 128))
            sum_accepts = np.zeros((128, 128))
            ctr = 0
            for acq in range(starts[direction], ends[direction]):
                img = case.dwi[:, :, _slice, acq]
                adc_img = -np.log((img/(b0 + eps)) + eps)/b 
                adc_img *= 1000
                with open(cvs_filename, 'a') as f:
                    for inx, metric in enumerate(metrics):
                        f.write('{},{},{},{},{},{}\n'.format(pt_no,'DWI', directions[direction], acq, metric, calculate_contrast(case, 1, img, 0)[inx]))
                        f.write('{},{},{},{},{},{}\n'.format(pt_no,'ADC', directions[direction], acq, metric, calculate_contrast(case, 1, adc_img, 0)[inx]))
				
                accept = case.accept[:, :, _slice, acq]
                sum_image += img
                sum_accepted += img*accept
                sum_accepts += accept
                ctr += 1	
            accepted_mean = sum_accepted/sum_accepts
            direction_mean = sum_image/ctr
            accepted_mean_adc = -np.log((accepted_mean/(b0 + eps)) + eps)/b 
            accepted_mean_adc *= 1000
            direction_mean_adc = -np.log((direction_mean/(b0 + eps)) + eps)/b
            direction_mean_adc *= 1000
            with open(cvs_filename, 'a') as f:	
                for inx, metric in enumerate(metrics):
                    f.write('{},{},{},{},{},{}\n'.format(pt_no,'DWI', directions[direction], 'mean', metric, calculate_contrast(case, 1, direction_mean, 0)[inx]))
                    f.write('{},{},{},{},{},{}\n'.format(pt_no,'ADC', directions[direction], 'mean', metric, calculate_contrast(case, 1, direction_mean_adc, 0)[inx]))
                    f.write('{},{},{},{},{},{}\n'.format(pt_no,'DWI_ERD', directions[direction], 'mean', metric, calculate_contrast(case, 1, accepted_mean, 0)[inx]))
                    f.write('{},{},{},{},{},{}\n'.format(pt_no,'ADC_ERD', directions[direction], 'mean', metric, calculate_contrast(case, 1, accepted_mean_adc, 0)[inx]))		


if __name__ == "__main__":
    main()
