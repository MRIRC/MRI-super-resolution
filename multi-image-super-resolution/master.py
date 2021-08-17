from nn_mri import cases, save_dicom # calculate_contrast, 
from utils.network import RAMS
from utils.prediction import predict_tensor,
import tensorflow as tf
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='Superresolution of DWI/ADC maps with Multi-image SR')
parser.add_argument('--out_folder', default='../experiments/', help='directory to save the quantitative results')
parser.add_argument('--out_img_folder', default='../output_images/', help='directory to save the images')
parser.add_argument('--exp_name', default='sr2', help='name of the experiment')

args = parser.parse_args()

SCALE = 3
FILTERS = 32
KERNEL_SIZE = 3
CHANNELS = 9
R = 8
N = 12
checkpoint_dir = f'ckpt/RED_RAMS'

def main():
    rams_network = RAMS(scale=SCALE, filters=FILTERS, 
                 kernel_size=KERNEL_SIZE, channels=CHANNELS, r=R, N=N)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                psnr=tf.Variable(1.0),
                                model=rams_network)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    for case in cases:
        _low_res_seq = case.dwi
        num_acq = _low_res_seq.shape[3]
        low_res_seq = _low_res_seq[:,:,case.cancer_slice,:]
        lor = np.expand_dims(low_res_seq, 0).astype('uint16')
        lor = (lor)*256
        mean_pred = np.zeros((128*3, 128*3))
        sample_size = 50
        for k in range(sample_size):
            inx = random.sample(list(range(num_acq)),9)
            #inx_str = [str(x) for x in sorted(inx)]
            #names = img_fname.split('_')
            #fname = names[0] + '-' + '-'.join(inx_str) + '.dcm'
            img = predict_tensor(rams_network, lor[:,:,:,inx])[0,:,:,0]
            mean_pred += img
        mean_pred /= sample_size
    pt_no = case.pt_id.split('-')[-1]
    filename = os.path.join(args.out_img_folder, args.exp_name, pt_no, 'DWI', 'mean.dcm')
    save_dicom(mean_pred, img_fname)

    
    
    
if __name__ == "__main__":
    main()