####################################################################################################
#Super Resolution for MR Images
#Author: Batuhan Gundogdu
####################################################################################################
import torch
from torch import nn
import numpy as np
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from PIL import Image#, ImageSequence
import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class SineLayer(nn.Module):
      def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
          super().__init__()
          self.omega_0 = omega_0
          self.is_first = is_first
          self.in_features = in_features
          self.linear = nn.Linear(in_features, out_features, bias=bias)
          self.init_weights()

     def init_weights(self):
         with torch.no_grad():
              if self.is_first:
                  self.linear.weight.uniform_(-1 / self.in_features,1 / self.in_features)
             else:
                  self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                               np.sqrt(6 / self.in_features) / self.omega_0)
     def forward(self, input):
          return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate



class Siren(nn.Module):
      def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                   first_omega_0=30, hidden_omega_0=30.):
          super().__init__()
	
      self.net = []
      self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

      for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

      if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
	
            with torch.no_grad():
                  final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
      else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

      self.net = nn.Sequential(*self.net)

      def forward(self, coords):
 
            coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
            output = self.net(coords)
            return output, coords

def get_image_tensor(img):
	
      sidelength, _ = img.size
      transform = Compose([Resize(sidelength),ToTensor(), Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
      img = transform(img)
      return img

class ImageFitting_set(Dataset):
	
      "This is rearranged for MR dataset"
		  
      def __init__(self, img_dataset):
            super().__init__()
            sidelength, sidelength = img_dataset[0].size
            self.orig = np.empty((len(img_dataset),img_dataset[0].size[0],img_dataset[0].size[1]))
            self.pixels = torch.empty((len(img_dataset),sidelength**2, 1))
            self.coords = torch.empty((len(img_dataset),sidelength**2, 2))
            for ctr, img in enumerate(img_dataset):
                  self.orig [ctr] = np.array(img)
                  img = get_image_tensor(img)
                  self.pixels[ctr] = img.permute(1, 2, 0).view(-1, 1)
                  self.coords[ctr] = get_mgrid(sidelength, 2)
            self.mean = sum(self.orig)/len(self.orig)

      def __len__(self):
            return len(self.pixels)

      def __getitem__(self, idx):
            return self.coords, self.pixels

def save_fig(array,filename,size=(2,2),dpi=600, cmap='gray'):
      fig = plt.figure()
      fig.set_size_inches(size)
      ax = plt.Axes(fig, [0., 0., 1., 1.])
      ax.set_axis_off()
      fig.add_axes(ax)
      plt.set_cmap(cmap)
      ax.imshow(array, aspect='equal')
      plt.savefig(filename, format='eps', dpi=dpi)

def main():

      parser = argparse.ArgumentParser()
      parser.add_argument("folder", help="the folder that the images are stored")
      parser.add_argument("steps", type=int, help="number of epochs to train")
      args = parser.parse_args()
	
      if not os.path.isdir(args.folder):
            print(f"No folder named {args.folder}")
            exit(1)

      img_dataset = []
      for f in os.listdir(args.folder):
            filename = os.path.join(args.folder,f)
            img = Image.open(filename)
            img_dataset.append(img)
      dataset = ImageFitting_set(img_dataset)
      orig = dataset.mean
      dataloader = DataLoader(dataset, batch_size=2, pin_memory=True, num_workers=0)
      img_siren = Siren(in_features=2, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True)
      img_siren.cuda()
      net = torch.nn.DataParallel(img_siren, device_ids=[0, 1, 2])
      torch.cuda.empty_cache()
      optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
      exp = args.folder +'_epochs'
      filename = os.path.join(exp, 'mean.eps')
      save_fig(orig,filename,size=(2,2),dpi=600)
      if not os.path.isdir(exp):
            os.mkdir(exp)
      for step in tqdm.tqdm(range(args.steps)):
            for sample in range(len(dataset)):
                  ground_truth, model_input = dataset.pixels[sample], dataset.coords[sample]
                  ground_truth, model_input = ground_truth.cuda(), model_input.cuda()
                  model_output, coords = net(model_input)
                  loss = ((model_output - ground_truth)**2).mean()
                  predicted = model_output.cpu().view(420,420).detach().numpy()
                  grad = img_grad.norm(dim=-1).cpu().view(420,420).detach().numpy()
                  optim.zero_grad()
                  loss.backward()
                  optim.step()
                  if not step % 25 and sample==11:
                        img_filename = os.path.join(exp, 'img_' + str(step) + '.eps')
                        save_fig(predicted,img_filename,size=(2,2),dpi=600)
                        PATH = os.path.join(exp,'model' + str(step) + '.pt')
                        torch.save(net.state_dict(), PATH)
      print('Done')


if __name__ == "__main__":
	main()

