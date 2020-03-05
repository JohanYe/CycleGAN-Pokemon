import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import numpy as np
import torch
from PIL import Image
import os
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torchvision.transforms import ToTensor
from torchvision import transforms
import pandas as pd

# types you can choose : 'Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Poison', 'Electric',
#                        'Ground', 'Fairy', 'Fighting', 'Psychic', 'Rock', 'Ghost', 'Ice',
#                        'Dragon', 'Dark', 'Steel', 'Flying'

class PokemonData(torch.utils.data.Dataset):
    def __init__(self, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), type='Normal',
                 data_path='/content/drive/My Drive/Project_Pokemon/images_with_types'):
        self.type = type
        self.transform = transform
        self.images_path = data_path + '/' + type
        self.images_name = os.listdir(self.images_path)

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):

        image_path = os.path.join(self.images_path, self.images_name[idx])
        image = Image.open(image_path)
        X = self.transform(image)

        return X
