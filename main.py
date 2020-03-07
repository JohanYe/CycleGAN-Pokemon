import importlib
import dataloader
import network
import loss
import training
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch
import os
import torchvision
import imageio


importlib.reload(dataloader)
importlib.reload(network)
importlib.reload(loss)
importlib.reload(training)



type1 = 'Water'
type2 = 'Grass'
save_path = './pokemon_results_' + type1 + '_' + type2 + '/'


# Get data
Xdset = dataloader.PokemonData(type=type1, data_path='./images_with_types')
dataloader_X = DataLoader(Xdset, batch_size=8, shuffle=True)
Ydset = dataloader.PokemonData(type=type2, data_path='./images_with_types')
dataloader_Y = DataLoader(Ydset, batch_size=8, shuffle=True)


# Get the model and train
G_XtoY, G_YtoX, D_X, D_Y = network.create_model()
losses, G_XtoY, G_YtoX, D_X, D_Y = training.training_loop(dataloader_X=dataloader_X, dataloader_Y=dataloader_Y,
                                                          n_epochs=4000, G_XtoY=G_XtoY, G_YtoX=G_YtoX, D_X=D_X, D_Y=D_Y,
                                                          lr=0.0005, beta1=0.5, beta2=0.999,
                                                          save_path=save_path)

# save the model
checkpoint_dir = './Checkpoints/'
G_XtoY_path = os.path.join(checkpoint_dir, 'G_XtoY.pkl')
G_YtoX_path = os.path.join(checkpoint_dir, 'G_YtoX.pkl')
D_X_path = os.path.join(checkpoint_dir, 'D_X.pkl')
D_Y_path = os.path.join(checkpoint_dir, 'D_Y.pkl')
torch.save(G_XtoY.state_dict(), G_XtoY_path)
torch.save(G_YtoX.state_dict(), G_YtoX_path)
torch.save(D_X.state_dict(), D_X_path)
torch.save(D_Y.state_dict(), D_Y_path)


# # only when you want to use saved model
# checkpoint_dir = '/content/drive/My Drive/Project_Pokemon/pokemon_results_WATER_GRASS/'
# G_XtoY_path = os.path.join(checkpoint_dir, 'G_XtoY.pkl')
# G_YtoX_path = os.path.join(checkpoint_dir, 'G_YtoX.pkl')


# # get saved model
# G_XtoY_path = os.path.join(checkpoint_dir, 'G_XtoY.pkl')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# G_XtoY.load_state_dict(torch.load(G_XtoY_path))
# G_XtoY.to(device)
# G_YtoX_path = os.path.join(checkpoint_dir, 'G_YtoX.pkl')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# G_YtoX.load_state_dict(torch.load(G_YtoX_path))
# G_YtoX.to(device)


Xdset = dataloader.PokemonData(type=type1,data_path='./images_with_types')
dataloader_X = DataLoader(Xdset, batch_size=8, shuffle=True)
Ydset = dataloader.PokemonData(type=type1,data_path='./images_with_types')
dataloader_Y = DataLoader(Ydset, batch_size=8, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from PIL import Image

for i in range(10):
    images_X = iter(dataloader_X)
    images_X = images_X.next()
    # images_X = training.scale(images_X)
    images_X = images_X.to(device)
    X_fake = G_XtoY(images_X)
    grid_image_real = torchvision.utils.make_grid(images_X.cpu())
    grid_image_fake = torchvision.utils.make_grid(X_fake.cpu())
    grid_image = torch.cat((grid_image_real, grid_image_fake), 1)
    saveim = np.transpose(grid_image.data.numpy(), (1, 2, 0))
    path = save_path + 'final_' + str(i)+ '_XtoY.jpg'
    imageio.imwrite(path, saveim)
    print('Saved {}'.format(path))


for i in range(10):
    images_Y = iter(dataloader_Y)
    images_Y = images_Y.next()
    # images_Y = training.scale(images_Y)
    images_Y = images_Y.to(device)
    Y_fake = G_YtoX(images_Y)
    grid_image_real = torchvision.utils.make_grid(images_Y.cpu())
    grid_image_fake = torchvision.utils.make_grid(Y_fake.cpu())
    grid_image = torch.cat((grid_image_real, grid_image_fake), 1)
    saveim = np.transpose(grid_image.data.numpy(), (1, 2, 0))
    path = save_path + 'final_' + str(i)+ '_YtoX.jpg'
    imageio.imwrite(path, saveim)
    print('Saved {}'.format(path))


fig, ax = plt.subplots(figsize=(12,8))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
plt.plot(losses.T[2], label='Generators', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.savefig('./Figure/Figure_1.png', bbox_inches='tight')