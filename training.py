import loss
import torch.optim as optim
import torch
import helper
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import torchvision
from PIL import Image
import imageio


# helper scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.'''

    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


def training_loop(dataloader_X, dataloader_Y, #test_dataloader_X, test_dataloader_Y,
                  n_epochs=1000,
                  G_XtoY=None, G_YtoX=None, D_X=None, D_Y=None, lr=0.0002, beta1=0.5, beta2=0.999, save_path = '/content/drive/My Drive/Project_Pokemon/'):

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
    d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
    d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])

    print_every = 5

    # keep track of losses over time
    losses = []

    # test_iter_X = iter(test_dataloader_X)
    # test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    # fixed_X = test_iter_X.next()[0]
    # fixed_Y = test_iter_Y.next()[0]
    # fixed_X = scale(fixed_X)  # make sure to scale to a range -1 to 1
    # fixed_Y = scale(fixed_Y)

    # batches per epoch
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))

    for epoch in range(1, n_epochs + 1):

        # Reset iterators for each epoch
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X = iter_X.next()
        real_images = images_X
        images_X = scale(images_X)  # make sure to scale to a range -1 to 1

        images_Y = iter_Y.next()
        real_images = images_X
        images_Y = scale(images_Y)

        # move images to GPU if available (otherwise stay on CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        ##   First: D_X, real and fake loss components   ##

        # Train with real images
        d_x_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        out_x = D_X(images_X)
        D_X_real_loss = loss.real_mse_loss(out_x)

        # Train with fake images

        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 3. Compute the fake loss for D_X
        out_x = D_X(fake_X)
        D_X_fake_loss = loss.fake_mse_loss(out_x)

        # 4. Compute the total loss and perform backprop
        d_x_loss = D_X_real_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()

        ##   Second: D_Y, real and fake loss components   ##

        # Train with real images
        d_y_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        out_y = D_Y(images_Y)
        D_Y_real_loss = loss.real_mse_loss(out_y)

        # Train with fake images

        # 2. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 3. Compute the fake loss for D_Y
        out_y = D_Y(fake_Y)
        D_Y_fake_loss = loss.fake_mse_loss(out_y)

        # 4. Compute the total loss and perform backprop
        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        ##    First: generate fake X images and reconstructed Y images    ##
        g_optimizer.zero_grad()

        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 2. Compute the generator loss based on domain X
        out_x = D_X(fake_X)
        g_YtoX_loss = loss.real_mse_loss(out_x)

        # 3. Create a reconstructed y
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_Y = G_XtoY(fake_X)
        # print(images_Y.shape) #[8, 3, 215, 215]
        # print(reconstructed_Y.shape) #[8, 3, 208, 208]
        reconstructed_y_loss = loss.cycle_consistency_loss(images_Y, reconstructed_Y, lambda_weight=10)

        ##    Second: generate fake Y images and reconstructed X images    ##

        # 1. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 2. Compute the generator loss based on domain Y
        out_y = D_Y(fake_Y)
        g_XtoY_loss = loss.real_mse_loss(out_y)

        # 3. Create a reconstructed x
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_X = G_YtoX(fake_Y)
        reconstructed_x_loss = loss.cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=10)

        # 5. Add up all generator and reconstructed losses and perform backprop
        g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_y_loss + reconstructed_x_loss
        g_total_loss.backward()
        g_optimizer.step()

        # Print the log info
        if epoch % 50== 0:
            filename = save_path + str(epoch)
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))

            # generate image
            X_fake = G_XtoY(images_X)
            grid_image_real = torchvision.utils.make_grid(images_X.cpu())
            grid_image_fake = torchvision.utils.make_grid(X_fake.cpu())
            grid_image = torch.cat((grid_image_real, grid_image_fake), 1)
            saveim = np.transpose(grid_image.data.numpy(), (1, 2, 0))  
            # plt.figure(figsize=(20, 10))
            # plt.imshow(saveim)
            # plt.savefig(filename + '_' + 'XtoY.jpg')
            path = filename + '_' + 'XtoY.jpg'
            imageio.imwrite(path, saveim)
            print('Saved {}'.format(path))

            Y_fake = G_YtoX(images_Y)
            grid_image_real = torchvision.utils.make_grid(images_Y.cpu())
            grid_image_fake = torchvision.utils.make_grid(Y_fake.cpu())
            grid_image = torch.cat((grid_image_real, grid_image_fake), 1)
            saveim = np.transpose(grid_image.data.numpy(), (1, 2, 0))  
            # plt.figure(figsize=(20, 10))
            # plt.imshow(saveim)
            # plt.savefig(filename + '_' + 'YtoX.jpg')
            path = filename + '_' + 'YtoX.jpg'
            imageio.imwrite(path, saveim)
            print('Saved {}'.format(path))

        # sample_every = 1
        # # Save the generated samples
        # if epoch % sample_every == 0:
        #     G_YtoX.eval()  # set generators to eval mode for sample generation
        #     G_XtoY.eval()
        #     helper.save_samples(epoch, images_Y, images_X, G_YtoX, G_XtoY, batch_size=8)
        #     G_YtoX.train()
        #     G_XtoY.train()

        # uncomment these lines, if you want to save your model
    #         checkpoint_every=1000
    #         # Save the model parameters
    #         if epoch % checkpoint_every == 0:
    #             checkpoint(epoch, G_XtoY, G_YtoX, D_X, D_Y)



    return losses, G_XtoY, G_YtoX, D_X, D_Y
