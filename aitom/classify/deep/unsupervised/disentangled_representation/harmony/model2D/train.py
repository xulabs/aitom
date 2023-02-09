import torch
from .utils import loss_fn, plot_loss
from .model import save_ckp, load_ckp


def train_model(dataset_name, siamese, optimizer, train_loader, test_loader, device,
                start_epoch, n_epochs, epoch_train_loss, epoch_valid_loss, valid_loss_min,
                z_dim=2, pixel=64, batch_size=100, w=1, scale=False):
    for epoch in range(start_epoch, n_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0
        # print("Training")
        siamese.train()
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device=device)
            data = images.reshape(batch_size, 1, pixel, pixel)
            image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = siamese(data, scale)
            loss = loss_fn(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, z_dim, w, scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        epoch_train_loss.append(train_loss / len(train_loader))
        # validate the model #
        ######################
        siamese.eval()
        for batch_idx, images in enumerate(test_loader):
            with torch.no_grad():
                images = images.to(device=device)
                data = images.reshape(batch_size, 1, pixel, pixel)
                image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2 = siamese(data, scale)
                loss = loss_fn(image_z1, image_z2, image_x_theta1, image_x_theta2, phi1, phi2, z_dim, w, scale)
                valid_loss += loss.item()

        epoch_valid_loss.append(valid_loss / len(test_loader))

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\t'.format(
            epoch,
            epoch_train_loss[epoch],
            epoch_valid_loss[epoch],
        ))

        if epoch % 10 == 0 and epoch_valid_loss[epoch] < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                            epoch_valid_loss[epoch]))
            # save checkpoint as best model

            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': epoch_valid_loss[epoch],
                'epoch_train_loss': epoch_train_loss,
                'epoch_valid_loss': epoch_valid_loss,
                'state_dict': siamese.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_ckp(checkpoint, 'best_model_Harmony_fc' + dataset_name + '_z_dim_{}_w_{}.pt'.format(z_dim, w))
            valid_loss_min = epoch_valid_loss[epoch]
    #plot_loss(epoch_train_loss=epoch_train_loss, epoch_valid_loss=epoch_valid_loss)
