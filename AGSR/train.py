import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import psutil

criterion = nn.MSELoss()
def get_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / float(2 ** 20)  # Memory usage in MB
    return mem

def train(model, subjects_adj, subjects_labels, args):
    bce_loss = nn.BCELoss()
    netD = Discriminator(args)
    print(netD)
    optimizerG = optim.AdamW(model.parameters(), lr=args.lr)
    optimizerD = optim.AdamW(netD.parameters(), lr=args.lr)

    # Initialize StepLR
    schedulerG = StepLR(optimizerG, step_size=args.step_size, gamma=args.gamma)
    schedulerD = StepLR(optimizerD, step_size=args.step_size, gamma=args.gamma)

    all_epochs_loss = []
    best_epochs_error = float('inf')
    max_patience = 25
    patience = max_patience
    epsilon = 0.0 # 0.00005
    label_smoothing_real = 0.9
    for epoch in range(args.epochs):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []
            total_memory_usage = 0

            for lr, hr in zip(subjects_adj, subjects_labels):
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                hr = pad_HR_adj(hr, args.padding)
                lr = torch.from_numpy(lr).type(torch.FloatTensor)
                padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)

                eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

                model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                    lr, args.lr_dim, args.hr_dim)

                mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                    model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)

                error = criterion(model_outputs, padded_hr)
                real_data = model_outputs.detach()
                fake_data = gaussian_noise_layer(padded_hr, args)

                d_real = netD(real_data)
                d_fake = netD(fake_data)

                
                dc_loss_real = bce_loss(d_real, torch.full((args.hr_dim, 1), label_smoothing_real))
                dc_loss_fake = bce_loss(d_fake, torch.zeros(args.hr_dim, 1))
                dc_loss = dc_loss_real + dc_loss_fake

                dc_loss.backward()
                optimizerD.step()

                d_fake = netD(gaussian_noise_layer(padded_hr, args))

                gen_loss = bce_loss(d_fake, torch.full((args.hr_dim, 1), label_smoothing_real))
                generator_loss = gen_loss + mse_loss
                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(generator_loss.item())
                epoch_error.append(error.item())

            # Before updating the learning rate, get the last learning rate for comparison
            last_lrG = schedulerG.get_last_lr()[0]
            last_lrD = schedulerD.get_last_lr()[0]

            # Update learning rate
            schedulerG.step()
            schedulerD.step()

           # After updating, check if the learning rate has decayed, and print a message if so
            current_lrG = schedulerG.get_last_lr()[0]
            current_lrD = schedulerD.get_last_lr()[0]
            if last_lrG != current_lrG:
                print(f"Generator Learning Rate Decayed from {last_lrG} to {current_lrG}")
            if last_lrD != current_lrD:
                print(f"Discriminator Learning Rate Decayed from {last_lrD} to {current_lrD}")


            avg_epochs_error = np.mean(epoch_error)
            print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                  "Error: ", avg_epochs_error*100, "%")
            memory_usage = get_memory_usage()
            total_memory_usage += memory_usage

            print(f"Epoch {epoch+1}: Memory Usage: {total_memory_usage} MB")

            all_epochs_loss.append(np.mean(epoch_loss))

        if avg_epochs_error + epsilon < best_epochs_error:
            best_epochs_error = avg_epochs_error
            patience = max_patience
        else:
            patience -= 1

        if patience == 0:
            print("Early stopping. No significant improvement in the error. Epoch:", epoch)
            break


def test(model_instance, test_data, model_parameters):
    predictions = []

    for data_point in test_data:
        is_all_zeros = not np.any(data_point)
        if not is_all_zeros:
            data_point_tensor = torch.from_numpy(data_point).type(torch.FloatTensor)
            prediction_output, _, _, _ = model_instance(data_point_tensor, model_parameters.lr_dim, model_parameters.hr_dim)
            prediction_output = unpad(prediction_output, model_parameters.padding).detach().numpy()
            predictions.append(prediction_output)

    return np.stack(predictions)

