import glob
import numpy as np
import obspy as obs
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
import os
import sys
import pandas as pd
import plotly.express as px
from scipy.signal import stft


def do_stft(x, fs = 125.0, nperseg = 96, wf_length=1251):
    x = np.pad(x, (0, wf_length - x.size), 'constant')
    f, t, Zxx = stft(x, fs, nperseg=nperseg)
    Zxx = np.abs(Zxx)
    return Zxx/np.max(np.max(Zxx))


if len(os.listdir('/loggerhead/lwrigh89/Numpy Array')) == 0:
    iterable = 0
    waveform_counter = 0
    station_counter = 1
    # waveforms = np.random.random((281658, 3, 49, 28))
    # get all the fracturing locations
    for dirs in list(next(os.walk('/loggerhead/coke/wf_Tony/trim/15_62.5_10s'))[1]):
        print('\n\t***** ' + str(dirs) + ' STATION BEGINS *****\n')

        waveforms = np.zeros((4082, 3, 49, 28))
        for wave in range(1, 4083):
            try:
                temp_wave1 = obs.read('/loggerhead/coke/wf_Tony/trim/15_62.5_10s/' + str(dirs) + '/DH1/event' + str(wave) + '.mseed')
                A1 = temp_wave1[0].data
                frequency1 = do_stft(A1)
                waveforms[iterable, 0, :, :] = frequency1
                temp_wave2 = obs.read('/loggerhead/coke/wf_Tony/trim/15_62.5_10s/' + str(dirs) + '/DH2/event' + str(wave) + '.mseed')
                A2 = temp_wave2[0].data
                frequency2 = do_stft(A2)
                waveforms[iterable, 1, :, :] = frequency2
                temp_wave3 = obs.read('/loggerhead/coke/wf_Tony/trim/15_62.5_10s/' + str(dirs) + '/DHZ/event' + str(wave) + '.mseed')
                A3 = temp_wave3[0].data
                frequency3 = do_stft(A3)
                waveforms[iterable, 2, :, :] = frequency3
            except:
                print('Found a problem with a waveform! Deleting now...')
                waveforms = np.delete(waveforms, iterable, axis=0)
                iterable -= 1
            if waveform_counter % 1000 == 0:
                print('\n\tnumber of waves read in: ' + str(waveform_counter) + '\tstation counter: ' + str(
                    station_counter) + '/69\n')
            iterable += 1
            waveform_counter += 1
        iterable = 0


        station_counter += 1
        np.save('/loggerhead/lwrigh89/Numpy Array/wavearr' + str(dirs) + '.npy', waveforms)
        print('\n\t***** STATION ' + str(dirs) + ' SAVED *****\n')
    sys.exit()
else:
    waves = glob.glob('/loggerhead/lwrigh89/Numpy Array' + '/*.npy')
    # waves.sort()
    waveforms = np.empty((0, 3, 49, 28), float)
    iterator = 1
    for nump in waves:
        station = np.load(nump)
        no_nan_station = np.nan_to_num(station)
        # print(station)
        if iterator <= 44:
            waveforms = np.append(waveforms, no_nan_station, axis=0)
            print('number of stations loaded: ' + str(iterator))

        iterator += 1
        
    print(waveforms.shape)



train_arr, test_arr = sklearn.model_selection.train_test_split(waveforms, train_size=0.9)


train_torch = torch.tensor(train_arr, requires_grad=True).clone().float()
test_torch = torch.tensor(test_arr, requires_grad=True).clone().float()

train_waves = train_torch
test_waves = test_torch
print(train_waves.shape)
print(test_waves.shape)


k = 7
p = k//2


use_decoder = True


class AutoEncoder(nn.Module):
    def __init__(self):
        #  make sure to always initialize the super class when using outside methods
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=k, padding=p), nn.LeakyReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=k, padding=p), nn.LeakyReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=k, padding=p), nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2), nn.LeakyReLU(),
            nn.Conv2d(8, 3, kernel_size=(2, 3), padding=1), nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        if use_decoder:
            x = self.decoder(x)
        return x


#   if directory is empty, create new model, else use model from directory
new_model = True
if len(os.listdir('/loggerhead/lwrigh89/Model')) == 0:
    model = AutoEncoder()
else:
    model = AutoEncoder()
    model.load_state_dict(torch.load('/loggerhead/lwrigh89/Model/newmodel.pt'))
    model.eval()
    new_model = False



loss_function_MSE = nn.MSELoss()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

model.to(device)


# Training function
def train_epoch(model, device, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = []
    train_tester = train_waves.clone().detach()
    # shuffle the training dataset
    train_tester = train_tester[torch.randperm(train_tester.size()[0])]
    iterable = 0
    for wave in train_tester:
        wave.unsqueeze_(0)
        wave = wave.to(device)
        output_thing = model(wave)
        loss = loss_fn((output_thing), (wave))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iterable % 1000 == 0:
            print('number of waves analyzed: ' + str(iterable))
        #   Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())
        iterable += 1

    return np.mean(train_loss)


# Testing function
def test_epoch(model, device, loss_fn):
    # Set evaluation mode for model
    model.eval()

    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for wave in test_waves:
            wave.unsqueeze_(0)
            # Move tensor to the proper device
            wave = wave.to(device)
            # model data
            output_thing = model(wave)
            # Append the network output and the original image to the lists
            conc_out.append(output_thing.cpu())
            conc_label.append(wave.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn((conc_out), (conc_label))
        
    return val_loss.data


sample_counter = 0


def plot_wave(arr, index, cluster_num):
    global sample_counter
    arr_nump = arr.cpu().detach().numpy()
    wave_torch_best = torch.from_numpy(arr_nump[index, :, :, :]).float().unsqueeze_(0)
    og = wave_torch_best.detach().cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('DH1, DH2, DH3')
    ax1.pcolormesh(og[0, 0, :])
    ax2.pcolormesh(og[0, 1, :])
    ax3.pcolormesh(og[0, 2, :])
    fig.savefig('/loggerhead/lwrigh89/Plots/Cluster Samples/sample' + str(cluster_num) + str(sample_counter) + '.png')
    sample_counter += 1


def plot_outputs(model):
    model.eval()
    rand_num = random.randint(0, 4082)
    wave_torch_best = torch.from_numpy(waveforms[rand_num, :, :, :]).float().unsqueeze_(0)
    reconstructed = wave_torch_best.to(device)
    reconstructed = model(reconstructed)
    new_numpy = reconstructed.detach().cpu().numpy()
    og = wave_torch_best.detach().cpu().numpy()
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('DH1, DH2, DH3')
    ax1.pcolormesh(og[0, 0, :])
    ax2.pcolormesh(og[0, 1, :])
    ax3.pcolormesh(og[0, 2, :])
    fig.savefig('/loggerhead/lwrigh89/Plots/Comparing Plots/original.png')
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.suptitle('DH1, DH2, DH3')
    ax1.pcolormesh(new_numpy[0, 0, :])
    ax2.pcolormesh(new_numpy[0, 1, :])
    ax3.pcolormesh(new_numpy[0, 2, :])
    fig.savefig('/loggerhead/lwrigh89/Plots/Comparing Plots/reconstructed.png')


num_epochs = 4
diz_loss = {'train_loss':[],'val_loss':[]}
if new_model:
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, device, loss_function_MSE, optimizer)
        val_loss = test_epoch(model, device, loss_function_MSE)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        # counter counts the number of epochs
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), '/loggerhead/lwrigh89/Model/newmodel.pt')
            print('\n\t***** MODEL SAVED ******\n')

            # plot og vs reconstructed
            plot_outputs(model)

            plt.figure(figsize=(10, 8))
            plt.semilogy(diz_loss['train_loss'], label='Train')
            plt.semilogy(diz_loss['val_loss'], label='Valid')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            # plt.grid()
            plt.legend()
            # plt.title('loss')
            # plt.show()
            plt.savefig('/loggerhead/lwrigh89/Plots/Epochs/epochgraph.png')
else:
    plot_outputs(model)

if True:

    encoded_waves = []
    use_decoder = False
    colours = ['blue', 'red', 'green']
    for sample in test_waves:
        sample = sample.to(device)
        model.eval()
        with torch.no_grad():
    #         model data
            output_thing = model(sample)
            encoded_wave = output_thing.flatten().cpu().numpy()
            encoded_waves.append(encoded_wave)


    data_waves = pd.DataFrame(list(encoded_waves))
    kmeans = KMeans(n_clusters=3)
    idx = kmeans.fit_predict(data_waves)
    tsne_results = TSNE().fit_transform(data_waves)
    labels_wave = kmeans.labels_


    # add kmeans clustering
    fig = px.scatter(tsne_results, x=0, y=1, labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'}, color=kmeans.labels_)
    fig.write_image('/loggerhead/lwrigh89/Plots/TSNE Scatterplot/tsnegraph.png')

    use_decoder = True

    for num in range(3):
        idx_cluster = np.where(kmeans.labels_ == num)
        print(idx_cluster)
        for rand in range(1, 5):
            index = random.randint(1, np.ma.size(idx_cluster))
            print(str(index))
            plot_wave(test_waves, index, num)
        sample_counter = 0


