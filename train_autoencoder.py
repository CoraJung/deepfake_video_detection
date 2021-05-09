import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime
import sys
import pandas as pd

from models import Autoencoder, FaceAutoencoder
from datasets import EncodedDeepfakeDataset, FaceDeepfakeDataset

print(str(sys.argv[1]))
print(int(sys.argv[2]))
train_folder = [str(sys.argv[1])]
num_epochs = int(sys.argv[2])


# train_folders = ['../deepfake-detection-challenge/train_sample_videos/']
train_folders = train_folder
epoch_size = 250


batch_size = 1
'''Splitting into Train and Validation'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

dataset = FaceDeepfakeDataset(train_folders, encoder=None, n_frames=1, device=device)
# dataset = EncodedDeepfakeDataset(train_folders, encoder=None, n_frames=1, device=device)
dataset_size = len(dataset)
print(f'dataset_size:{dataset_size}')

val_split = .3
val_size = int(val_split * dataset_size)
print(f'val size: {val_size}')
train_size = dataset_size - val_size
print(f'train size: {train_size}')
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def train_autoencoder(train_folders, num_epochs, train_loader, val_loader, device):
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    start_time = datetime.datetime.now()
    print(f"train_encoder start time: {str(start_time)}")
    print('Using device:', device)

    # model = Autoencoder()
    model = FaceAutoencoder()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    '''Train_Loop'''
    train_losses = []
    val_losses = []
    epoch_times = []
    best_loss = np.inf
    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()
        epoch_t_loss = 0
        epoch_t_acc = 0

        model.train()
        for i, batch in enumerate(train_loader):
            # if i * batch_size >= epoch_size:
            #     break
            # print(f'size of batch: {batch.shape}')
            data, _, _ = batch
            #print(f'size of data: {data.shape}') # [1,3,1920, 1080]

            data = data.to(device)
            data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
            # data = data.squeeze(0)
            optimizer.zero_grad()
            output = model(data)
            #print(f'size of output: {output.shape}') # [1,3,160,160]
            loss = criterion(output, data)
            # epoch_t_loss += loss.item()*len(data)
            epoch_t_loss += loss.item()
            loss.backward()
            optimizer.step()
            print('.', end='', flush=True)

        model.eval()
        epoch_v_loss = 0
        epoch_v_acc = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # if i * batch_size >= epoch_size:
                #     break
                data, _, _ = batch
                data = data.to(device)
                data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
                # data = data.squeeze(0)
                # i think we don't need optimizer
                # optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data)
                
                # epoch_v_loss += loss.item()*len(data)
                epoch_v_loss += loss.item()
                
                # we don't need these i think.
                # loss.backward()
                # optimizer.step()
                print('.', end='', flush=True)

        train_losses.append(epoch_t_loss/len(train_loader))
        val_losses.append(epoch_t_loss/len(val_loader))

        epoch_end_time = datetime.datetime.now()
        epoch_exec_time = epoch_end_time - epoch_start_time

        epoch_times.append(epoch_exec_time)

        print(f'\nepoch: {epoch}, train loss: {train_losses[-1]}, val loss: {val_losses[-1]}, executed in: {str(epoch_exec_time)}')

        ### Saving model per best validation loss 
        if best_loss > val_losses[-1]: 
            end_time = datetime.datetime.now()
            best_loss = val_losses[-1]
            torch.save(model.state_dict(), 'best_faceautoencoder.pt')

    exec_time = end_time - start_time
    print(f"train_encoder executed in: {str(exec_time)}, end time: {str(end_time)}")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    df = make_df(train_losses, val_losses, exec_time, epoch_times)

    return df

def make_df(train_losses, val_losses, wall_time, epoch_times):
    df = pd.DataFrame()

    df['train_loss'] = train_losses
    df['val_loss'] = val_losses
    df['epoch_times'] = epoch_times
    df['wall_time'] = wall_time
    
    return df

# if __name__ == "__main__":
time_now = datetime.datetime.now()
df = train_autoencoder(train_folders, num_epochs, train_loader, val_loader, device)
df.to_csv(f'autoencoder_{time_now}.csv', index=False)

