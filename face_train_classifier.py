import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime
import sys
import pandas as pd
import os
from models import FaceAutoencoder,Autoencoder,FaceClassifier
from datasets import EncodedDeepfakeDataset, FaceDeepfakeDataset


base_path = 'deepfake-detection-challenge'
        
train_folder = os.listdir(str(sys.argv[1]))
train_folders = []
for path in train_folder:
    full_path = os.path.join(base_path, 'train_folders', path)
    print(f"train folder: {full_path}")
    train_folders.append(str(full_path))

test_folder = os.listdir(str(sys.argv[2]))
test_folders = []
for path in test_folder:
    full_path = os.path.join(base_path, 'test_folders', path)
    print(f"test folder: {full_path}")
    test_folders.append(str(full_path))

batch_size = int(sys.argv[3])
num_epochs = int(sys.argv[4])
n_frames = int(sys.argv[5])
lr = float(sys.argv[6])

TRAIN_FOLDERS = train_folders
TEST_FOLDERS = test_folders
print(f"all train folders: {train_folders}, {type(train_folders)}")
print(f"all test folders: {test_folders}, {type(test_folders)}")
# AUTOENCODER = 'autoencoder_H10M46S22_04-11-21.pt'

# batch_size = 10
# num_epochs = 1
# epoch_size = 500
# n_frames = 30
milestones = [6,12,18]
gamma = 0.1
n_vid_features = 36*36 # 3600
n_aud_features = 1
n_head = 8
n_layers = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder = FaceAutoencoder()
if len(sys.argv) > 7:
    print("pretrained autoencoder is loaded")
    AUTOENCODER = str(sys.argv[7])
    autoencoder.load_state_dict(torch.load(AUTOENCODER, map_location=device))
autoencoder.to(device)
autoencoder.eval()

model = FaceClassifier(n_vid_features, n_aud_features, n_head, n_layers)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

start_time = datetime.datetime.now()
print(f'start time: {str(start_time)}')
print(f'using device: {device}')

'''Splitting into Train and Validation'''
train_dataset = FaceDeepfakeDataset(TRAIN_FOLDERS, autoencoder.encoder, n_frames=n_frames, n_audio_reads=576, device=device, cache_folder="face_encode_cache")
test_dataset = FaceDeepfakeDataset(TEST_FOLDERS, autoencoder.encoder, n_frames=n_frames, n_audio_reads=576, device=device)
# dataset_size = len(dataset)
# val_split = .3
# val_size = int(val_split * dataset_size)
# train_size = dataset_size - val_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

'''Train_Loop'''
train_losses = []
val_losses = []
best_loss = np.inf
train_accuracies = []
val_accuracies = []
epoch_times = []


for epoch in range(num_epochs):
    epoch_start_time = time.time()
    epoch_t_loss = 0
    epoch_v_loss = 0
    t_count = 0
    t_count_wrong = 0

    model.train()
    for i, batch in enumerate(train_loader):
        # if i * batch_size >= epoch_size:
        #     break
        video_data, audio_data, labels = batch
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)
        
        output = model(video_data, audio_data)
        loss = criterion(output, labels)

        output = torch.sigmoid(output)
        output = output.round()
        n_wrong = (labels - output).abs().sum()
        t_count_wrong += n_wrong
        t_count += labels.shape[0]

        epoch_t_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('.', end='', flush=True)

    model.eval()
    with torch.no_grad():
        v_count = 0
        v_count_wrong = 0
        for i, batch in enumerate(val_loader):
            # if i * batch_size >= epoch_size:
        #        break
            video_data, audio_data, labels = batch
            video_data = video_data.to(device)
            audio_data = audio_data.to(device)
            # optimizer.zero_grad()
            output = model(video_data, audio_data)
            loss = criterion(output, labels)

            output = torch.sigmoid(output)
            output = output.round()
            n_wrong = (labels - output).abs().sum()
            v_count_wrong += n_wrong
            v_count += labels.shape[0]

            epoch_v_loss += loss.item()

            # loss.backward()
            # optimizer.step()
            print('.', end='', flush=True)

    epoch_end_time = time.time()
    epoch_exec_time = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_exec_time)
    train_losses.append(epoch_t_loss/len(train_loader))
    val_losses.append(epoch_t_loss/len(val_loader))

    t_count_right = t_count - t_count_wrong
    v_count_right = v_count - v_count_wrong
    t_accuracy = t_count_right / t_count
    v_accuracy = v_count_right / v_count

    train_accuracies.append(t_accuracy)
    val_accuracies.append(v_accuracy)

    print(f'\nepoch: {epoch}, train loss: {train_losses[-1]}, val loss: {val_losses[-1]}, executed in: {str(epoch_exec_time)}')
    print(f"train total: {t_count}, train correct: {t_count_right}, train incorrect: {t_count_wrong}, train accuracy: {t_accuracy}")
    print(f"valid total: {v_count}, valid correct: {v_count_right}, valid incorrect: {v_count_wrong}, valid accuracy: {v_accuracy}")

    scheduler.step()
    ### Saving model per best validation loss 
    if best_loss > val_losses[-1]: 
        best_loss = val_losses[-1]
        end_time = datetime.datetime.now()
        torch.save(model.state_dict(), f'classifier_{n_frames}_bigger.pt')

end_time = datetime.datetime.now()
print(f"end time: {str(end_time)}")
exec_time = end_time - start_time
print(f"executed in: {str(exec_time)}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

df = pd.DataFrame()
df['train_loss'] = train_losses
df['val_loss'] = val_losses
df['train_acc'] = train_accuracies
df['val_acc'] = val_accuracies
df['epoch_times'] = epoch_times
    
df.to_csv(f'train_classifier_nframes{n_frames}_bs{batch_size}_lr{lr}.csv', index=False)
