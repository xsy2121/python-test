import os, sys
import subprocess
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from opensmile import Smile
sys.path.append('./utils/')
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from CustomDataset import CustomDataset
from MyAttention import MyAttention
from ImageModel import ImageModel
from NumModel import NumModel
from FusionModel import FusionModel

# 需要安装 ffmpeg
def VideoToWav():
    video_folder_path = "./data/video_data"
    audio_folder_path = "./data/audio_data"
    for root, dirs, files in os.walk(video_folder_path):
        for file in files:
            if file.endswith(".mp4"):
                input_file = os.path.join(root, file)
                output_file = os.path.join(root.replace(video_folder_path, audio_folder_path), file.replace('.mp4', '.wav'))
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                command = f'ffmpeg -y -i "{input_file}" -ac 1 -ar 16000 "{output_file}"'
                subprocess.run(command, shell=True, check=True)

# 需要有 librosa , opensmile 包
def extract_upper_envelope(y, sr):
    y_hilbert = hilbert(y)
    envelope = np.abs(y_hilbert)
    plt.figure(figsize=(14, 5))
    plt.plot(y, label='Signal')
    plt.plot(envelope, label='Envelope')
    plt.title('Upper Envelope')
    plt.legend()
    plt.savefig('envelope.png')
    plt.close()

def extract_spectrogram(y, sr):
    plt.figure(figsize=(14, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig('spectrogram.png')
    plt.close()

def extract_melspectrogram(y, sr):
    plt.figure(figsize=(14, 5))
    S = librosa.feature.melspectrogram(y, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.savefig('mel_spectrogram.png')
    plt.close()

def AudioFeature():
    audio_folder_path = "./data/audio_data"
    features_folder_path = "./feature/audio_fea"
    smile = Smile(feature_set='ComParE_2016')  # emolarge 在 ComParE_2016 feature set 中
    emolarge_df = pd.DataFrame()

    for root, dirs, files in os.walk(audio_folder_path):
        for file in files:
            if file.endswith('.wav'):
                input_file = os.path.join(root, file)
                output_folder = os.path.join(features_folder_path, file.replace('.wav', ''))
                os.makedirs(output_folder, exist_ok=True)
                
                y, sr = librosa.load(input_file)
                os.chdir(output_folder)

                extract_upper_envelope(y, sr)
                extract_spectrogram(y, sr)
                extract_melspectrogram(y, sr)

                features = smile.process_file(input_file)
                features['name'] = file.replace('.wav', '')
                emolarge_df = emolarge_df.append(features)

    emolarge_df.to_csv('./features/audio_fea/emolarge.csv', index=False)

def train(model, train_loader, criterion, optimizer_class, learning_rate, device, num_epochs):
    model.to(device)

    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training
        model.train()
        for (img1, img2, img3, num_features, labels) in train_loader:
            img1, img2, img3, num_features, labels = img1.to(device), img2.to(device), img3.to(device), num_features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2, img3, num_features)
            loss = criterion(outputs.view(-1), labels.float())
            loss.backward()
            optimizer.step()
    return model

def dumpmodel():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载数据
    folder_labels = {
        "NC": 0,
        "Mild": 1,
        "Moderate": 1,
        "Severe": 1,
    }
    base_path = './data/audio_data/CNRAC'
    feature_path = './features/audio_fea'
    df = pd.read_csv('./features/audio_fea/emolarge.csv')
    data_list_img1, data_list_img2, data_list_img3, data_list_num = [], [], [], []
    label_list = []
    for folder, label in folder_labels.items():
        for root, dirs, files in os.walk(os.path.join(base_path, folder)):
            for file in files:
                data_list_img1.append(os.path.join(feature_path, file[:-6], 'envelope.png'))
                data_list_img2.append(os.path.join(feature_path, file[:-6], 'spectrogram.png'))
                data_list_img3.append(os.path.join(feature_path, file[:-6], 'mel_spectrogram.png'))
                data_list_num.append(df[df['name'] == file[:-6]].drop(columns=['name','class']).values.tolist()[0])
                label_list.append(label)

    dataset = CustomDataset(data_list_img1=data_list_img1,data_list_img2=data_list_img2,data_list_img3=data_list_img3,data_list_num=data_list_num, label_list=label_list, transform=data_transform)

    # Set up training configurations
    num_epochs = 50
    device = torch.device('cuda')

    param_grid = {
        'criterion': [nn.BCEWithLogitsLoss()],
        'activation': [nn.ReLU()],
        'optimizer': [optim.SGD],
        'learning_rate': [0.001],
    }

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size_img1, input_size_img2, input_size_img3, input_size_num = dataset.feature_size()
    num_heads = 8
    img_model = ImageModel(param_grid['activation'][0])
    num_model = NumModel(input_size_num)
    model = FusionModel(img_model, num_model, num_heads=num_heads)

    criterion = param_grid['criterion'][0]
    optimizer_class = param_grid['optimizer'][0]

    model = train(model, train_loader, criterion, optimizer_class, param_grid['learning_rate'][0], device, num_epochs)

    # 保存训练的模型
    torch.save(model.state_dict(), './model/'+'audio_model.pth')

if __name__ == '__main__':
    VideoToWav()
    AudioFeature()
    dumpmodel()