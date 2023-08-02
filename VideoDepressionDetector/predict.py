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

def predict(video_file_path, model_path='./model/audio_model.pth'):
    # Convert the new video to audio
    video_folder_path = os.path.dirname(video_file_path)
    audio_folder_path = "./data/audio_data_new"
    file = os.path.basename(video_file_path)
    if file.endswith(".mp4"):
        input_file = video_file_path
        output_file = os.path.join(audio_folder_path, file.replace('.mp4', '.wav'))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        command = f'ffmpeg -y -i "{input_file}" -ac 1 -ar 16000 "{output_file}"'
        subprocess.run(command, shell=True, check=True)

    # Extract audio features
    audio_folder_path = "./data/audio_data_new"
    features_folder_path = "./feature/audio_fea_new"
    smile = Smile(feature_set='ComParE_2016')
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

    # Load the model
    device = torch.device('cuda')
    input_size_img1, input_size_img2, input_size_img3, input_size_num = 3, 3, 3, emolarge_df.drop(columns=['name','class']).shape[1]
    num_heads = 8
    img_model = ImageModel(nn.ReLU())
    num_model = NumModel(input_size_num)
    model = FusionModel(img_model, num_model, num_heads=num_heads)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Prepare the data
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    data_list_img1 = [os.path.join(features_folder_path, file[:-6], 'envelope.png')]
    data_list_img2 = [os.path.join(features_folder_path, file[:-6], 'spectrogram.png')]
    data_list_img3 = [os.path.join(features_folder_path, file[:-6], 'mel_spectrogram.png')]
    data_list_num = [emolarge_df[emolarge_df['name'] == file[:-6]].drop(columns=['name','class']).values.tolist()[0]]

    dataset = CustomDataset(data_list_img1=data_list_img1, data_list_img2=data_list_img2, data_list_img3=data_list_img3, data_list_num=data_list_num, transform=data_transform)
    data_loader = DataLoader(dataset, batch_size=1)

    # Predict
    with torch.no_grad():
        for img1, img2, img3, num_features in data_loader:
            img1, img2, img3, num_features = img1.to(device), img2.to(device), img3.to(device), num_features.to(device)
            outputs = model(img1, img2, img3, num_features)
            preds = torch.sigmoid(outputs).item() > 0.5
            print("Prediction: ", "Positive" if preds else "Negative")
    return preds

if __name__ == '__main__':
    video_file_path = 'path_to_your_video.mp4'  # replace with your video file path
    predict(video_file_path)