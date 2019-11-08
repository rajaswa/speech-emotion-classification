#IMPORTS
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

print("Imported required packages ....")

#Lists and dictionaries for mapping
categories = ['happy', 'fear', 'disgust', 'sad', 'neutral']
cols = ['File name', 'prediction', 'file']
label_dict = {'happy':0, 'fear':1, 'disgust':2, 'sad':3, 'neutral':4}
inv_label_dict = {v: k for k, v in label_dict.items()}

#Taking path argument
parser = argparse.ArgumentParser()
parser.add_argument("test_path")
args = parser.parse_args()
path = args.test_path

#Reading files
test_df = pd.DataFrame(columns=cols)

files = os.listdir(path)
file_complete_path = [path + f for f in files]
test_df['File name'] = files
test_df['file'] = file_complete_path

print("Read all the filenames ...")

#Making Pytorch Dataset
class MFCC_Dataset(Dataset):
    """MFCC Spectrogram dataset"""

    def __init__(self, df, transform=None):       
        self.df = df
        self.transform = transform
        self.max_pad_len = 300

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['File name'][idx]
        sample = self.df['file'][idx]
        waveform, sample_rate = torchaudio.load(sample)
        
        
        #MFCC
        mfcc = torchaudio.transforms.MFCC(sample_rate=8000, log_mels=True, n_mfcc=80)(waveform)
        mfcc = mfcc.detach().numpy()
        pad_width = self.max_pad_len - mfcc.shape[2] 
        if pad_width > 0:       
          mfcc = (np.pad(mfcc, pad_width=((0,0), (0,0), (0,pad_width)), mode='constant'))
        else:
          mfcc = (mfcc[:, :, :self.max_pad_len])


        return [mfcc, str(filename)]

test_dataset = MFCC_Dataset(df = test_df)

print("Prepared torch dataset ... ")

# Convolutional neural network 
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))        
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(3192, 5)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)
        out = self.softmax(self.fc1(out))

        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
model.load_state_dict(torch.load('midas_cnn.pt'))
model.eval()

print("Model loaded ...")

filename = []
predictions = []
for i in range(len(test_dataset)):
	sample = test_dataset[i]
	filename.append(sample[1])

	x = (torch.tensor(sample[0])); x = x.view(1, 2, 80, 300); x = x.to(device)
	outputs = model(x)
	_, predicted = torch.max(outputs.data, 1)
	predictions.append(str(inv_label_dict[predicted.item()]))
	print(sample[1], str(inv_label_dict[predicted.item()]))

print("Predictions done ...")

sub_df = pd.DataFrame(columns=['File name', 'prediction'])
sub_df['File name'] = filename
sub_df['prediction'] = predictions

sub_df.to_csv(r'output.txt', index=None, sep=',', mode='a')



