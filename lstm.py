import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from block import Dense
from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    data, labels = zip(*batch)
    padded_data = pad_sequence(data, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True)
    num_seqs_per_sample = torch.Tensor([len(x) for x in data])

    return padded_data, padded_labels, num_seqs_per_sample

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_temporal_abaw5_dataset(annotation_path, img_path, feature_dict, max_length, train):
    X, Y = [], []
    filenames = os.listdir(annotation_path)

    for filename in filenames:
        with open(annotation_path + filename) as f:
            lines = f.read().splitlines()[1:]

        x, y, leng = [], [], 0
        for i, label in enumerate(lines):
            expression = int(label)
            if expression < 0:
                continue

            imagename = "{}/{:05d}.jpg".format(filename[:-4], i+1)
            imagepath = os.path.join(img_path, imagename)
            if not os.path.isfile(imagepath):
                continue

            if imagename in feature_dict:
                x.append(np.expand_dims(np.concatenate((feature_dict[imagename][0], feature_dict[imagename][1])), axis=0))
                y.append(expression)
                leng = leng + 1

            if leng < max_length and (i == (len(lines) - 1)) and train:
                overlap = leng - max_length
                x = prev_x[overlap:] + x
                y = prev_y[overlap:] + y

            if leng == max_length or (i == (len(lines) - 1)):
                if train:
                    prev_x = x
                    prev_y = y
                X.append(np.concatenate(x, axis=0))
                Y.append(np.array(y))
                x, y, leng = [], [], 0

    return X, Y

class SequenceFeatureABAW5(Dataset):
    def __init__(self, annoation_path, img_path, feature_dict, max_length, train):
        super(SequenceFeatureABAW5, self).__init__()
        self.X , self.y = get_temporal_abaw5_dataset(annoation_path, img_path, feature_dict, max_length, train)
        self.img_path = img_path
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ]
        )
    def __getitem__(self, i):
        return self.X[i] , self.y[i]

    def __len__(self):
        return len(self.X)

class LSTM(nn.Module):
    def __init__(self, num_class=8, feature_size=1288):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(feature_size, feature_size, num_layers=2, batch_first=True)
        self.head = nn.Sequential(
            Dense(feature_size, 512, activation='relu', drop=0.2),
            Dense(512, 64, activation='relu', drop=0.2),
            Dense(64, num_class),
        )

    def forward(self, x):
        self.rnn.flatten_parameters()
        output, _ = self.rnn(x)
        output = self.head(output)

        return output
