import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_temporal_abaw5_dataset(annotation_path, img_path, feature_dict, max_length, split):
    X, Y = [], []
    filenames = os.listdir(annotation_path)

    for filename in filenames:
        with open(annotation_path + filename) as f:
            lines = f.read().splitlines()[1:]

        x, y, leng = [], [], 0
        for i, label in enumerate(lines):
            expression = int(label)
            if expression >= 0:
                j = i
                overflow = False
                file_exist = False
                while not file_exist:
                    j = j + 1
                    imagename = "{}/{:05d}.jpg".format(filename[:-4], j)
                    imagepath = os.path.join(img_path, imagename)
                    file_exist = os.path.isfile(imagepath)
                    if j > len(lines):
                        overflow = True
                        break

                if overflow == True:
                    j = i+1
                    file_exist = False
                    while not file_exist:
                        j = j - 1
                        imagename = "{}/{:05d}.jpg".format(filename[:-4], j)
                        imagepath = os.path.join(img_path, imagename)
                        file_exist = os.path.isfile(imagepath)

                x.append(np.expand_dims(np.concatenate((feature_dict[imagename][0], feature_dict[imagename][1])), axis=0))
                y.append(expression)
                leng = leng + 1

            if leng < max_length and (i == (len(lines) - 1)) and split == 'train':
                overlap = leng - max_length
                x = prev_x[overlap:] + x
                y = prev_y[overlap:] + y

            if leng == max_length or (i == (len(lines) - 1)):
                if split == 'train':
                    prev_x = x
                    prev_y = y
                X.append(np.concatenate(x, axis=0))
                Y.append(np.array(y))
                x, y, leng = [], [], 0

    return X, Y

def get_temporal_test_abaw5_dataset(annotation_path, img_path, feature_dict, max_length):
    X, Y = [], []
    filenames = os.listdir(annotation_path)

    for filename in filenames:
        with open(annotation_path + filename) as f:
            lines = f.read().splitlines()[1:]

        x, y, leng = [], [], 0
        for i, name in enumerate(lines):
            j = i
            overflow = False
            file_exist = False
            while not file_exist:
                if j == len(lines):
                    overflow = True
                    break
                imagename = "{}/{}.jpg".format(filename[:-4], lines[j])
                imagepath = os.path.join(img_path, imagename)
                file_exist = os.path.isfile(imagepath)
                j = j + 1

            if overflow == True:
                j = i
                file_exist = False
                while not file_exist:
                    j = j - 1
                    imagename = "{}/{}.jpg".format(filename[:-4], lines[j])
                    imagepath = os.path.join(img_path, imagename)
                    file_exist = os.path.isfile(imagepath)

            x.append(np.expand_dims(np.concatenate((feature_dict[imagename][0], feature_dict[imagename][1])), axis=0))
            y.append("{}/{}.jpg".format(filename[:-4], name))
            leng = leng + 1


            if leng == max_length or (i == (len(lines) - 1)):
                X.append(np.concatenate(x, axis=0))
                Y.append(y)
                x, y, leng = [], [], 0

    return X, Y

def get_temporal_lsd_dataset(annotation_path, feature_dict, max_length):
    X, Y = [], []
    x, y, leng = [], [], 0

    with open(annotation_path) as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        imagename, expression = line.split(',')
        expression = int(expression) + 1

        x.append(np.expand_dims(np.concatenate((feature_dict[imagename][0], feature_dict[imagename][1])), axis=0))
        y.append(expression)
        leng = leng + 1

        if leng < max_length and (i == (len(lines) - 1)):
            overlap = leng - max_length
            x = prev_x[overlap:] + x
            y = prev_y[overlap:] + y

        if leng == max_length or (i == (len(lines) - 1)):
            prev_x = x
            prev_y = y
            X.append(np.concatenate(x, axis=0))
            Y.append(np.array(y))
            x, y, leng = [], [], 0
    
    return X, Y

class SequenceFeatureABAW5(Dataset):
    def __init__(self, annoation_path, img_path, feature_dict, max_length, split='train'):
        super(SequenceFeatureABAW5, self).__init__()
        if split == 'test':
            self.X , self.y = get_temporal_test_abaw5_dataset(annoation_path, img_path, feature_dict, max_length)
        else:
            self.X , self.y = get_temporal_abaw5_dataset(annoation_path, img_path, feature_dict, max_length, split)

    def __getitem__(self, i):
        return self.X[i] , self.y[i]

    def __len__(self):
        return len(self.X)

class CombineDataset(Dataset):
    def __init__(self, annoation_paths, img_path, feature_dicts, max_length):
        super(CombineDataset, self).__init__()
        path_abaw5, path_lsd_t, path_lsd_v = annoation_paths
        dict_abaw5, dict_lsd_t, dict_lsd_v = feature_dicts
        X_abaw5, Y_abaw5 = get_temporal_abaw5_dataset(path_abaw5, img_path, dict_abaw5, max_length, True)
        X_lsd_t, Y_lsd_t = get_temporal_lsd_dataset(path_lsd_t, dict_lsd_t, max_length)
        X_lsd_v, Y_lsd_v = get_temporal_lsd_dataset(path_lsd_v, dict_lsd_v, max_length)
        
        self.X = X_abaw5 + X_lsd_t + X_lsd_v
        self.Y = Y_abaw5 + Y_lsd_t + Y_lsd_v
    def __getitem__(self, i):
        return self.X[i] , self.Y[i]

    def __len__(self):
        return len(self.X)

    def ex_weight(self):
        y = np.concatenate(self.Y, axis=0).flatten()
        unique, counts = np.unique(y.astype(int), return_counts=True)
        emo_cw = 1 / counts
        emo_cw/= emo_cw.min()
        return emo_cw
