import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def get_lsd_dataset(filename, feature_dict=None):
    with open(filename) as f:
        mtl_lines = f.read().splitlines()

    X, Y = [], []
    for line in mtl_lines[1:]:
        splitted_line = line.split(',')
        imagename = splitted_line[0]
        expression = int(splitted_line[1])

        if feature_dict == None:
            X.append(imagename)
            Y.append(expression)
            continue
        elif imagename in feature_dict:
            X.append(np.concatenate((feature_dict[imagename][0], feature_dict[imagename][1])))
            Y.append(expression)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def get_abaw5_dataset(annotation_path, img_path, feature_dict=None):
    X, Y = [], []
    filenames = os.listdir(annotation_path)

    for filename in filenames:
        with open(annotation_path + filename) as f:
            mtl_lines = f.read().splitlines()

        for i, label in enumerate(mtl_lines[1:]):
            expression = int(label)
            if expression < 0:
                continue

            imagename = "{}/{:05d}.jpg".format(filename[:-4], i+1)
            imagepath = os.path.join(img_path, imagename)
            if not os.path.isfile(imagepath):
                continue

            if feature_dict == None:
                X.append(imagename)
                Y.append(expression)
            elif imagename in feature_dict:
                X.append(np.concatenate((feature_dict[imagename][0], feature_dict[imagename][1])))
                Y.append(expression)

    Y = np.array(Y)
    return X, Y

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class LSD(Dataset):
    def __init__(self, filename, img_path):
        super(LSD, self).__init__()
        self.X , self.y = get_lsd_dataset(filename)
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
        img = pil_loader(os.path.join(self.img_path, self.X[i]))
        img = self.transform(img)
        return img, self.y[i]

    def __len__(self):
        return len(self.X)

    def ex_weight(self):
        unique, counts = np.unique(self.y.astype(int), return_counts=True)
        emo_cw = 1 / counts
        emo_cw/= emo_cw.min()
        return emo_cw

class FeatureLSD(Dataset):
    def __init__(self, filename, feature_dict):
        super(FeatureLSD, self).__init__()
        self.X , self.y = get_lsd_dataset(filename, feature_dict)

    def __getitem__(self, i):
        return self.X[i] , self.y[i]
    def __len__(self):
        return len(self.X)

    def ex_weight(self):
        unique, counts = np.unique(self.y.astype(int), return_counts=True)
        emo_cw = 1 / counts
        emo_cw/= emo_cw.min()
        return emo_cw


class ABAW5(Dataset):
    def __init__(self, annoation_path, img_path):
        super(ABAW5, self).__init__()
        self.X , self.y = get_abaw5_dataset(annoation_path, img_path)
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
        img = pil_loader(os.path.join(self.img_path, self.X[i]))
        img = self.transform(img)
        return img, self.y[i]

    def __len__(self):
        return len(self.X)

    def ex_weight(self):
        unique, counts = np.unique(self.y.astype(int), return_counts=True)
        emo_cw = 1 / counts
        emo_cw/= emo_cw.min()
        return emo_cw

class FeatureABAW5(Dataset):
    def __init__(self, annoation_path, img_path, feature_dict):
        super(FeatureABAW5, self).__init__()
        self.X , self.y = get_abaw5_dataset(annoation_path, img_path, feature_dict)
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
        img = pil_loader(os.path.join(self.img_path, self.X[i]))
        img = self.transform(img)
        return img, self.y[i]

    def __len__(self):
        return len(self.X)

    def ex_weight(self):
        unique, counts = np.unique(self.y.astype(int), return_counts=True)
        emo_cw = 1 / counts
        emo_cw/= emo_cw.min()
        return emo_cw

