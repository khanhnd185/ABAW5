import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def get_dataset(filename, feature_dict, nameonly=False):
    with open(filename) as f:
        mtl_lines = f.read().splitlines()
    num_missed=0
    X,y_expr=[],[]
    for line in mtl_lines[1:]:
        splitted_line=line.split(',')
        imagename=splitted_line[0]
        expression=int(splitted_line[1])

        if nameonly == True:
            X.append(imagename)
            y_expr.append(expression)
            continue

        if imagename in feature_dict:
            X.append(np.concatenate((feature_dict[imagename][0],feature_dict[imagename][1])))
            y_expr.append(expression)
        else:
            num_missed+=1
    X=np.array(X)
    y_expr=np.array(y_expr)

    return X,y_expr

class LSD(Dataset):
    def __init__(self, filename, feature_dict):
        super(LSD, self).__init__()
        self.X , self.y = get_dataset(filename, feature_dict)

    def __getitem__(self, i):
        return self.X[i] , self.y[i]
    def __len__(self):
        return len(self.X)

    def ex_weight(self):
        unique, counts = np.unique(self.y.astype(int), return_counts=True)
        emo_cw = 1 / counts
        emo_cw/= emo_cw.min()
        return emo_cw


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class RawLSD(Dataset):
    def __init__(self, filename, img_path):
        super(RawLSD, self).__init__()
        self.X , self.y = get_dataset(filename, {}, nameonly=True)
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