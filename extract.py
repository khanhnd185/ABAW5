
import torch
from torchvision import transforms
import numpy as np
import os
from tqdm.notebook import tqdm
from PIL import Image
import pickle

def get_probab(features, logits=True):
    x=np.dot(features,np.transpose(classifier_weights))+classifier_bias
    if logits:
        return x
    e_x = np.exp(x - np.max(x,axis=0))
    return e_x / e_x.sum(axis=1)[:,None]

print(f"Torch: {torch.__version__}")
use_cuda = torch.cuda.is_available()
print(use_cuda)

DEVICE = 'cuda'
PATH='enet_b0_8_best_vgaf.pt'
#PATH='enet_b2_8.pt'
IMG_SIZE=224
DATA_DIR = '../../../Data/ABAW4/synthetic_challenge/'

test_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]
)

print(PATH)
feature_extractor_model = torch.load('../MTL-ABAW3/model/'+ PATH)

classifier_weights=feature_extractor_model.classifier[0].weight.cpu().data.numpy()
classifier_bias=feature_extractor_model.classifier[0].bias.cpu().data.numpy()
print(classifier_weights.shape,classifier_weights)
print(classifier_bias.shape,classifier_bias)

feature_extractor_model.classifier=torch.nn.Identity()
feature_extractor_model=feature_extractor_model.to(DEVICE)
feature_extractor_model.eval()


print(test_transforms)
data_dir=os.path.join(DATA_DIR,'validation')
print(data_dir)
img_names=[]
X_global_features=[]
imgs=[]
 
for img_name in os.listdir(data_dir):
    if img_name.lower().endswith('.jpg'):
        img = Image.open(os.path.join(data_dir,img_name))
        img_tensor = test_transforms(img)
        if img.size:
            img_names.append(img_name)
            imgs.append(img_tensor)
            if len(imgs)>=96: #48: #96: #32:
                features = feature_extractor_model(torch.stack(imgs, dim=0).to(DEVICE))
                features=features.data.cpu().numpy()
                
                if len(X_global_features)==0:
                    X_global_features=features
                else:
                    X_global_features=np.concatenate((X_global_features,features),axis=0)
                imgs=[]

if len(imgs)>0:        
    features = feature_extractor_model(torch.stack(imgs, dim=0).to(DEVICE))
    features = features.data.cpu().numpy()

    if len(X_global_features)==0:
        X_global_features=features
    else:
        X_global_features=np.concatenate((X_global_features,features),axis=0)

    imgs=[]

X_scores=get_probab(X_global_features)

filename2featuresAll={img_name:(global_features,scores) for img_name,global_features,scores in zip(img_names,X_global_features,X_scores)}
print(len(filename2featuresAll))

with open(os.path.join(DATA_DIR, 'lsd_valid_enet_b0_8_best_vgaf.pickle'), 'wb') as handle:
    pickle.dump(filename2featuresAll, handle, protocol=pickle.HIGHEST_PROTOCOL)