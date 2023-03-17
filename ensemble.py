import os
import pickle
import argparse
from tqdm import tqdm
from torch import nn as nn
from dataset import SequenceFeatureABAW5
from transformer import Transformer
from torch.utils.data import DataLoader
from utils import *

def val(net1, net2, net3, validldr):
    net1.eval()
    net2.eval()
    net3.eval()
    all_y = None
    all_yhat = None
    for batch_idx, (inputs, y) in enumerate(tqdm(validldr)):
        with torch.no_grad():
            y = y.long()
            inputs = inputs.cuda()
            y = y.cuda()
            yhat1 = net1(inputs)
            yhat2 = net2(inputs)
            yhat3 = net3(inputs)
            yhat = (yhat1 + yhat2 + yhat3) / 3
            y = y.squeeze(0)
            yhat = yhat.squeeze(0)

            if all_y == None:
                all_y = y.clone()
                all_yhat = yhat.clone()
            else:
                all_y = torch.cat((all_y, y), 0)
                all_yhat = torch.cat((all_yhat, yhat), 0)
    all_y = all_y.cpu().numpy()
    all_yhat = all_yhat.cpu().numpy()
    print(type(all_y), type(all_yhat))
    print((all_y.shape), (all_yhat.shape))
    metrics = ex_metric(all_y, all_yhat)
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train Emotion')

    parser.add_argument('--net1', default='83dd7186.pth', help='Net name')
    parser.add_argument('--net2', default='83dd7186-h8.pth', help='Net name')
    parser.add_argument('--net3', default='83dd7186-N6.pth', help='Net name')
    parser.add_argument('--datadir', default='archive/dataset/affwild2/', help='Dataset folder')
    parser.add_argument('--length', default=64, type=int, help="max sequence length")
    args = parser.parse_args()

    valid_annotation_path = args.datadir + 'annotations/EX/Validation_Set/'

    with open(os.path.join(args.datadir, 'abaw5.pickle'), 'rb') as handle:
        abaw5_feature = pickle.load(handle)

    image_path = args.datadir + 'cropped_aligned/batch1/cropped_aligned/'
    validset = SequenceFeatureABAW5(valid_annotation_path, image_path, abaw5_feature, args.length, 'val')
    validldr = DataLoader(validset, batch_size=1, shuffle=False, num_workers=0)

    net1 = Transformer(1288, 8, 512, 4, 512, 0.1, 4)
    net2 = Transformer(1288, 8, 512, 8, 512, 0.1, 4)
    net3 = Transformer(1288, 8, 512, 4, 512, 0.1, 6)

    net1 = load_state_dict(net1, 'archive/models/' + args.net1)
    net2 = load_state_dict(net2, 'archive/models/' + args.net2)
    net3 = load_state_dict(net3, 'archive/models/' + args.net3)

    net1 = nn.DataParallel(net1).cuda()
    net2 = nn.DataParallel(net2).cuda()
    net3 = nn.DataParallel(net3).cuda()

    val_metrics = val(net1, net2, net3, validldr)
    print('F1-score {:.5f}'.format(val_metrics))


if __name__=="__main__":
    main()
