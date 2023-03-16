import os
import pickle
import argparse
from tqdm import tqdm
from lstm import SequenceFeatureABAW5
from transformer import Transformer
from torch.utils.data import DataLoader
from helpers import *

def test(net1, net2, net3, validldr):
    net1.eval()
    net2.eval()
    net3.eval()
    all_y = []
    all_yhat = None
    for batch_idx, (inputs, y) in enumerate(tqdm(validldr)):
        with torch.no_grad():
            inputs = inputs.cuda()
            yhat1 = net1(inputs)
            yhat2 = net2(inputs)
            yhat3 = net3(inputs)
            yhat = (yhat1 + yhat2 + yhat3) / 3
            yhat = yhat.squeeze(0)
            y = [i[0] for i in y]

            if all_yhat == None:
                all_y = y.copy()
                all_yhat = yhat.clone()
            else:
                all_y = all_y + y
                all_yhat = torch.cat((all_yhat, yhat), 0)
    all_yhat = all_yhat.cpu().numpy()
    return all_y, all_yhat

def generate_output(filename, imagenames, predictions):
    with open(filename, 'w') as f:
        f.write("image_location,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n")

        for i, name in enumerate(imagenames):
            infostr = '{},{}\n'.format(name, np.argmax(predictions[i,:]))
            f.write(infostr)


def main():
    parser = argparse.ArgumentParser(description='Train Emotion')

    parser.add_argument('--net1', default='transformer-CombineDataset-83dd7186', help='Net name')
    parser.add_argument('--net2', default='transformer-CombineDataset-83dd7186-h8', help='Net name')
    parser.add_argument('--net3', default='transformer-CombineDataset-83dd7186-N6', help='Net name')
    parser.add_argument('--output', default='predictions', help='Output name')
    parser.add_argument('--datadir', default='../../../Data/ABAW5/', help='Dataset folder')
    parser.add_argument('--length', default=64, type=int, help="max sequence length")
    args = parser.parse_args()

    valid_annotation_path = args.datadir + 'testset/framelist/'

    with open(os.path.join(args.datadir, 'cropped_aligned/batch1/abaw5.pickle'), 'rb') as handle:
        abaw5_feature = pickle.load(handle)

    image_path = args.datadir + 'cropped_aligned/batch1/cropped_aligned/'
    validset = SequenceFeatureABAW5(valid_annotation_path, image_path, abaw5_feature, args.length, 'test')
    validldr = DataLoader(validset, batch_size=1, shuffle=False, num_workers=0)

    net1 = Transformer(1288, 8, 512, 4, 512, 0.1, 4)
    net2 = Transformer(1288, 8, 512, 8, 512, 0.1, 4)
    net3 = Transformer(1288, 8, 512, 4, 512, 0.1, 6)

    net1 = load_state_dict(net1, 'results/' + args.net1 + '/best_val_perform.pth')
    net2 = load_state_dict(net2, 'results/' + args.net2 + '/best_val_perform.pth')
    net3 = load_state_dict(net3, 'results/' + args.net3 + '/best_val_perform.pth')

    net1 = nn.DataParallel(net1).cuda()
    net2 = nn.DataParallel(net2).cuda()
    net3 = nn.DataParallel(net3).cuda()

    imagenames, predictions = test(net1, net2, net3, validldr)
    generate_output(args.output, imagenames, predictions)


if __name__=="__main__":
    main()
