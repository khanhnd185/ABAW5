import os
import pickle
import argparse
import pandas as pd
import torch.optim as optim
from sam import SAM
from tqdm import tqdm
from lstm import SequenceFeatureABAW5, LSTM, CombineDataset
from lion_pytorch import Lion
from dataset import FeatureABAW5
from transformer import Transformer
from torch.utils.data import DataLoader
from helpers import *

def train(net, trainldr, optimizer, epoch, epochs, criteria, lr):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    for batch_idx, (inputs, y) in enumerate(tqdm(trainldr)):
        B, N, C = y.shape
        adjust_learning_rate(optimizer, epoch, epochs, lr, batch_idx, train_loader_len)
        y = y.float()
        inputs = inputs.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        yhat = net(inputs)
        y = y.view(-1, C)
        yhat = yhat.view(-1, C)
        mask = torch.ones(B*N)
        mask = mask.float()
        mask = mask.cuda()
        loss = criteria(yhat, y, mask)
        loss.backward()
        optimizer.step()
        total_losses.update(loss.data.item(), inputs.size(0))
    return total_losses.avg()


def val(net, validldr):
    net.eval()
    all_y = None
    all_yhat = None
    for batch_idx, (inputs, y) in enumerate(tqdm(validldr)):
        with torch.no_grad():
            y = y.long()
            inputs = inputs.cuda()
            y = y.cuda()
            yhat = net(inputs)
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
    metrics, _ = au_metric(all_y, all_yhat)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Emotion')

    parser.add_argument('--arc', default='transformer', help='Net name')
    parser.add_argument('--input', default='', help='Input file')
    parser.add_argument('--dataset', default='AUFeatureABAW5', help='Dataset name')
    parser.add_argument('--datadir', default='../../../Data/ABAW5/', help='Dataset folder')
    parser.add_argument('--sam', default=False, action='store_true', help='Apply SAM')
    parser.add_argument('--lion', default=False, action='store_true', help='Lion optimizer')
    parser.add_argument('--config', default='0', help="config")
    parser.add_argument('--epochs', default=20, type=int, help="number of epoch")
    parser.add_argument('--batch', default=64, type=int, help="batch size")
    parser.add_argument('--length', default=64, type=int, help="max sequence length")
    parser.add_argument('--head', default=4, type=int, help="Num of head")
    parser.add_argument('--layer', default=4, type=int, help="Num of layer")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    output_dir = '{}-{}-{}'.format(args.arc, args.dataset, args.config)

    if args.dataset == 'AUFeatureABAW5':
        train_annotation_path = args.datadir + 'annotations/AU/Train_Set/'
        valid_annotation_path = args.datadir + 'annotations/AU/Validation_Set/'
        with open(os.path.join(args.datadir, 'cropped_aligned/batch1/abaw5.pickle'), 'rb') as handle:
            abaw5_feature = pickle.load(handle)

        image_path = args.datadir + 'cropped_aligned/batch1/cropped_aligned/'
        trainset = SequenceFeatureABAW5(train_annotation_path, image_path, abaw5_feature, args.length, 'train')
        validset = SequenceFeatureABAW5(valid_annotation_path, image_path, abaw5_feature, args.length, 'val')
        
        trainexw = torch.from_numpy(trainset.au_weight())
        trainexw = trainexw.float()
        trainexw = trainexw.cuda()
    else: 
        train_annotation_path = args.datadir + 'annotations/EX/Train_Set/'
        valid_annotation_path = args.datadir + 'annotations/EX/Validation_Set/'
        with open(os.path.join(args.datadir, 'cropped_aligned/batch1/abaw5.pickle'), 'rb') as handle:
            feature = pickle.load(handle)
        image_path = args.datadir + 'cropped_aligned/batch1/cropped_aligned/'
        
        trainset = FeatureABAW5(train_annotation_path, image_path, feature)
        validset = FeatureABAW5(valid_annotation_path, image_path, feature)
        trainexw = torch.from_numpy(trainset.ex_weight())
        trainexw = trainexw.float()
        trainexw = trainexw.cuda()

        trainset = SequenceFeatureABAW5(train_annotation_path, image_path, feature, args.length, 'train')
        validset = SequenceFeatureABAW5(valid_annotation_path, image_path, feature, args.length, 'val')

    trainldr = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=0)
    validldr = DataLoader(validset, batch_size=1, shuffle=False, num_workers=0)

    start_epoch = 0
    if args.arc == 'lstm':
        net = LSTM()
    else:
        net = Transformer(1288, 12, 512, args.head, 512, 0.1, args.layer)

    if args.input != '':
        print("Resume form | {} ]".format(args.input))
        net = load_state_dict(net, args.input)

    net = nn.DataParallel(net).cuda()
    criteria = WeightedAsymmetricLoss(weight=trainexw)

    if args.sam:
        optimizer = SAM(net.parameters(), torch.optim.SGD, lr=args.lr, momentum=0.9, weight_decay=1.0/args.batch)
    elif args.lion:
        optimizer = Lion(net.parameters(), lr=args.lr, weight_decay=10.0/args.batch)
    else:
        optimizer = optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=1.0/args.batch)
    best_performance = 0.0

    df = {}
    df['epoch'] = []
    df['lr'] = []
    df['train_loss'] = []
    df['val_metrics'] = []

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]['lr']
        train_loss = train(net, trainldr, optimizer, epoch, args.epochs, criteria, args.lr)
        val_metrics = val(net, validldr)

        infostr = {'Epoch {}: {:.5f},{:.5f},{:.5f}'
                .format(epoch,
                        lr,
                        train_loss,
                        val_metrics)}
        print(infostr)

        os.makedirs(os.path.join('results', output_dir), exist_ok = True)

        if val_metrics >= best_performance:
            checkpoint = {'state_dict': net.state_dict()}
            torch.save(checkpoint, os.path.join('results', output_dir, 'best_val_perform.pth'))
            best_performance = val_metrics

        checkpoint = {'state_dict': net.state_dict()}
        torch.save(checkpoint, os.path.join('results', output_dir, 'cur_model.pth'))

        df['epoch'].append(epoch),
        df['lr'].append(lr),
        df['train_loss'].append(train_loss),
        df['val_metrics'].append(val_metrics)

    df = pd.DataFrame(df)
    csv_name = os.path.join('results', output_dir, 'train.csv')
    df.to_csv(csv_name)

if __name__=="__main__":
    main()
