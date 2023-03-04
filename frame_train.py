import os
import pickle
import argparse
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from frameformer import greedy_decode
from frameformer import Frameformer, FrameABAW5, VFrameABAW5
from torch.utils.data import DataLoader
from helpers import *

def train(net, trainldr, optimizer, epoch, epochs, criteria, lr):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    for i, batch in enumerate(tqdm(trainldr)):
        src, src_mask, tgt, tgt_mask, tgt_y = batch
        adjust_learning_rate(optimizer, epoch, epochs, lr, i, train_loader_len)
        src = src.cuda()
        src_mask = src_mask.unsqueeze(1)
        src_mask = src_mask > 0
        src_mask = src_mask.cuda()
        tgt = tgt.cuda()
        tgt_mask = tgt_mask > 0
        tgt_mask = tgt_mask.cuda()
        optimizer.zero_grad()

        yhat = net(src, tgt, src_mask, tgt_mask)
        y = torch.argmax(tgt_y, dim=-1).cuda()
        loss = criteria(yhat.permute(0, 2, 1), y)
        loss.backward()
        optimizer.step()
        total_losses.update(loss.data.item(), src.size(0))
    return total_losses.avg()


def val(net, validldr):
    net.eval()
    all_y = None
    all_yhat = None
    for batch_idx, batch in enumerate(tqdm(validldr)):
        with torch.no_grad():
            src, src_mask, tgt, tgt_mask, y = batch
            _, max_len, _ = src.shape
            src = src.cuda()
            src_mask = src_mask.unsqueeze(1)
            src_mask = src_mask > 0
            src_mask = src_mask.cuda()
            tgt = tgt.cuda()
            tgt_mask = tgt_mask > 0
            tgt_mask = tgt_mask.cuda()

            yhat = greedy_decode(net, src, src_mask, max_len=max_len)
            y = y.squeeze(0)
            yhat = yhat[:,1:,:].squeeze(0)

            if all_y == None:
                all_y = y.clone()
                all_yhat = yhat.clone()
            else:
                all_y = torch.cat((all_y, y), 0)
                all_yhat = torch.cat((all_yhat, yhat), 0)
    all_y = all_y.cpu().numpy()
    all_yhat = all_yhat.cpu().numpy()
    metrics = EX_metric(all_y, all_yhat)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Emotion')

    parser.add_argument('--input', default='', help='Input file')
    parser.add_argument('--datadir', default='../../../Data/ABAW5/', help='Dataset folder')
    parser.add_argument('--config', default=0, type=int, help="config number")
    parser.add_argument('--epochs', default=20, type=int, help="number of epoch")
    parser.add_argument('--batch', default=64, type=int, help="batch size")
    parser.add_argument('--length', default=64, type=int, help="max sequence length")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    output_dir = 'frameformer-{}'.format(args.config)


    train_annotation_path = args.datadir + 'annotations/EX/Train_Set/'
    valid_annotation_path = args.datadir + 'annotations/EX/Validation_Set/'
    path_lsd_t = '../../../Data/ABAW4/synthetic_challenge/training.txt'
    path_lsd_v = '../../../Data/ABAW4/synthetic_challenge/validation.txt'
    train_annotation_path = (train_annotation_path, path_lsd_t, path_lsd_v)

    with open(os.path.join(args.datadir, 'cropped_aligned/batch1/abaw5.pickle'), 'rb') as handle:
        abaw5_feature = pickle.load(handle)
    with open(os.path.join(args.datadir, '../ABAW4/synthetic_challenge/lsd_train_enet_b0_8_best_vgaf.pickle'), 'rb') as handle:
        lsd_t_feature = pickle.load(handle)
    with open(os.path.join(args.datadir, '../ABAW4/synthetic_challenge/lsd_valid_enet_b0_8_best_vgaf.pickle'), 'rb') as handle:
        lsd_v_feature = pickle.load(handle)
    feature_set = (abaw5_feature, lsd_t_feature, lsd_v_feature)

    image_path = args.datadir + 'cropped_aligned/batch1/cropped_aligned/'
    trainset = FrameABAW5(train_annotation_path, image_path, feature_set, args.length)
    validset = VFrameABAW5(valid_annotation_path, image_path, abaw5_feature, args.length)
    
    trainexw = torch.from_numpy(trainset.weigth).cuda()

    trainldr = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=0)
    validldr = DataLoader(validset, batch_size=1, shuffle=False, num_workers=0)

    start_epoch = 0
    net = Frameformer(1288, 8, 512, 8, 2048, 0.1, 6)
    net = net.double()

    if args.input != '':
        print("Resume form | {} ]".format(args.input))
        net = load_state_dict(net, args.input)

    net = net.cuda()
    criteria = nn.CrossEntropyLoss(reduction='mean', weight=trainexw, ignore_index=-1)
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
