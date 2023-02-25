import os
import pickle
import pandas as pd
from dataset import LSD, ABAW5
from tqdm import tqdm
from torch.utils.data import DataLoader
from helpers import *
from block import Dense, AFER, MLP
import torch.optim as optim
import argparse
from sam import SAM

def train(net, trainldr, optimizer, epoch, epochs, criteria, lr):
    total_losses = AverageMeter()
    net.train()
    train_loader_len = len(trainldr)
    yhat = {}
    for batch_idx, (inputs, y) in enumerate(tqdm(trainldr)):
        adjust_learning_rate(optimizer, epoch, epochs, lr, batch_idx, train_loader_len)
        y = y.long()
        inputs = inputs.cuda()
        y = y.cuda()
        optimizer.zero_grad()
        yhat = net(inputs)
        loss = criteria(yhat, y)
        loss.backward()
        optimizer.step()
        total_losses.update(loss.data.item(), inputs.size(0))
    return total_losses.avg()


def val(net, validldr, criteria):
    total_losses = AverageMeter()
    yhat = {}
    net.eval()
    all_y = None
    all_yhat = None
    for batch_idx, (inputs, y) in enumerate(tqdm(validldr)):
        with torch.no_grad():
            y = y.long()
            inputs = inputs.cuda()
            y = y.cuda()
            yhat = net(inputs)
            loss = criteria(yhat, y)
            total_losses.update(loss.data.item(), inputs.size(0))

            if all_y == None:
                all_y = y.clone()
                all_yhat = yhat.clone()
            else:
                all_y = torch.cat((all_y, y), 0)
                all_yhat = torch.cat((all_yhat, yhat), 0)
    all_y = all_y.cpu().numpy()
    all_yhat = all_yhat.cpu().numpy()
    metrics = EX_metric(all_y, all_yhat)
    return total_losses.avg(), metrics


def main():
    parser = argparse.ArgumentParser(description='Train Emotion')

    parser.add_argument('--arc', default='AFER', help='Net name')
    parser.add_argument('--input', default='', help='Input file')
    parser.add_argument('--dataset', default='ABAW5', help='Output folder name')
    parser.add_argument('--datadir', default='../../../Data/ABAW5/', help='Output folder name')
    parser.add_argument('--sam', default=False, action='store_true', help='Apply SAM')
    parser.add_argument('--config', default=0, type=int, help="config number")
    parser.add_argument('--epochs', default=20, type=int, help="number of epoch")
    parser.add_argument('--batch', default=256, type=int, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    resume = args.input
    use_sam = args.sam
    net_name = args.arc
    epochs = args.epochs

    output_dir = '{}-{}-{}'.format(args.arc, args.dataset, args.config)

    if args.dataset == 'LSD':
        train_file = os.path.join(args.datadir, 'training.txt')
        valid_file = os.path.join(args.datadir, 'validation.txt')
        trainset = LSD(train_file, args.datadir + 'training')
        validset = LSD(valid_file, args.datadir + 'validation')
    else:
        train_annotation_path = args.datadir + 'annotations/EX/Train_Set/'
        valid_annotation_path = args.datadir + 'annotations/EX/Validation_Set/'
        image_path = args.datadir + 'cropped_aligned/batch1/cropped_aligned/'
        trainset = ABAW5(train_annotation_path, image_path)
        validset = ABAW5(valid_annotation_path, image_path)

    trainldr = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=0)
    validldr = DataLoader(validset, batch_size=args.batch, shuffle=False, num_workers=0)
    trainexw = torch.from_numpy(trainset.ex_weight())
    validexw = torch.from_numpy(validset.ex_weight())
    trainexw = trainexw.float()
    validexw = validexw.float()

    start_epoch = 0
    if net_name == 'AFER':
        net = AFER()
    elif net_name == 'MLP':
        net = MLP()
    else:
        net = Dense(1288, 6, activation='softmax')

    if resume != '':
        print("Resume form | {} ]".format(resume))
        net = load_state_dict(net, resume)

    net = nn.DataParallel(net).cuda()
    trainexw = trainexw.cuda()
    validexw = validexw.cuda()

    train_criteria = nn.CrossEntropyLoss(reduction='mean', weight=trainexw, ignore_index=-1)
    valid_criteria = nn.CrossEntropyLoss(reduction='mean', weight=validexw, ignore_index=-1)

    if use_sam:
        optimizer = SAM(net.parameters(), torch.optim.SGD, lr=args.lr, momentum=0.9, weight_decay=1.0/args.batch)
    else:
        optimizer = optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=args.lr, weight_decay=1.0/args.batch)
    best_performance = 0.0
    epoch_from_last_improvement = 0

    df = {}
    df['epoch'] = []
    df['lr'] = []
    df['train_loss'] = []
    df['val_loss'] = []
    df['val_metrics'] = []

    for epoch in range(start_epoch, epochs):
        lr = optimizer.param_groups[0]['lr']
        train_loss = train(net, trainldr, optimizer, epoch, epochs, train_criteria, args.lr)
        val_loss, val_metrics = val(net, validldr, valid_criteria)

        infostr = {'Epoch {}: {:.5f},{:.5f},{:.5f},{:.5f}'
                .format(epoch,
                        lr,
                        train_loss,
                        val_loss,
                        val_metrics)}
        print(infostr)

        os.makedirs(os.path.join('results', output_dir), exist_ok = True)

        if val_metrics >= best_performance:
            checkpoint = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join('results', output_dir, 'best_val_perform.pth'))
            best_performance = val_metrics
            epoch_from_last_improvement = 0
        else:
            epoch_from_last_improvement += 1

        checkpoint = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join('results', output_dir, 'cur_model.pth'))

        df['epoch'].append(epoch),
        df['lr'].append(lr),
        df['train_loss'].append(train_loss),
        df['val_loss'].append(val_loss),
        df['val_metrics'].append(val_metrics)

    df = pd.DataFrame(df)
    csv_name = os.path.join('results', output_dir, 'train.csv')
    df.to_csv(csv_name)

if __name__=="__main__":
    main()
