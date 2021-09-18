import argparse
import os
import shutil
import torch
from SqueezeNet1 import *
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tripleloader1 import TripletImageLoader, TestImageLoader
from tripletnet import Tripletnet
import numpy as np
import csv
from tqdm import tqdm
import codecs
from losses import TripletLoss

parser = argparse.ArgumentParser(description='squeezenet-d')
parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--loss', default="SRT", help='loss Triplet or SRT')
parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='runs/TripletNet_vgg2/model_bestdataset2.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNet_vgg2', type=str,
                    help='name of experiment')

best_acc = 0


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
        TestImageLoader('test_disguised.txt','./VGG2-Disguised/',
                        transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            # transforms.CenterCrop((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    base_model = squeezeNet1()

    model = base_model

    tnet = Tripletnet(model)
    if args.cuda:
        tnet.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = TripletLoss(distance=args.loss).cuda()
    optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    acc = test(test_loader, tnet)


def test(test_loader, tnet):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()

    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3
        data1, data2 = Variable(data1), Variable(data2)

        # compute output
        embedded_x, embedded_y = tnet.embeddingnet(data1), tnet.embeddingnet(data2)
        dis = F.pairwise_distance(embedded_x, embedded_y, 2)
        dis = dis.detach().cpu().numpy()

        # measure accuracy and record loss
        acc = accuracy(dis, data3)
        accs.update(acc, 1)

        print('Test set({:d}/{:d}): Average loss: {:.4f}, Accuracy: {:.2f}%'.format(batch_idx, len(test_loader),
                                                                                    losses.avg, 100. * accs.avg))

    return accs.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(dista, label):
    index1 = np.where(label == 0)
    ta = dista[index1]
    ta1 = len(np.where(ta < 90)[0])
    index2 = np.where(label == 1)
    tb = dista[index2]
    tb1 = len(np.where(tb > 90)[0])
    return (ta1 + tb1) / len(label)


if __name__ == '__main__':
    main()
