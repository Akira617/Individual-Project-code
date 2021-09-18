from SSM import *
from network import *
import os
import argparse
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import csv
import codecs
import torchvision.transforms as transforms
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

class newmodel(nn.Module):

    def __init__(self, base_model, num_class=2):
        super().__init__()
        self.base = base_model
        #是squeezenet 则为512
        self.block = block(1000)
        self.fft = att(3)

        self.fc = nn.Linear(576, num_class)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x1 = self.base(x)
        x1 = self.block(x1)
        x2=self.fft(x)
        x3=x1+x2
        x4 = self.fc(x3)
        x4 = self.softmax(x4)
        return x4

def test_model(test_iter, model, device):
    model.eval()
    total_loss = 0.0
    accuracy = 0
    y_true = []
    y_pred = []
    total_test_num = len(test_iter.dataset)
    predict_container = np.zeros((0, 2))
    target_container = np.zeros((0, 2))
    for i, batch in enumerate(test_iter):
        feature, target = batch
        with torch.no_grad():
            feature, target = feature.to(device), target.to(device)
            out = model(feature)
            loss = F.cross_entropy(out, target)
            total_loss += loss.item()
            accuracy += (torch.argmax(out, dim=1) == target).sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
            pred_temp = out.data.cpu().numpy()
            predict_container = np.concatenate((predict_container, pred_temp), axis=0)
            onehot = torch.zeros(target.shape[0], 2).to(device)
            onehot.scatter_(1, target.unsqueeze(1), 1)
            target_container = np.concatenate((target_container, onehot.data.cpu().numpy()), axis=0)
            target_container = target_container.astype(np.uint8)
            print('>>> batch_{}/{}, Test loss is {}, Accuracy:{} '.format(i, len(test_iter), loss.item(), (
                (torch.argmax(out, dim=1) == target).sum().item()) / target.size(0)))
    # np.save('target.npy', target_container)
    # np.save('predict.npy', predict_container)
    print('>>> Test loss:{}, Accuracy:{} \n'.format(total_loss / total_test_num, accuracy / total_test_num))
    score = accuracy_score(y_true, y_pred)
    print(score)
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_true, y_pred)
    print(confusion_matrix)
    from sklearn.metrics import classification_report
    target_names = ['fake','real']
    print(classification_report(y_true, y_pred, target_names=target_names,digits=5))


    # ticks 坐标轴的坐标点
    # label 坐标轴标签说明
    indices = range(len(confusion_matrix))
    # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
    # plt.xticks(indices, [0, 1, 2])
    # plt.yticks(indices, [0, 1, 2])
    plt.figure(figsize=(8,8))
    plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
    plt.xticks(indices, target_names,rotation=45)
    plt.yticks(indices, target_names)

    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('混淆矩阵')

    # plt.rcParams两行是用于解决标签不能显示汉字的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示数据
    for first_index in range(len(confusion_matrix)):  # 第几行
        for second_index in range(len(confusion_matrix[first_index])):  # 第几列
            plt.text(first_index, second_index, confusion_matrix[first_index][second_index])
    # 在matlab里面可以对矩阵直接imagesc(confusion)
    # 显示
    plt.tight_layout()
    plt.savefig('test.png',dpi=300)
    plt.show()


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")


def train_model(train_iter, dev_iter, model, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    epochs = 20
    print('training...')
    lossData1 = [[]]
    lossData2 = [[]]
    acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        accuracy = 0
        total_train_num = len(train_iter.dataset)
        for i, batch in enumerate(train_iter):
            feature, target = batch
            feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            accuracy += (torch.argmax(logit, dim=1) == target).sum().item()
            print('>>> batch_{}/{}, Train loss is {}, Accuracy:{} '.format(i, len(train_iter), loss.item(), (
                (torch.argmax(logit, dim=1) == target).sum().item()) / target.size(0)))
            lossData1.append(
                [epoch, i, loss.item(), ((torch.argmax(logit, dim=1) == target).sum().item()) / target.size(0)])
        print('>>> Epoch_{}, Train loss is {}, Accuracy:{} '.format(epoch, total_loss / total_train_num,
                                                                    accuracy / total_train_num))
        model.eval()

        total_loss = 0.0
        accuracy = 0
        total_valid_num = len(dev_iter.dataset)
        for j, batch in enumerate(dev_iter):
            with torch.no_grad():
                feature, target = batch
                feature, target = feature.to(device), target.to(device)
                out = model(feature)
                loss = F.cross_entropy(out, target)
                total_loss += loss.item()
                accuracy += (torch.argmax(out, dim=1) == target).sum().item()
                print('>>> batch_{}/{}, Test loss is {}, Accuracy:{} '.format(j, len(test_iter), loss.item(), (
                    (torch.argmax(out, dim=1) == target).sum().item()) / target.size(0)))
                lossData2.append(
                    [epoch, j, loss.item(), ((torch.argmax(out, dim=1) == target).sum().item()) / target.size(0)])
        print('>>> Epoch_{}, Valid loss:{}, Accuracy:{} '.format(epoch, total_loss / total_valid_num,
                                                                 accuracy / total_valid_num))
        if accuracy / total_valid_num > acc:
            saveModel(model, 'squeeze')
            acc = accuracy / total_valid_num

        # data_write_csv("squeeze_train.csv", lossData1)
        # data_write_csv("squeeze_test.csv", lossData2)


def saveModel(model, name):
    torch.save(model.state_dict(), 'modelfile/' + name + '_model.pth')
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        h, w, c = img.shape
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        img = N + img
        img[img > 255] = 255  # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img
if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(description='Pretrain the models')
    parser.add_argument('-ft_dir', type=str, default='./dataset/FFHQ_TRAIN', help='train image directory')
    parser.add_argument('-val_dir', type=str, default='./dataset/FFHQ_TEST',
                        help='validation image directory')
    parser.add_argument('-img_height', type=str, default=64, help='image height')
    parser.add_argument('-img_width', type=int, default=64, help='image width')
    parser.add_argument('-batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('-es_patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('-reduce_factor', type=int, default=0.1, help='reduce factor')
    parser.add_argument('-reduce_patience', default=20, type=int, help='reduce patience')
    parser.add_argument('-epochs', type=int, default=30, help='epochs')
    parser.add_argument('-dropout_rate', type=int, default=0.2, help='dropout rate')
    parser.add_argument('-gpu_ids', type=str, default='0', help='select the GPU to use')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids


    base_model = squeezeNet1()

    ft_dir = args.ft_dir

    train_transform = transforms.Compose([
        transforms.Resize(86),
        transforms.RandomCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(86),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=args.ft_dir, transform=train_transform)
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)


    test_dataset = torchvision.datasets.ImageFolder(root=args.val_dir, transform=test_transform)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = newmodel(base_model=base_model)

    state_dict = torch.load('./modelfile/squeeze_model.pth')
    model.load_state_dict(state_dict)
    model = model.to(device)
    # 训练模型
    # train_model(train_iter, test_iter, model, device)
    # 测试模型
    test_model(test_iter, model, device)



