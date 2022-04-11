import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
from utils import *
from sklearn import mixture
from sklearn.metrics import roc_auc_score

if torch.cuda.is_available:
    device = 'cuda:0'
else:
    device = 'cpu'

# Hyper parameter
task = 'PCG (PhysioNet)'
EPOCH_MAX = 100
block = 'LSTM'  # GRU , LSTM 可选
optimizer = 'Adam'  # SGD , Adam 可选
dropout = 0
latent_length = 30
batch_size = 640  # 560
input_size = 70  # 14 * 5 frequency domain 5 frames altogether
hidden1 = 32  # 64  128
hidden2 = 32  # 64  128
hidden3 = 16  # 32  64
hidden4 = 16  # 32  64
learning_rate = 0.01
# 自动调整学习率
# 每次更新参数的幅度，幅度过大，参数值波动，不收敛；幅度过小，待优化的参数收敛慢
training_loss_plot = []
val_loss_plot = []
test_loss_plot = []


def adjust_learning_rate(optimizer, epoch):
    global learning_rate
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def init_layer(layer, nonlinearity='leaky_relu'):
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


def init_rnnLayers(rLayer):
    for param in rLayer.parameters():
        if len(param.shape) >= 2:
            torch.nn.init.orthogonal_(param.data)
        else:
            torch.nn.init.normal_(param.data)


class Encoder(nn.Module):
    def __init__(self, input_size=input_size, hidden1=hidden1, hidden2=hidden2,
                 hidden3=hidden3, hidden4=hidden4, latent_length=latent_length):
        super(Encoder, self).__init__()

        # 定义属性
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.latent_length = latent_length

        # 设定网络
        self.input_to_hidden1 = nn.Linear(self.input_size, self.hidden1)
        self.hidden1_to_hidden2 = nn.Linear(self.hidden1, self.hidden2)
        self.hidden2_to_hidden3 = nn.Linear(self.hidden2, self.hidden3)
        self.hidden3_to_hidden4 = nn.Linear(self.hidden3, self.hidden4)

        self.hidden4_to_mean = nn.Linear(self.hidden4, self.latent_length)
        self.hidden4_to_logvar = nn.Linear(self.hidden4, self.latent_length)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        # 加入批标准化
        self.Batch1 = nn.BatchNorm1d(self.hidden1)
        self.Batch2 = nn.BatchNorm1d(self.hidden2)
        self.Batch3 = nn.BatchNorm1d(self.hidden3)
        self.Batch4 = nn.BatchNorm1d(self.hidden4)
        self.Batch = nn.BatchNorm1d(self.latent_length)
        # ???????
        nn.init.xavier_uniform_(self.hidden4_to_mean.weight)  # 为了通过网络层时，输入和输出的方差相同 服从均匀分布
        nn.init.xavier_uniform_(self.hidden4_to_logvar.weight)  # 为了通过网络层时，输入和输出的方差相同

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_bn(self.bn6)
        init_rnnLayers(self.rnnLayer)

    def forward(self, x):
        hidden1 = self.ReLU(self.Batch1(self.input_to_hidden1(x)))
        hidden2 = self.ReLU(self.Batch2(self.hidden1_to_hidden2(hidden1)))
        hidden3 = self.ReLU(self.Batch3(self.hidden2_to_hidden3(hidden2)))
        hidden4 = self.ReLU(self.Batch4(self.hidden3_to_hidden4(hidden3)))

        self.latent_mean = self.hidden4_to_mean(hidden4)
        self.latent_logvar = self.hidden4_to_logvar(hidden4)
        std = torch.exp(0.5 * self.latent_logvar)  # 化为log形式保证std为正值
        eps = torch.randn_like(std)  # 定义一个和std一样大小的服从标准正态分布的张量
        latent = torch.mul(eps, std) + self.latent_mean  # 标准正太分布乘以标准差后加上均值 latent.shape(batch,latent_length)
        return latent, self.latent_mean, self.latent_logvar  # x.shape(sqe,batch,input)
        # return latent


class Decoder(nn.Module):
    def __init__(self, output_size=input_size, hidden1=hidden1,
                 hidden2=hidden2, hidden3=hidden3, hidden4=hidden4, latent_length=latent_length):
        super(Decoder, self).__init__()

        # 定义属性
        self.output_size = output_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.latent_length = latent_length

        # 设定网络
        self.latent_to_hidden4 = nn.Linear(self.latent_length, self.hidden4)
        self.hidden4_to_hidden3 = nn.Linear(self.hidden4, self.hidden3)
        self.hidden3_to_hidden2 = nn.Linear(self.hidden3, self.hidden2)
        self.hidden2_to_hidden1 = nn.Linear(self.hidden2, self.hidden1)
        self.hidden1_to_output = nn.Linear(self.hidden1, self.output_size)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        # 加入批标准化
        self.Batch1 = nn.BatchNorm1d(self.hidden1)
        self.Batch2 = nn.BatchNorm1d(self.hidden2)
        self.Batch3 = nn.BatchNorm1d(self.hidden3)
        self.Batch4 = nn.BatchNorm1d(self.hidden4)

    def init_weights(self):
        init_layer(self.deconv1)
        init_layer(self.deconv2)
        init_layer(self.deconv3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_bn(self.bn6)
        init_rnnLayers(self.rnnLayer)

    def forward(self, latent):
        # BatchNorm1d + 线性变换
        hidden4 = self.ReLU(self.Batch4(self.latent_to_hidden4(latent)))
        hidden3 = self.ReLU(self.Batch3(self.hidden4_to_hidden3(hidden4)))
        hidden2 = self.ReLU(self.Batch2(self.hidden3_to_hidden2(hidden3)))
        hidden1 = self.ReLU(self.Batch1(self.hidden2_to_hidden1(hidden2)))
        output = self.hidden1_to_output(hidden1)

        return output


class Autoencoder(nn.Module):
    def __init__(self, input_size=input_size, hidden1=hidden1,
                 hidden2=hidden2, hidden3=hidden3, hidden4=hidden4, latent_length=latent_length):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)
        self.decoder = Decoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)

    def forward(self, x):
        latent, latent_mean, latent_logvar = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon, latent, latent_mean, latent_logvar


def calculation_latent():
    global optimizer, block, EPOCH_MAX, batch_size, learning_rate, \
        input_size, hidden1, hidden2, hidden3, hidden4, latent_length, device, training_loss_plot, val_loss_plot, \
        test_loss_plot
    X_train = np.load('data/' + '/X_train' + '.npy')  # (n_samples*batch,70)
    X_val_normal = np.load('data/' + '/X_val_normal' + '.npy')  # (n_samples*batch,70)
    X_val_abnormal = np.load('data/' + '/X_val_abnormal' + '.npy')  # (n_samples*batch,70)
    # X_train = X_train[:32, :]

    autoencoder = Autoencoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)
    autoencoder = autoencoder.float()
    criterion = nn.MSELoss()

    if optimizer == 'SGD':
        optimizer = optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9)  # 学习率，权重衰减
    else:
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    autoencoder = autoencoder.to(device)
    criterion = criterion.to(device)

    train_dataset = data.TensorDataset(torch.from_numpy(X_train))
    val_dataset = data.TensorDataset(torch.from_numpy(X_val_normal))
    test_dataset = data.TensorDataset(torch.from_numpy(X_val_abnormal))

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True  # 锁页内存 提高运行速度
    )

    validation_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=28,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=28,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    data_loaders = {"train": train_loader, "val": validation_loader, "test": test_loader}

    all_running_loss = []
    all_val_loss = []
    all_test_loss = []
    all_kl_loss = []
    all_training_latent = []
    all_val_latent = []
    all_test_latent = []
    optimizer.zero_grad()
    epoch_loss_everyfiveframes_normal = np.array([])
    epoch_loss_everyfiveframes_abnormal = np.array([])

    for epoch in range(EPOCH_MAX):
        for phase in ['train', 'val', 'test']:
            # for phase in ['train', 'test']:
            if phase == 'train':
                autoencoder.train()
            else:
                autoencoder.eval()

            for step, data_sample in enumerate(data_loaders[phase]):
                inputs = data_sample
                inputs = inputs[0]
                inputs = inputs.to(device)  # inputs.shape = (640, 70)

                # VAE
                outputs, latent, latent_mean, latent_logvar = autoencoder(inputs.float())

                latent_com = latent
                latent_out = latent_com.detach()

                kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                # kl_divergence()
                all_kl_loss.append(kl_loss.item())
                loss_real = criterion(inputs, outputs)
                loss = loss_real + kl_loss
                # # AE
                # outputs, latent = autoencoder(inputs.float())

                # latent_com = latent
                # latent_out = latent_com.detach()
                # loss_real = criterion(inputs, outputs)
                # loss = loss_real

                if phase == 'train':
                    all_running_loss.append(loss_real.item())  # 统计所有训练loss

                    if step == 0:
                        all_training_latent = latent_out.cpu().numpy()
                    else:
                        # 上下拼接
                        all_training_latent = np.concatenate((all_training_latent, latent_out.cpu().numpy()), axis=0)

                    loss.backward()
                    optimizer.step()

                elif phase == 'val':
                    all_val_loss.append(loss_real.item())  # 统计所有测试正常loss
                    all_val_latent.append(latent_out.cpu().numpy())  # 统计所有测试正常latent

                else:
                    all_test_loss.append(loss_real.item())  # 统计所有测试异常loss
                    all_test_latent.append(latent_out.cpu().numpy())  # 统计所有测试异常latent
                optimizer.zero_grad()

        # adjust_learning_rate(optimizer=optimizer, epoch=epoch)

        # running_loss average
        running_loss = np.mean(all_running_loss)
        if not training_loss_plot:
            training_loss_plot = [running_loss]
        else:
            training_loss_plot.append(running_loss)

        # val_loss average
        val_loss = np.mean(all_val_loss)
        if not val_loss_plot:
            val_loss_plot = [val_loss]
        else:
            val_loss_plot.append(val_loss)

        # test_loss average
        test_loss = np.mean(all_test_loss)
        if not test_loss_plot:
            test_loss_plot = [test_loss]
        else:
            test_loss_plot.append(test_loss)

        if os.path.exists('clustering/' + '/epoch/' + str(epoch)):
            pass
        else:
            os.makedirs('clustering/' + '/epoch/' + str(epoch))  # 一次创建到底

        # save latent
        np.save('clustering/' + '/epoch/' + str(epoch) + '/all_training_latent', all_training_latent)
        np.save('clustering/' + '/epoch/' + str(epoch) + '/all_val_latent', all_val_latent)
        np.save('clustering/' + '/epoch/' + str(epoch) + '/all_test_latent', all_test_latent)

        np.save('clustering/' + '/epoch/' + str(epoch) + '/all_running_loss', all_running_loss)
        np.save('clustering/' + '/epoch/' + str(epoch) + '/all_val_loss', all_val_loss)
        np.save('clustering/' + '/epoch/' + str(epoch) + '/all_test_loss', all_test_loss)

        np.save('clustering' + '/epoch/' + str(epoch) + '/loss_everyfiveframes_normal',
                epoch_loss_everyfiveframes_normal)
        np.save('clustering' + '/epoch/' + str(epoch) + '/loss_everyfiveframes_abnormal',
                epoch_loss_everyfiveframes_abnormal)

        # 除最后一轮外每个epoch清零
        if epoch == EPOCH_MAX - 1:
            pass
        else:
            all_running_loss = []
            all_val_loss = []
            all_test_loss = []
            all_kl_loss = []
            all_training_latent = []
            all_val_latent = []
            all_test_latent = []
            epoch_loss_everyfiveframes_normal = np.array([])
            epoch_loss_everyfiveframes_abnormal = np.array([])

        print('Train Epoch: {0}\nTraining Loss: {1:.5f}\nValidation Loss: {2:.5f}\nTest Loss: {3:.5f}\n'.format(
            epoch, running_loss, val_loss, test_loss))


def main():
    print('device:' + device)
    print('EPOCH_MAX:' + str(EPOCH_MAX))
    normal = np.load('data/test(other_file)_normal.npy')
    abnormal = np.load('data/test(other_file)_abnormal.npy')
    model = torch.load('autoencoder(0.805 200epochs).pkl')
    model.eval()
    val_dataset = data.TensorDataset(torch.from_numpy(normal))
    test_dataset = data.TensorDataset(torch.from_numpy(abnormal))
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    loss_normal = []
    loss_abnormal = []
    VALIDATE = data.DataLoader(
        dataset=val_dataset,
        batch_size=28,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    TEST = data.DataLoader(
        dataset=test_dataset,
        batch_size=28,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    for step, data_sample in enumerate(VALIDATE):
        inputs = data_sample
        inputs = inputs[0]
        inputs = inputs.to(device)  # inputs.shape = (640, 70)
        output, _, _, _ = model(inputs.float())
        loss = criterion(inputs, output)
        loss_normal.extend([loss.item()])

    for step, data_sample in enumerate(TEST):
        inputs = data_sample
        inputs = inputs[0]
        inputs = inputs.to(device)  # inputs.shape = (640, 70)
        output, _, _, _ = model(inputs.float())
        loss = criterion(inputs, output)
        loss_abnormal.extend([loss.item()])


    y_true = [0] * len(loss_normal) + [1] * len(loss_abnormal)
    y_pred = np.concatenate((loss_normal, loss_abnormal), axis=0)
    y_pred = np.array(y_pred)
    auc = roc_auc_score(y_true, y_pred)
    print(auc)


if __name__ == '__main__':
    main()
