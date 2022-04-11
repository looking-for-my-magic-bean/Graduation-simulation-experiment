import torch
import torch.nn as nn
import torch.nn.functional as F
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
EPOCH_MAX = 50  # 100
block = 'LSTM'  # GRU , LSTM 可选
optimizer = 'Adam'  # SGD , Adam 可选
dropout = 0
latent_length = 16
batch_size = 640  # 520
input_size = 70  # 14 * 5 frequency domain 5 frames altogether
hidden1 = 32  # 64  128
hidden2 = 32  # 64  128
hidden3 = 16  # 32  64
hidden4 = 16  # 32  64
learning_rate = 0.0001
ratio = str('AE')
# 自动调整学习率
# 每次更新参数的幅度，幅度过大，参数值波动，不收敛；幅度过小，待优化的参数收敛慢
training_loss_plot = []
val_loss_plot = []
test_loss_plot = []
data_type = '_de'

np.random.seed(128)  # 66:90% 128:91% 192: 89% 256:89%  不同初始化种子会导致a_e训练后对域外数据识别率提高
torch.manual_seed(128)


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


def weight_init(net):
    # 递归获得net的所有子代Module
    for op in net.modules():
        # 针对不同类型操作采用不同初始化方式
        if isinstance(op, nn.Linear):
            nn.init.constant_(op.weight.data, val=0.01)
            nn.init.constant_(op.bias.data, val=0.01)
        # 这里可以对Conv等操作进行其它方式的初始化
        else:
            pass


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

        latent = self.hidden4_to_mean(hidden4)
        return latent  # x.shape(sqe,batch,input)
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
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon, latent


def calculation_latent():
    global optimizer, block, EPOCH_MAX, batch_size, learning_rate, \
        input_size, hidden1, hidden2, hidden3, hidden4, latent_length, device, training_loss_plot, val_loss_plot, \
        test_loss_plot, ratio

    a_normal = np.load('data/' + 'a_normal' + '.npy')  # (n_samples*batch,70)
    a_abnormal = np.load('data/' + 'a_abnormal' + '.npy')  # (n_samples*batch,70)
    b_normal = np.load('data/' + 'b_normal' + '.npy')  # (n_samples*batch,70)
    b_abnormal = np.load('data/' + 'b_abnormal' + '.npy')  # (n_samples*batch,70)
    c_normal = np.load('data/' + 'c_normal' + '.npy')  # (n_samples*batch,70)
    c_abnormal = np.load('data/' + 'c_abnormal' + '.npy')  # (n_samples*batch,70)
    d_normal = np.load('data/' + 'd_normal' + '.npy')  # (n_samples*batch,70)
    d_abnormal = np.load('data/' + 'd_abnormal' + '.npy')  # (n_samples*batch,70)
    e_normal = np.load('data/' + 'e_normal' + '.npy')  # (n_samples*batch,70)
    e_abnormal = np.load('data/' + 'e_abnormal' + '.npy')  # (n_samples*batch,70)
    f_normal = np.load('data/' + 'f_normal' + '.npy')  # (n_samples*batch,70)
    f_abnormal = np.load('data/' + 'f_abnormal' + '.npy')  # (n_samples*batch,70)
    m_normal = np.load('data/' + 'm_normal' + '.npy')  # (n_samples*batch,70)
    m_abnormal = np.load('data/' + 'm_abnormal' + '.npy')  # (n_samples*batch,70)

    if '_be' in data_type:
        X_train = np.concatenate((b_normal[:9716], e_normal[:49336]), axis=0)
        X_val_normal = np.concatenate((b_normal[9716:], e_normal[49336:]), axis=0)
        X_val_abnormal = np.concatenate((b_abnormal, e_abnormal), axis=0)
    elif '_de' in data_type:
        X_train = np.concatenate((d_normal[:672], e_normal[:49336]), axis=0)
        X_val_normal = np.concatenate((d_normal[672:], e_normal[49336:]), axis=0)
        X_val_abnormal = np.concatenate((d_abnormal, e_abnormal), axis=0)
    else:
        print('请输入正确的data_type以便进行计算')
        X_train = []
        X_val_normal = []
        X_val_abnormal = []
    X_tset_normal = m_normal
    X_test_abnormal = m_abnormal

    # ind_cut = int(0.8 * X_all_normal.shape[0])  # 训练样本与测试样本个数分界线
    # ind = np.random.permutation(X_all_normal.shape[0])  # 产生N个随机数，用于将样本随机化
    #
    # X_train = X_all_normal[ind[:ind_cut]]
    # X_train = X_train.reshape((-1, 70))
    # X_val_normal = X_all_normal[ind[ind_cut:]]
    # X_val_normal = X_val_normal.reshape((-1, 70))
    # X_val_abnormal = np.concatenate((a_abnormal, b_abnormal, c_abnormal, d_abnormal, e_abnormal, f_abnormal), axis=0)
    # X_tset_normal = c_normal
    # X_test_abnormal = c_abnormal

    # X_train = np.concatenate((e_normal[:49336], f_normal[:1288]), axis=0)
    # X_val_normal = np.concatenate((e_normal[49336:], f_normal[1288:]), axis=0)
    # X_val_abnormal = np.concatenate((e_abnormal, f_abnormal), axis=0)
    # X_tset_normal = np.concatenate((a_normal, b_normal, c_normal, d_normal), axis=0)
    # X_test_abnormal = np.concatenate((a_abnormal, b_abnormal, c_abnormal, d_abnormal), axis=0)

    # X_train = X_train[:32, :]

    autoencoder = Autoencoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)
    # weight_init(autoencoder)
    # for param in autoencoder.named_parameters():
    #     print(param)
    # print('******************************************************************************')
    autoencoder = autoencoder.float()
    criterion = nn.MSELoss()

    if optimizer == 'SGD':
        optimizer = optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9)  # 学习率，权重衰减
    else:
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    autoencoder = autoencoder.to(device)
    criterion = criterion.to(device)

    train_dataset = data.TensorDataset(torch.from_numpy(X_train))
    val_dataset1 = data.TensorDataset(torch.from_numpy(X_val_normal))  # a,e组的正常样本
    test_dataset1 = data.TensorDataset(torch.from_numpy(X_val_abnormal))  # a,e组的异常样本
    val_dataset2 = data.TensorDataset(torch.from_numpy(X_tset_normal))  # 其他组的正常样本
    test_dataset2 = data.TensorDataset(torch.from_numpy(X_test_abnormal))  # 其他组的异常样本

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True  # 锁页内存 提高运行速度
    )

    validation_loader1 = data.DataLoader(
        dataset=val_dataset1,
        batch_size=28,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    validation_loader2 = data.DataLoader(
        dataset=val_dataset2,
        batch_size=28,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    test_loader1 = data.DataLoader(
        dataset=test_dataset1,
        batch_size=28,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    test_loader2 = data.DataLoader(
        dataset=test_dataset2,
        batch_size=28,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    data_loaders = {"train": train_loader,
                    "val1": validation_loader1,
                    "val2": validation_loader2,
                    "test1": test_loader1,
                    "test2": test_loader2
                    }
    # loss
    all_running_loss = []
    all_val1_loss = []
    all_val2_loss = []
    all_test1_loss = []
    all_test2_loss = []
    train_loss = []
    test_ae_normal_loss = []
    test_ae_abnormal_loss = []
    test_other_normal_loss = []
    test_other_abnormal_loss = []

    # kl_loss
    all_running_kl_loss = []
    all_val1_kl_loss = []
    all_val2_kl_loss = []
    all_test1_kl_loss = []
    all_test2_kl_loss = []
    train_kl_loss = []
    test_ae_normal_kl_loss = []
    test_ae_abnormal_kl_loss = []
    test_other_normal_kl_loss = []
    test_other_abnormal_kl_loss = []

    # latent
    all_training_latent = []
    all_val1_latent = []
    all_val2_latent = []
    all_test1_latent = []
    all_test2_latent = []
    optimizer.zero_grad()

    for epoch in range(EPOCH_MAX):
        for phase in ['train', 'val1', 'val2', 'test1', 'test2']:
            if phase == 'train':
                autoencoder.train()
            else:
                autoencoder.eval()

            for step, data_sample in enumerate(data_loaders[phase]):
                inputs = data_sample
                inputs = inputs[0]
                inputs = inputs.to(device)  # inputs.shape = (640, 70)

                # AE
                outputs, latent = autoencoder(inputs.float())

                latent_com = latent
                latent_out = latent_com.detach()

                if phase == 'train':
                    loss_real = criterion(inputs, outputs)
                else:
                    loss_real = []
                    for j in range(28):
                        loss_real.append(criterion(inputs[j], outputs[j]).item())

                if phase == 'train':
                    all_running_loss.append(loss_real.item())  # 统计所有训练loss

                    if step == 0:
                        all_training_latent = latent_out.cpu().numpy()
                    else:
                        # 上下拼接
                        all_training_latent = np.concatenate((all_training_latent, latent_out.cpu().numpy()), axis=0)

                    loss_real.backward()
                    optimizer.step()

                elif phase == 'val1':
                    all_val1_loss.extend(loss_real)  # 统计所有a,e正常loss
                    all_val1_latent.append(latent_out.cpu().numpy())  # 统计所有a,e正常latent

                elif phase == 'val2':
                    all_val2_loss.extend(loss_real)  # 统计所有other正常loss
                    all_val2_latent.append(latent_out.cpu().numpy())  # 统计所有other正常latent

                elif phase == 'test1':
                    all_test1_loss.extend(loss_real)  # 统计所有a,e异常loss
                    all_test1_latent.append(latent_out.cpu().numpy())  # 统计所有a,e异常latent

                elif phase == 'test2':
                    all_test2_loss.extend(loss_real)  # 统计所有other异常loss
                    all_test2_latent.append(latent_out.cpu().numpy())  # 统计所有other异常latent
                optimizer.zero_grad()

        # adjust_learning_rate(optimizer=optimizer, epoch=epoch)

        # loss average
        running_loss = np.mean(all_running_loss)
        val1_loss = np.mean(all_val1_loss)
        val2_loss = np.mean(all_val2_loss)
        test1_loss = np.mean(all_test1_loss)
        test2_loss = np.mean(all_test2_loss)

        train_loss.append(running_loss)
        test_ae_normal_loss.append(val1_loss)
        test_ae_abnormal_loss.append(test1_loss)
        test_other_normal_loss.append(val2_loss)
        test_other_abnormal_loss.append(test2_loss)

        y_true_ae = [0] * len(all_val1_loss) + [1] * len(all_test1_loss)
        y_pred_ae = np.concatenate((all_val1_loss, all_test1_loss), axis=0)
        y_true_ae = np.array(y_true_ae)
        auc_ae = roc_auc_score(y_true_ae, y_pred_ae)

        y_true_other = [0] * len(all_val2_loss) + [1] * len(all_test2_loss)
        y_pred_other = np.concatenate((all_val2_loss, all_test2_loss), axis=0)
        y_true_other = np.array(y_true_other)
        auc_other = roc_auc_score(y_true_other, y_pred_other)

        if os.path.exists('clustering' + data_type + '/' + '/epoch/' + str(epoch)):
            pass
        else:
            os.makedirs('clustering' + data_type + '/' + '/epoch/' + str(epoch))  # 一次创建到底

        # save latent
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_training_latent' + ratio, all_training_latent)
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_val1_latent' + ratio, all_val1_latent)
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_val2_latent' + ratio, all_val2_latent)
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_test1_latent' + ratio, all_test1_latent)
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_test2_latent' + ratio, all_test2_latent)
        # save loss
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_running_loss' + ratio, all_running_loss)
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_val1_loss' + ratio, all_val1_loss)
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_val2_loss' + ratio, all_val2_loss)
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_test1_loss' + ratio, all_test1_loss)
        np.save('clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/all_test2_loss' + ratio, all_test2_loss)

        torch.save(autoencoder, 'clustering' + data_type + '/' + 'epoch/' + str(epoch) + '/autoencoder' + ratio + '.pkl')

        # 除最后一轮外每个epoch清零
        if epoch == EPOCH_MAX - 1:
            pass
        else:
            # loss
            all_running_loss = []
            all_val1_loss = []
            all_val2_loss = []
            all_test1_loss = []
            all_test2_loss = []
            # kl_loss
            all_running_kl_loss = []
            all_val1_kl_loss = []
            all_val2_kl_loss = []
            all_test1_kl_loss = []
            all_test2_kl_loss = []
            # latent
            all_training_latent = []
            all_val1_latent = []
            all_val2_latent = []
            all_test1_latent = []
            all_test2_latent = []

        print('******************* Train Epoch: {0} ********************************\n'
              'Training Loss: {1:.5f}\n'
              'Validation(a,e) Loss: {2:.5f}       Validation(other) Loss: {3:.5f}\n'
              'Test(a,e) Loss: {4:.5f}             Test(other) Loss: {5:.5f}\n'
              '\nauc(a,e) : {6:.5f}   auc(other) : {7:.5f}\n'
              .format(epoch, running_loss, val1_loss, val2_loss, test1_loss, test2_loss, auc_ae, auc_other))

    # np.save('data/' + 'train_loss', train_loss)
    # np.save('data/' + 'test_ae_normal_loss', test_ae_normal_loss)
    # np.save('data/' + 'test_ae_abnormal_loss', test_ae_abnormal_loss)
    # np.save('data/' + 'test_other_normal_loss', test_other_normal_loss)
    # np.save('data/' + 'test_other_abnormal_loss', test_other_abnormal_loss)
    #
    # np.save('data/' + 'train_kl_loss', train_kl_loss)
    # np.save('data/' + 'test_ae_normal_kl_loss', test_ae_normal_kl_loss)
    # np.save('data/' + 'test_ae_abnormal_kl_loss', test_ae_abnormal_kl_loss)
    # np.save('data/' + 'test_other_normal_kl_loss', test_other_normal_kl_loss)
    # np.save('data/' + 'test_other_abnormal_kl_loss', test_other_abnormal_kl_loss)

    np.save('data' + data_type + '/' + 'train_loss' + ratio, train_loss)
    np.save('data' + data_type + '/' + 'test_ae_normal_loss' + ratio, test_ae_normal_loss)
    np.save('data' + data_type + '/' + 'test_ae_abnormal_loss' + ratio, test_ae_abnormal_loss)
    np.save('data' + data_type + '/' + 'test_other_normal_loss' + ratio, test_other_normal_loss)
    np.save('data' + data_type + '/' + 'test_other_abnormal_loss' + ratio, test_other_abnormal_loss)


def main():
    print('device:' + device)
    print('EPOCH_MAX:' + str(EPOCH_MAX))
    # #
    # if os.path.exists('data/X_train.npy') and os.path.exists('data/X_val_abnormal.npy') and os.path.exists(
    #         'data/X_val_normal.npy'):
    #     pass
    # else:
    #     data_processing()
    # data_processing()
    calculation_latent()

    # input = np.load('data/' + 'X_val_abnormal.npy')
    # input_real = input[52*28:52*28+28, :]
    # model = torch.load('autoencoder.pkl')
    # model.eval()
    # input_real = torch.from_numpy(input_real).to(device).float()
    # output = model(input_real)
    # criterion = nn.MSELoss()
    # criterion = criterion.to(device)
    # loss = criterion(input_real, output[0])
    # print(loss)


if __name__ == '__main__':
    main()
