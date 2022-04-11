import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import os
import scipy.spatial
from utils import *
from sklearn import mixture
from sklearn.metrics import roc_auc_score

if torch.cuda.is_available:
    device = 'cuda:0'
else:
    device = 'cpu'

# 重采样的存在会导致相同输入每次通过同一个自编码器输出会有差别。

# Hyper parameter
task = 'PCG (PhysioNet)'
EPOCH_MAX = 50  # 100
block = 'LSTM'  # GRU , LSTM 可选
optimizer = 'Adam'  # SGD , Adam 可选
dropout = 0
latent_length = 16  # 32
batch_size = 640  # 1200
input_size = 70  # 32（14） * 5 frequency domain 5 frames altogether
hidden1 = 32  # 32 64  128
hidden2 = 32  # 32 64  128
hidden3 = 16  # 16 32  64
hidden4 = 16  # 16 32  64
learning_rate = 0.0001
kl_ratio = 0.00
loss_ratio = 1.00
ratio = str('%.2f-%.2f' % (kl_ratio, loss_ratio))
save = True
# 自动调整学习率
# 每次更新参数的幅度，幅度过大，参数值波动，不收敛；幅度过小，待优化的参数收敛慢
training_loss_plot = []
val_loss_plot = []
test_loss_plot = []
data_type = '_m_a_density_ratio'  # _b_a_density_ratio , _only_b_a

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
        test_loss_plot, ratio
    # ############################# noise #####################################
    a_normal = np.load('data/' + 'a_normal' + '.npy')  # (n_samples*batch,70)
    a_abnormal = np.load('data/' + 'a_abnormal' + '.npy')  # (n_samples*batch,70)
    b_normal = np.load('data/' + 'b_normal' + '.npy')  # (n_samples*batch,70)
    b_abnormal = np.load('data/' + 'b_abnormal' + '.npy')  # (n_samples*batch,70)
    # c_normal = np.load('data/' + 'c_normal' + '.npy')  # (n_samples*batch,70)
    # c_abnormal = np.load('data/' + 'c_abnormal' + '.npy')  # (n_samples*batch,70)
    d_normal = np.load('data/' + 'd_normal' + '.npy')  # (n_samples*batch,70)
    d_abnormal = np.load('data/' + 'd_abnormal' + '.npy')  # (n_samples*batch,70)
    e_normal = np.load('data/' + 'e_normal' + '.npy')  # (n_samples*batch,70)
    e_abnormal = np.load('data/' + 'e_abnormal' + '.npy')  # (n_samples*batch,70)
    f_normal = np.load('data/' + 'f_normal' + '.npy')  # (n_samples*batch,70)
    f_abnormal = np.load('data/' + 'f_abnormal' + '.npy')  # (n_samples*batch,70)
    m_normal = np.load('data/' + 'm_normal' + '.npy')  # (n_samples*batch,70)
    m_abnormal = np.load('data/' + 'm_abnormal' + '.npy')  # (n_samples*batch,70)

    # f_normal_noise_6 = np.load('data/' + 'f_normal_noise_6' + '.npy')  # (n_samples*batch,70)
    # f_abnormal_noise_6 = np.load('data/' + 'f_abnormal_noise_6' + '.npy')  # (n_samples*batch,70)
    # # ############################# denoise #####################################
    # # a_normal = np.load('data_denoise/' + 'a_normal' + '.npy')  # (n_samples*batch,70)
    # # a_abnormal = np.load('data_denoise/' + 'a_abnormal' + '.npy')  # (n_samples*batch,70)
    # b_normal = np.load('data_denoise/' + 'b_normal' + '.npy')  # (n_samples*batch,70)
    # b_abnormal = np.load('data_denoise/' + 'b_abnormal' + '.npy')  # (n_samples*batch,70)
    # # c_normal = np.load('data_denoise/' + 'c_normal' + '.npy')  # (n_samples*batch,70)
    # # c_abnormal = np.load('data_denoise/' + 'c_abnormal' + '.npy')  # (n_samples*batch,70)
    # # d_normal = np.load('data_denoise/' + 'd_normal' + '.npy')  # (n_samples*batch,70)
    # # d_abnormal = np.load('data_denoise/' + 'd_abnormal' + '.npy')  # (n_samples*batch,70)
    # e_normal = np.load('data_denoise/' + 'e_normal' + '.npy')  # (n_samples*batch,70)
    # e_abnormal = np.load('data_denoise/' + 'e_abnormal' + '.npy')  # (n_samples*batch,70)
    # # f_normal = np.load('data_denoise/' + 'f_normal' + '.npy')  # (n_samples*batch,70)
    # # f_abnormal = np.load('data_denoise/' + 'f_abnormal' + '.npy')  # (n_samples*batch,70)
    # ############################# denoise_sw #####################################
    # # a_normal = np.load('data_denoise_sw/' + 'a_normal' + '.npy')  # (n_samples*batch,70)
    # # a_abnormal = np.load('data_denoise_sw/' + 'a_abnormal' + '.npy')  # (n_samples*batch,70)
    # b_normal = np.load('data_denoise_sw/' + 'b_normal' + '.npy')  # (n_samples*batch,70)
    # b_abnormal = np.load('data_denoise_sw/' + 'b_abnormal' + '.npy')  # (n_samples*batch,70)
    # # c_normal = np.load('data_denoise_sw/' + 'c_normal' + '.npy')  # (n_samples*batch,70)
    # # c_abnormal = np.load('data_denoise_sw/' + 'c_abnormal' + '.npy')  # (n_samples*batch,70)
    # # d_normal = np.load('data_denoise_sw/' + 'd_normal' + '.npy')  # (n_samples*batch,70)
    # # d_abnormal = np.load('data_denoise_sw/' + 'd_abnormal' + '.npy')  # (n_samples*batch,70)
    # e_normal = np.load('data_denoise_sw/' + 'e_normal' + '.npy')  # (n_samples*batch,70)
    # e_abnormal = np.load('data_denoise_sw/' + 'e_abnormal' + '.npy')  # (n_samples*batch,70)
    # # f_normal = np.load('data_denoise_sw/' + 'f_normal' + '.npy')  # (n_samples*batch,70)
    # # f_abnormal = np.load('data_denoise_sw/' + 'f_abnormal' + '.npy')  # (n_samples*batch,70)

    # X_all_normal = np.concatenate((a_normal, b_normal, c_normal, d_normal, e_normal, f_normal), axis=0)
    # X_all_normal = X_all_normal.reshape((-1, 28, 70))
    #
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
    if '_a_a' in data_type:
        X_train = a_normal[:2940]  # a:2940 b:9716 c:168 d:672 f:2016 e: 49336
        X_val_normal = a_normal[2940:]
        X_val_abnormal = a_abnormal
    elif '_b_a' in data_type:
        X_train = b_normal[:9716]  # a:2940 b:9716 c:168 d:672 f:2016 e: 49336
        X_val_normal = b_normal[9716:]
        X_val_abnormal = b_abnormal
    elif '_d_a' in data_type:
        X_train = d_normal[:672]  # a:2940 b:9716 c:168 d:672 f:2016 e: 49336
        X_val_normal = d_normal[672:]
        X_val_abnormal = d_abnormal
    elif '_e_a' in data_type:
        X_train = e_normal[:49336]  # a:2940 b:9716 c:168 d:672 f:2016 e: 49336
        X_val_normal = e_normal[49336:]
        X_val_abnormal = e_abnormal
    elif '_f_a' in data_type:
        X_train = f_normal[:2016]  # a:2940 b:9716 c:168 d:672 f:2016 e: 49336
        X_val_normal = f_normal[2016:]
        X_val_abnormal = f_abnormal
        # X_train = f_normal_noise_6[:2016]  # a:2940 b:9716 c:168 d:672 f:2016 e: 49336
        # X_val_normal = f_normal_noise_6[2016:]
        # X_val_abnormal = f_abnormal_noise_6
    elif '_m_a' in data_type:
        X_train = m_normal[:364]  # a:2940 b:9716 c:168 d:672 f:2016 e: 49336
        X_val_normal = m_normal[364:]
        X_val_abnormal = m_abnormal
    # elif '_ae_a' in data_type:
    #     X_train = np.concatenate((a_normal[:2940], e_normal[:49336]), axis=0)
    #     X_val_normal = np.concatenate((a_normal[2940:], e_normal[49336:]), axis=0)
    #     X_val_abnormal = np.concatenate((a_abnormal, e_abnormal), axis=0)
    # elif '_ef_a' in data_type:
    #     X_train = np.concatenate((e_normal[:49336], f_normal[:2016]), axis=0)
    #     X_val_normal = np.concatenate((e_normal[49336:], f_normal[2016:]), axis=0)
    #     X_val_abnormal = np.concatenate((e_abnormal, f_abnormal), axis=0)
    # elif '_all_a' in data_type:
    #     X_train = np.concatenate((a_normal[:2940], b_normal[:9716], d_normal[:672], e_normal[:49336], f_normal[:2016]), axis=0)
    #     X_val_normal = np.concatenate((a_normal[2940:], b_normal[9716:], d_normal[672:], e_normal[49336:], f_normal[2016:]), axis=0)
    #     X_val_abnormal = np.concatenate((a_abnormal, b_abnormal, d_abnormal, e_abnormal, f_abnormal), axis=0)
    elif '_be' in data_type:
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
    X_tset_normal = a_normal[2940:]
    X_test_abnormal = a_abnormal

    autoencoder = Autoencoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)
    autoencoder = autoencoder.float()
    criterion = nn.MSELoss()
    criterion_d = nn.MSELoss()
    # criterion_d = nn.CosineEmbeddingLoss()
    # criterion_d = nn.CosineSimilarity(dim=1, eps=1e-6)

    if optimizer == 'SGD':
        optimizer = optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9)  # 学习率，权重衰减
    else:
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    autoencoder = autoencoder.to(device)
    criterion = criterion.to(device)
    criterion_d = criterion_d.to(device)

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
        batch_size=28,  # 60
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    validation_loader2 = data.DataLoader(
        dataset=val_dataset2,
        batch_size=28,  # 28
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    test_loader1 = data.DataLoader(
        dataset=test_dataset1,
        batch_size=28,  # 28
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    test_loader2 = data.DataLoader(
        dataset=test_dataset2,
        batch_size=28,  # 28
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
    train_density_loss = []
    test_ae_normal_loss = []
    test_ae_abnormal_loss = []
    test_other_normal_loss = []
    test_other_abnormal_loss = []
    all_loss_density = []

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

                if numpy.isnan(inputs.cpu().numpy()).any():  # 判断数组中是否存在nan，存在则返回True
                    print('此处数组存在nan')

                # VAE
                outputs, latent, latent_mean, latent_logvar = autoencoder(inputs.float())
# ############################################# 求虚拟中心向量 #######################################################
                latent_com = latent
                latent_out = latent_com.detach()
                latent_out = latent_out.cpu().numpy()
                # ########### 新增部分，加入了密度中心隐藏向量latent_core ##########################
                latent_max = np.amax(latent_out, axis=0)
                latent_min = np.amin(latent_out, axis=0)
                latent_core = (latent_max + latent_min)/2

                latent_core = np.tile(latent_core, (latent_out.shape[0], 1))  # Construct an array by repeating A the number of times given by reps.
# ####################################################################################################################################
                # # ############ 新增部分，加入了密度中心隐藏向量latent_core，采用均值 ##########################
                # latent_core = np.mean(latent_out, axis=0)
                # latent_core = np.tile(latent_core, (latent_out.shape[0], 1))

                # output_com = outputs
                # outputs_out = output_com.detach()
                # outputs_out = outputs_out.cpu().numpy()
                # ########### 新增部分，加入了密度中心隐藏向量output_core ##########################
                # outputs_out_max = np.amax(outputs_out, axis=0)
                # outputs_out_min = np.amin(outputs_out, axis=0)
                # outputs_out_core = (outputs_out_max + outputs_out_min)/2
                # outputs_out_core = np.tile(outputs_out_core, (latent_out.shape[0], 1))

                kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                # kl_divergence()

                if phase == 'train':
                    loss_real = criterion(inputs, outputs)
                    # latent = latent.cpu().numpy()
                    # loss_density = scipy.spatial.distance.cdist(latent_core, latent.cpu(), "cosine")[0]
                    # loss_density = criterion_d(torch.unsqueeze(latent, 0), torch.unsqueeze(torch.from_numpy(latent_core).to(device), 0), torch.ones_like(latent))
                    # loss_density = criterion_d(latent.view(1, -1), torch.from_numpy(latent_core).view(1, -1).to(device), torch.ones_like(latent))

                    # loss = loss_ratio * loss_real + kl_ratio * kl_loss + loss_density
                    # loss = loss_ratio * loss_real + kl_ratio * kl_loss  # β-VAE用这个
                    # all_loss_density.append((kl_ratio * kl_loss).item())  # 直接保存的是系数乘密度距离之后的结果，这样就能看密度距离最终所占的数值了

                    loss_density = criterion_d(latent, torch.from_numpy(latent_core).to(device))
                    loss = loss_ratio * loss_real + kl_ratio * loss_density  # 密度距离用这个
                    # loss = loss_ratio * loss_real + kl_ratio * kl_loss  # kl散度用这个
                    all_loss_density.append((kl_ratio * loss_density).item())  # 直接保存的是系数乘密度距离之后的结果，这样就能看密度距离最终所占的数值了

                else:
                    loss = None
                    loss_real = []
                    for j in range(28):  # 一个样本由多少行的小样本构成 60
                        # a = inputs[j].view(1, -1)
                        # b = outputs[j].view(1, -1)
                        # test = criterion_d(a, b, torch.ones_like(a)).item()
                        # loss_real.append(test)
                        loss_real.append(criterion(inputs[j], outputs[j]).item())

                if phase == 'train':
                    all_running_loss.append(loss_real.item())  # 统计所有训练loss
                    all_running_kl_loss.append(kl_loss.item())  # 统计所有训练kl_loss

                    if step == 0:
                        all_training_latent = latent_out
                    else:
                        # 上下拼接
                        all_training_latent = np.concatenate((all_training_latent, latent_out), axis=0)

                    loss.backward()
                    optimizer.step()

                elif phase == 'val1':
                    all_val1_loss.extend(loss_real)  # 统计所有a,e正常loss
                    all_val1_kl_loss.append(kl_loss.item())  # 统计所有a,e正常kl_loss
                    all_val1_latent.append(latent_out)  # 统计所有a,e正常latent

                elif phase == 'val2':
                    all_val2_loss.extend(loss_real)  # 统计所有other正常loss
                    all_val2_kl_loss.append(kl_loss.item())  # 统计所有other正常kl_loss
                    all_val2_latent.append(latent_out)  # 统计所有other正常latent

                elif phase == 'test1':
                    all_test1_loss.extend(loss_real)  # 统计所有a,e异常loss
                    all_test1_kl_loss.append(kl_loss.item())  # 统计所有s,e异常kl_loss
                    all_test1_latent.append(latent_out)  # 统计所有a,e异常latent

                elif phase == 'test2':
                    all_test2_loss.extend(loss_real)  # 统计所有other异常loss
                    all_test2_kl_loss.append(kl_loss.item())  # 统计所有other异常kl_loss
                    all_test2_latent.append(latent_out)  # 统计所有other异常latent
                optimizer.zero_grad()

        # adjust_learning_rate(optimizer=optimizer, epoch=epoch)

        # loss average
        running_loss = np.mean(all_running_loss)
        val1_loss = np.mean(all_val1_loss)
        val2_loss = np.mean(all_val2_loss)
        test1_loss = np.mean(all_test1_loss)
        test2_loss = np.mean(all_test2_loss)
        density_loss = np.mean(all_loss_density)
        print('density距离', density_loss)

        train_loss.append(running_loss)
        test_ae_normal_loss.append(val1_loss)
        test_ae_abnormal_loss.append(test1_loss)
        test_other_normal_loss.append(val2_loss)
        test_other_abnormal_loss.append(test2_loss)
        train_density_loss.append(density_loss)

        # kl_loss average
        running_kl_loss = np.mean(all_running_kl_loss)
        val1_kl_loss = np.mean(all_val1_kl_loss)
        val2_kl_loss = np.mean(all_val2_kl_loss)
        test1_kl_loss = np.mean(all_test1_kl_loss)
        test2_kl_loss = np.mean(all_test2_kl_loss)

        train_kl_loss.append(running_kl_loss)
        test_ae_normal_kl_loss.append(val1_kl_loss)
        test_ae_abnormal_kl_loss.append(test1_kl_loss)
        test_other_normal_kl_loss.append(val2_kl_loss)
        test_other_abnormal_kl_loss.append(test2_kl_loss)

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

        # # save latent
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_training_latent', all_training_latent)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_val1_latent', all_val1_latent)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_val2_latent', all_val2_latent)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_test1_latent', all_test1_latent)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_test2_latent', all_test2_latent)
        # # save loss
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_running_loss', all_running_loss)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_val1_loss', all_val1_loss)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_val2_loss', all_val2_loss)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_test1_loss', all_test1_loss)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_test2_loss', all_test2_loss)
        # # save kl_loss
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_running_kl_loss', all_running_kl_loss)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_val1_kl_loss', all_val1_kl_loss)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_val2_kl_loss', all_val2_kl_loss)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_test1_kl_loss', all_test1_kl_loss)
        # np.save('clustering/' + '/epoch/' + str(epoch) + '/all_test2_kl_loss', all_test2_kl_loss)

        if save:
            # save latent
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_training_latent' + ratio, all_training_latent)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_val1_latent' + ratio, all_val1_latent)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_val2_latent' + ratio, all_val2_latent)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_test1_latent' + ratio, all_test1_latent)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_test2_latent' + ratio, all_test2_latent)
            # save loss
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_running_loss' + ratio, all_running_loss)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_val1_loss' + ratio, all_val1_loss)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_val2_loss' + ratio, all_val2_loss)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_test1_loss' + ratio, all_test1_loss)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_test2_loss' + ratio, all_test2_loss)
            # save kl_loss
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_running_kl_loss' + ratio, all_running_kl_loss)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_val1_kl_loss' + ratio, all_val1_kl_loss)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_val2_kl_loss' + ratio, all_val2_kl_loss)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_test1_kl_loss' + ratio, all_test1_kl_loss)
            np.save('clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/all_test2_kl_loss' + ratio, all_test2_kl_loss)

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
              .format(epoch, running_loss, val1_loss, val2_loss, test1_loss, test2_loss))
        print('Training KL_Loss: {0:.5f}\n'
              'Validation(a,e) KL_Loss: {1:.5f}    Validation(other) KL_Loss: {2:.5f}\n'
              'Test(a,e) KL_Loss: {3:.5f}          Test(other) KL_Loss: {4:.5f}\n'
              '\nauc(a,e) : {5:.5f}   auc(other) : {6:.5f}\n'
              .format(running_kl_loss, val1_kl_loss, val2_kl_loss, test1_kl_loss, test2_kl_loss, auc_ae, auc_other))
        torch.save(autoencoder, 'clustering' + data_type + '/' + '/epoch/' + str(epoch) + '/autoencoder' + ratio + '.pkl')

    if os.path.exists('data' + data_type):
        pass
    else:
        os.makedirs('data' + data_type)  # 一次创建到底

    if save:
        # 画图用loss，loss随着epoch进行的变化曲线
        np.save('data' + data_type + '/' + 'train_loss' + ratio, train_loss)
        np.save('data' + data_type + '/' + 'test_ae_normal_loss' + ratio, test_ae_normal_loss)
        np.save('data' + data_type + '/' + 'test_ae_abnormal_loss' + ratio, test_ae_abnormal_loss)
        np.save('data' + data_type + '/' + 'test_other_normal_loss' + ratio, test_other_normal_loss)
        np.save('data' + data_type + '/' + 'test_other_abnormal_loss' + ratio, test_other_abnormal_loss)

        np.save('data' + data_type + '/' + 'train_kl_loss' + ratio, train_kl_loss)
        np.save('data' + data_type + '/' + 'test_ae_normal_kl_loss' + ratio, test_ae_normal_kl_loss)
        np.save('data' + data_type + '/' + 'test_ae_abnormal_kl_loss' + ratio, test_ae_abnormal_kl_loss)
        np.save('data' + data_type + '/' + 'test_other_normal_kl_loss' + ratio, test_other_normal_kl_loss)
        np.save('data' + data_type + '/' + 'test_other_abnormal_kl_loss' + ratio, test_other_abnormal_kl_loss)
        # save density_loss
        np.save('data' + data_type + '/' + 'train_density_loss' + ratio, train_density_loss)


def main():
    print('device:' + device)
    print('EPOCH_MAX:' + str(EPOCH_MAX))
    # data_processing()
    kl_ratio_list = [0, 0.01, 0.10, 1.00, 10.00, 100.00]
    # kl_ratio_list = [10.00]
    for i in kl_ratio_list:
        global kl_ratio, loss_ratio, ratio
        kl_ratio = i
        loss_ratio = 1.00
        ratio = str('%.2f-%.2f' % (kl_ratio, loss_ratio))
        print(ratio)
        calculation_latent()


if __name__ == '__main__':
    main()
