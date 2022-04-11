import sys

import librosa
import numpy as np
import os
from python_speech_features import *
import pylab as pl
from matplotlib import pyplot
import matplotlib.pyplot as plt
from filter import *
# Hyper parameters
n_mels = 14
n_fft = 1024
hop_length = 512
power = 2.0


def calculate_normal():
    path = r'D:\PhysioNet\PCG\train\a0195.wav'
    y, samplerate = librosa.load(path=path, sr=2000, mono=False)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=2000,
                                                     n_fft=1024,
                                                     hop_length=512,
                                                     n_mels=14,
                                                     power=2.0)  # [14, 32]

    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
    log_mel_spectrogram = log_mel_spectrogram[0:14, 0:32]
    if log_mel_spectrogram.ndim == 2:
        axis = 0
    elif log_mel_spectrogram.ndim == 3:
        axis = (0, 1)
    mean_normal = np.mean(log_mel_spectrogram, axis=axis)
    std_normal = np.std(log_mel_spectrogram, axis=axis)
    np.save('mean_normal.npy', mean_normal)
    np.save('std_normal.npy', std_normal)


def calculate_abnormal():
    path = r'D:\PhysioNet\PCG\test\abnormal\a0001.wav'
    y, samplerate = librosa.load(path=path, sr=2000, mono=False)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=2000,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)  # [14, 32]
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
    log_mel_spectrogram = log_mel_spectrogram[0:14, 0:32]
    if log_mel_spectrogram.ndim == 2:
        axis = 0
    elif log_mel_spectrogram.ndim == 3:
        axis = (0, 1)
    mean_abnormal = np.mean(log_mel_spectrogram, axis=axis)
    std_abnormal = np.std(log_mel_spectrogram, axis=axis)
    np.save('mean_abnormal.npy', mean_abnormal)
    np.save('std_abnormal.npy', std_abnormal)


def data_train(datadir=r'D:\PhysioNet\PCG\train/', frames=5, numcep=14):
    frequency_spectrum_train = []
    files1 = os.listdir(datadir)

    for everyone in files1:
        print(everyone)
        train_signal, samplerate = librosa.load(datadir + everyone, sr=None, mono=False)
        train_frequency = file_to_vector_array(train_signal, n_mels=numcep, frames=frames, n_fft=1024,
                                               hop_length=512)

        if frequency_spectrum_train == []:
            frequency_spectrum_train = train_frequency
        else:
            frequency_spectrum_train = np.concatenate((frequency_spectrum_train, train_frequency), axis=0)
            # 按第一维度拼接， return (n samples * 28, 70)
            # 返回一个二维的矩阵
    train = frequency_spectrum_train

    return train


def data_process(group, datadir='D:/PhysioNet/ALL/', frames=5, numcep=14, nfft=1024, hoplength=512, filter='SW', snr=6):
    # frames=5, numcep=32, nfft=512, hoplength=266

    frequency_spectrum_test_normal = []
    frequency_spectrum_test_abnormal = []
    files_normal = os.listdir(datadir + 'normal/')
    files_abnormal = os.listdir(datadir + 'abnormal/')
    k = 0
    for everyone in files_normal:
        if group in everyone:
            if k == 0:
                print('normal')
                k += 1
            print(everyone)
            test_signal, samplerate = librosa.load(datadir + 'normal/' + everyone, sr=2000, mono=False)
            if test_signal.shape[0] == 2:
                test_signal = test_signal[0]

            # if len(test_signal) < 18000:  # 低于9秒  2000hz*9=18000个点
            #     test_signal = np.concatenate((test_signal, test_signal), axis=0)
            if filter == 'SW':
                test_signal = denosie_SWFMH(test_signal, 10)  # SWFMH滤波器
            elif filter == 'CW':
                test_signal = denosie_CWFMH(test_signal, 10)  # CWFMH滤波器
            elif filter == 'noise':
                test_signal = awgn(test_signal, snr=snr, out='signal', method='vectorized', axis=0)  # 加噪声处理 信噪比为snr
            else:
                pass

            test_frequency = file_to_vector_array(test_signal, n_mels=numcep, frames=frames, n_fft=nfft,
                                                  hop_length=hoplength)
            if test_frequency.shape[0] != 28:
                print(test_frequency.shape)

            if frequency_spectrum_test_normal == []:
                frequency_spectrum_test_normal = test_frequency  # equals to test_normal
            else:
                # 上下拼接
                frequency_spectrum_test_normal = np.concatenate((frequency_spectrum_test_normal, test_frequency),
                                                                axis=0)
    k = 0
    for everyone in files_abnormal:
        if group in everyone:
            if k == 0:
                print('abnormal')
                k += 1
            print(everyone)
            test_signal, samplerate = librosa.load(datadir + 'abnormal/' + everyone, sr=2000, mono=False)
            if test_signal.shape[0] == 2:
                test_signal = test_signal[0]
            # if len(test_signal) < 18000:  # 低于9秒  2000hz*9=18000个点
            #     test_signal = np.concatenate((test_signal, test_signal), axis=0)

            if filter == 'SW':
                test_signal = denosie_SWFMH(test_signal, 10)  # SWFMH滤波器
            elif filter == 'CW':
                test_signal = denosie_CWFMH(test_signal, 10)  # CWFMH滤波器
            elif filter == 'noise':
                test_signal = awgn(test_signal, snr=snr, out='signal', method='vectorized', axis=0)  # 加噪声处理 信噪比为snr
            else:
                pass

            test_frequency = file_to_vector_array(test_signal, n_mels=numcep, frames=frames, n_fft=nfft,
                                                  hop_length=hoplength)
            if test_frequency.shape[0] != 28:
                print(test_frequency.shape)

            if frequency_spectrum_test_abnormal == []:
                frequency_spectrum_test_abnormal = test_frequency
            else:
                frequency_spectrum_test_abnormal = np.concatenate((frequency_spectrum_test_abnormal, test_frequency),
                                                                  axis=0)  # 上下拼接

    normal = frequency_spectrum_test_normal
    abnormal = frequency_spectrum_test_abnormal
    return normal, abnormal


def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    # std[std == 0] = float("inf")
    if np.any(std == 0):
        print('std=0')
        x = x
    else:
        x = (x - mean) / std
    return x


def file_to_vector_array(y, n_mels, frames, n_fft, hop_length, power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    # y = (y - y.mean()) / y.std()
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=2000,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)  # [14, 32]
    # can be modified and
    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    if log_mel_spectrogram.shape[1] < 32:  # 改 64
        print('前:', log_mel_spectrogram.shape)
        log_mel_spectrogram = np.concatenate((log_mel_spectrogram, log_mel_spectrogram), axis=1)
        print('后:', log_mel_spectrogram.shape)
        log_mel_spectrogram = log_mel_spectrogram[:, 0:32]
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, 0:32]

    log_mel_spectrogram = calculate_scalar_of_tensor(log_mel_spectrogram)
    # 04 calculate total vector size

    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array  # [28, 70]  # [60,160]


def data_test_normal_anomaly(datadir=r'D:\PhysioNet\PCG\test_otherfile/', frames=5, numcep=14):
    frequency_spectrum_test_normal = []
    frequency_spectrum_test_abnormal = []
    files_normal = os.listdir(datadir + 'normal/')
    files_abnormal = os.listdir(datadir + 'abnormal/')
    for everyone in files_normal:
        print(everyone)
        test_signal, samplerate = librosa.load(datadir + 'normal/' + everyone, sr=2000, mono=False)
        test_frequency = file_to_vector_array(test_signal, n_mels=numcep, frames=frames, n_fft=1024,
                                              hop_length=512)
        if frequency_spectrum_test_normal == []:
            frequency_spectrum_test_normal = test_frequency  # equals to test_normal
        else:
            # 上下拼接
            frequency_spectrum_test_normal = np.concatenate((frequency_spectrum_test_normal, test_frequency), axis=0)
    for everyone in files_abnormal:
        print(everyone)
        test_signal, samplerate = librosa.load(datadir + 'abnormal/' + everyone, sr=2000, mono=False)
        test_frequency = file_to_vector_array(test_signal, n_fft=1024, n_mels=numcep, frames=frames,
                                              hop_length=512)
        if frequency_spectrum_test_abnormal == []:
            frequency_spectrum_test_abnormal = test_frequency
        else:
            frequency_spectrum_test_abnormal = np.concatenate((frequency_spectrum_test_abnormal, test_frequency),
                                                              axis=0)  # 上下拼接

    normal = frequency_spectrum_test_normal
    abnormal = frequency_spectrum_test_abnormal
    return normal, abnormal


def data_processing():

    ff = 'noise'
    # a_normal, a_abnormal = data_process('a0', filter=ff)  # .wav中含有a
    # b_normal, b_abnormal = data_process('b0', filter=ff)
    # c_normal, c_abnormal = data_process('c0', filter=ff)
    # d_normal, d_abnormal = data_process('d0', filter=ff)
    # e_normal, e_abnormal = data_process('e0', filter=ff)
    # f_normal, f_abnormal = data_process('f0', filter=ff)
    # m_normal, m_abnormal = data_process('0', 'D:/Michigan_training/')
    # f_normal_noise_6, f_abnormal_noise_6 = data_process('f0', filter=ff, snr=6)

    if os.path.exists('data/'):
        pass
    else:
        os.makedirs('data/')  # 一次创建到底

    # np.save('data/' + '/a_normal', a_normal)
    # np.save('data/' + '/a_abnormal', a_abnormal)
    # np.save('data/' + '/b_normal', b_normal)
    # np.save('data/' + '/b_abnormal', b_abnormal)
    # np.save('data/' + '/c_normal', c_normal)
    # np.save('data/' + '/c_abnormal', c_abnormal)
    # np.save('data/' + '/d_normal', d_normal)
    # np.save('data/' + '/d_abnormal', d_abnormal)
    # np.save('data/' + '/e_normal', e_normal)
    # np.save('data/' + '/e_abnormal', e_abnormal)
    # np.save('data/' + '/f_normal', f_normal)
    # np.save('data/' + '/f_abnormal', f_abnormal)
    # np.save('data/' + '/m_normal', m_normal)
    # np.save('data/' + '/m_abnormal', m_abnormal)
    # np.save('data/' + '/f_normal_noise_6', f_normal_noise_6)
    # np.save('data/' + '/f_abnormal_noise_6', f_abnormal_noise_6)

    # np.save('data_denoise/' + '/a_normal', a_normal)
    # np.save('data_denoise/' + '/a_abnormal', a_abnormal)
    # np.save('data_denoise/' + '/b_normal', b_normal)
    # np.save('data_denoise/' + '/b_abnormal', b_abnormal)
    # np.save('data_denoise/' + '/c_normal', c_normal)
    # np.save('data_denoise/' + '/c_abnormal', c_abnormal)
    # np.save('data_denoise/' + '/d_normal', d_normal)
    # np.save('data_denoise/' + '/d_abnormal', d_abnormal)
    # np.save('data_denoise/' + '/e_normal', e_normal)
    # np.save('data_denoise/' + '/e_abnormal', e_abnormal)
    # np.save('data_denoise/' + '/f_normal', f_normal)
    # np.save('data_denoise/' + '/f_abnormal', f_abnormal)

    # np.save('data_denoise_sw/' + '/a_normal', a_normal)
    # np.save('data_denoise_sw/' + '/a_abnormal', a_abnormal)
    # np.save('data_denoise_sw/' + '/b_normal', b_normal)
    # np.save('data_denoise_sw/' + '/b_abnormal', b_abnormal)
    # np.save('data_denoise_sw/' + '/c_normal', c_normal)
    # np.save('data_denoise_sw/' + '/c_abnormal', c_abnormal)
    # np.save('data_denoise_sw/' + '/d_normal', d_normal)
    # np.save('data_denoise_sw/' + '/d_abnormal', d_abnormal)
    # np.save('data_denoise_sw/' + '/e_normal', e_normal)
    # np.save('data_denoise_sw/' + '/e_abnormal', e_abnormal)
    # np.save('data_denoise_sw/' + '/f_normal', f_normal)
    # np.save('data_denoise_sw/' + '/f_abnormal', f_abnormal)


def plot_loss(train_loss, val_loss, test_loss):

    plt.subplot(2, 1, 1)
    plt.plot(train_loss, color='r', label='train_loss')
    plt.plot(val_loss, color='b', label='test_normal_loss')
    plt.plot(test_loss, color='y', label='test_abnormal_loss')
    plt.legend()
    plt.xlabel('Epoch 0-99')
    plt.ylabel('Loss')
    plt.savefig('clustering/Epoch 0-99.png')

    plt.subplot(2, 1, 2)
    plt.plot(train_loss[10:], color='r', label='train_loss')
    plt.plot(val_loss[10:], color='b', label='test_normal_loss')
    plt.plot(test_loss[10:], color='y', label='test_abnormal_loss')
    plt.legend()
    plt.xlabel('Epoch 10-99')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('clustering/Epoch 10-99.png')

    plt.show()


# def delete_wrong_files():
#     import os
#     path = 'clustering/epoch/'
#     for i in range(100):
#         os.remove(path+'{}/all_traning_latent.npy'.format(i))
#         print(path+'{}/all_traning_latent.npy'.format(i)+' deleted')
#

def reload_loss(epoch):
    test_abnormal_loss = np.load('clustering/epoch/'+str(epoch)+'/all_test_loss.npy')
    test_normal_loss = np.load('clustering/epoch/'+str(epoch)+'/all_val_loss.npy')
    train_loss = np.load('clustering/epoch/'+str(epoch)+'/all_running_loss.npy')
    print(test_abnormal_loss.shape)
    print(test_normal_loss.shape)
    print(train_loss.shape)


def reload_latent(epoch):
    test_abnormal_latent = np.load('clustering/epoch/'+str(epoch)+'/all_test_latent.npy', allow_pickle=True)
    test_normal_latent = np.load('clustering/epoch/'+str(epoch)+'/all_val_latent.npy', allow_pickle=True)
    train_latent = np.load('clustering/epoch/'+str(epoch)+'/all_training_latent.npy', allow_pickle=True)
    print(train_latent.shape)
    print(test_normal_latent.shape)
    print(test_abnormal_latent.shape)


def main():
    epoch_loss_normal = np.load('clustering/epoch/99/loss_everyfiveframes_normal.npy', allow_pickle=True)
    epoch_loss_abnormal = np.load('clustering/epoch/99/loss_everyfiveframes_abnormal.npy', allow_pickle=True)
    # global i
    epoch_loss_abnormal_remove = np.array([])
    for i in range(epoch_loss_abnormal.shape[0]):
        if np.mean(epoch_loss_abnormal[i]) > 1000:
            break
        else:
            if not epoch_loss_abnormal_remove.any():
                epoch_loss_abnormal_remove = epoch_loss_abnormal[i]
            else:
                epoch_loss_abnormal_remove = np.vstack((epoch_loss_abnormal_remove, epoch_loss_abnormal))
    print(epoch_loss_abnormal_remove.shape)


    # loss_normal = np.mean(epoch_loss_normal, axis=0)
    # loss_abnormal = np.mean(epoch_loss_abnormal, axis=0)
    #
    # print(loss_abnormal.shape)
    # print(loss_normal.shape)
    # print(loss_normal)
    # print(loss_abnormal)



    # plt.plot(loss_normal)
    # plt.plot(loss_abnormal)
    # plt.show()

if __name__ == '__main__':
    calculate_normal()
    calculate_abnormal()
