import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import mode
import seaborn as sns
import os
import copy
from pylab import mpl
from more_itertools import chunked
import librosa
import scipy.io.wavfile as wavf

# xx = [0.3*i for i in range(75)]
# s = np.zeros(25).tolist() + (14*np.ones(25)).tolist() + np.zeros(25).tolist() + xx + (22.2*np.ones(25)).tolist() + np.zeros(15).tolist()
# noise = np.random.normal(0, 5, len(s))
# x_noise = np.zeros(II).tolist() + (np.array(s) + noise).tolist() + np.zeros(II).tolist()
# s = np.zeros(II).tolist() + s + np.zeros(II).tolist()
# s_noise = copy.deepcopy(x_noise)
# h = [i for i in range(10)]

test_signal, samplerate = librosa.load('D:/PhysioNet/ALL_副本/normal/b0025.wav', sr=2000, mono=False)
real_test_signal = copy.deepcopy(test_signal)


def median(ll):
    ll.sort()
    return ll[len(ll)//2]


def denosie_median(x, II=10):
    for k in range(II, len(x)-II):
        x[k] = median(x[k-II:k+II+1])
    return x


def denosie_CWFMH(x, II=10):
    w1, w2, w3, w4, w5 = 1, 1, 3, 1, 1  # CWFMH
    h = [i for i in range(II)]
    for i in range(1, II+1):
        h[i-1] = (4*II-6*i+2)/(II*(II-1))
    x = np.concatenate((np.zeros(II), x, np.zeros(II)), axis=0)
    for k in range(II, len(x)-II):
        y1, y2, y3, y4, y5 = 0, 0, 0, 0, 0
        for i in range(II):
            y1 = y1 + x[k-(i+1)]
            y2 = y2 + h[i]*x[k-(i+1)]
            y3 = x[k]
            y4 = y4 + x[k+(i+1)]
            y5 = y5 + h[i]*x[k+(i+1)]
        y1 = y1 / II
        y4 = y4 / II
        w = [w1, w2, w3, w4, w5]
        y = [y1, y2, y3, y4, y5]
        ll = (np.array(w)*np.array(y)).tolist()
        x[k] = median(ll)
    return x[II:-II]


def denosie_SWFMH(x, II=10):
    w1, w2, w3, w4, w5 = 2, 1, 1, 2, 1  # SWFMH
    h = [i for i in range(II)]
    for i in range(1, II+1):
        h[i-1] = (4*II-6*i+2)/(II*(II-1))
    x = np.concatenate((np.zeros(II), x, np.zeros(II)), axis=0)
    for k in range(II, len(x)-II):
        y1, y2, y3, y4, y5 = 0, 0, 0, 0, 0
        for i in range(II):
            y1 = y1 + x[k-(i+1)]
            y2 = y2 + h[i]*x[k-(i+1)]
            y3 = x[k]
            y4 = y4 + x[k+(i+1)]
            y5 = y5 + h[i]*x[k+(i+1)]
        y1 = y1 / II
        y4 = y4 / II
        w = [w1, w2, w3, w4, w5]
        y = [y1, y2, y3, y4, y5]
        ll = (np.array(w)*np.array(y)).tolist()
        x[k] = median(ll)
    return x[II:-II]


def awgn(x, snr, out='signal', method='vectorized', axis=0):

    # Signal power
    if method == 'vectorized':
        N = x.size
        Ps = np.sum(x ** 2 / N)

    elif method == 'max_en':
        N = x.shape[axis]
        Ps = np.max(np.sum(x ** 2 / N, axis=axis))

    elif method == 'axial':
        N = x.shape[axis]
        Ps = np.sum(x ** 2 / N, axis=axis)

    else:
        raise ValueError('method \"' + str(method) + '\" not recognized.')

    # Signal power, in dB
    Psdb = 10 * np.log10(Ps)

    # Noise level necessary
    Pn = Psdb - snr

    # Noise vector (or matrix)
    n = np.sqrt(10 ** (Pn / 10)) * np.random.normal(0, 1, x.shape)

    if out == 'signal':
        return x + n
    elif out == 'noise':
        return n
    elif out == 'both':
        return x + n, n
    else:
        return x + n

# s_denoise = denosie_SWFMH(test_signal)
# # s_denoise = denosie_median(x_noise)
# plt.plot(real_test_signal, label='signal')
# # plt.plot(s_noise, label='signal_noise')
# plt.plot(s_denoise, label='signal_denoise')
# plt.legend()
# plt.show()
# wavf.write('D:/PhysioNet/ALL_副本/normal/b0025_denoise_sw.wav', samplerate, s_denoise)
#
# print('good luck')
