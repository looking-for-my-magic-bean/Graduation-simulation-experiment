import sys

import librosa
import numpy as np
import os
from python_speech_features import *
import pylab as pl
from matplotlib import pyplot
import matplotlib.pyplot as plt
from pydub import AudioSegment

group = 'abnormal'
input_wav_path = 'D:/Michigan_training/%s/old/' % group
output_wav_path = 'D:/Michigan_training/%s/' % group
everyone_list = os.listdir(input_wav_path)
# everyone_list = ['01_normal.mp3', '14_normal.mp3']
k = 0


def get_ms_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    start_time = int(start_time)
    end_time = int(end_time)

    sound = AudioSegment.from_mp3(main_wav_path)
    word = sound[start_time:end_time]

    word.export(part_wav_path, format="wav")


for everyone in everyone_list:
    test_signal, samplerate = librosa.load(input_wav_path + everyone, sr=2000, mono=False)
    if test_signal.shape[0] == 2:
        test_signal = test_signal[0]
    epoch = len(test_signal) // (16*1000)
    for i in range(epoch):
        get_ms_part_wav(input_wav_path + everyone, 0 + 8*1000 * i, 8*1000 * (i + 1),
                        output_wav_path + '000%d_%s.wav' % (k, group))
        print(k)
        k = k + 1


print('Good Luck')
