import tensorflow as tf
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from sys import stderr
import argparse

parser = argparse.ArgumentParser(description='Style transfer')
parser.add_argument('--content', help='Original audio path', required=True)
parser.add_argument('--style', help='Style audio path', required=True)
parser.add_argument('--out', help='Output audio path', required=True)
args = parser.parse_args()
N_FFT = 2048


def read_audio_spectum(filename):
    x, fs = librosa.load(filename, duration=5.0)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)

    S = np.log1p(np.abs(S[:, :430]))
    return S, fs


a_content, fs = read_audio_spectum(args.content)
a_style, fs = read_audio_spectum(args.style)
r_style, fs = read_audio_spectum(args.out)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('First')
plt.imshow(a_content[:400, :])
plt.subplot(1, 3, 2)
plt.title('Second ')
plt.imshow(a_style[:400, :])
plt.subplot(1, 3, 3)
plt.title('Result')
plt.imshow(r_style[:400, :])
plt.show()
