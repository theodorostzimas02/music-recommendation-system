from librosa import load    # Loads the MP3 files.
from librosa.feature import fourier_tempogram, tempogram, tempo
from scipy.signal import spectrogram
from scipy.signal.windows import hann
from librosa.display import specshow
import numpy as np
from math import log10
import matplotlib.pyplot as plt

def Plotting(x, title="Title", ylabel="ylabel", xlabel="xlabel"):
    plt.figure()
    plt.plot(x)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

def local_average_normalization(x, N=0):
    # N is the number of samples per local area.
    if N == 0: N = int(len(x)/6)

    # Number of local areas
    local_areas = int(len(x)/N)
        
    for i in range(local_areas):    # For every local area
        temp_sum = 0
        avg = 0
        for j in range(i*N, (i+1)*N):
            temp_sum += x[j]

        avg = temp_sum/N    # Compute the mean of the local area
        for k in range(i*N, (i+1)*N):
            x[k] = max(0, x[k] - avg)   # And subtract it from all of the values.

    return x

def novelty_energy(x):
    Sxx = x**2  # Amplitude squaring.

    # Obtaining energy envelope using a hann window.
    w = hann(2048)
    Sxx = np.convolve(Sxx, w, 'same')

    # Differentiation and Half wave rectification.
    for i in range(len(x)-1):
        Sxx[i] = max(0, Sxx[i+1] - Sxx[i])

    return x

def novelty_spectral(x, sr, C=10**8):
    f, t, Sxx = spectrogram(x=x, fs=sr) # Spectrogram
    # print(Sxx)

    # Logarithmic Compression.
    for i in range(len(Sxx)):
        for j in range(len(Sxx[i])):
            Sxx[i][j] = log10(1 + C*abs(Sxx[i][j]))

    # Differentiation and Half wave rectification.
    for i in range(len(Sxx)):   # For every frequency bin.
        for j in range(len(Sxx[i])-1):
            Sxx[i][j] = max(0, Sxx[i][j+1] - Sxx[i][j])

    # Accumulation.
    Sxx = np.sum(Sxx, axis=0)
    Sxx = np.concatenate((Sxx, np.array([0])))

    # Normalization by subtracting the local average.
    Sxx = local_average_normalization(Sxx)

    return Sxx

x, sr = load('Beethoven - Moonlight Sonata (1st Movement).mp3', duration=4.5) # Mono sound

nov0 = novelty_energy(x)
nov1 = novelty_spectral(x, sr)

Tempo0 = tempogram(onset_envelope=nov0, sr=sr)    #Looks wrong.
Tempo1 = tempogram(onset_envelope=nov1, sr=sr)
Tempo2 = fourier_tempogram(onset_envelope=nov0, sr=sr)    #Looks wrong.
Tempo3 = fourier_tempogram(onset_envelope=nov1, sr=sr)

specshow(data=Tempo0, sr=sr, cmap='magma')  # Takes time.
plt.show()

specshow(data=Tempo1, sr=sr, cmap='magma')
plt.show()

specshow(data=abs(Tempo2), sr=sr, cmap='magma') # Takes time.
plt.show()

specshow(data=abs(Tempo3), sr=sr, cmap='magma')
plt.show()

# tempo0 = tempo(onset_envelope=nov0, sr=sr) # Error
tempo1 = tempo(onset_envelope=nov1, sr=sr)

print(tempo1)