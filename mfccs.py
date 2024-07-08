from librosa import load    # Loads the MP3 files.
from librosa.feature import mfcc

x, sr = load('Debussy Arabesque No.1.mp3', duration=None) # Mono sound
mfccs = mfcc(y=x, sr=sr, n_mfcc=20)
print(mfccs)