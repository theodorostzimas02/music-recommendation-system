import librosa
import numpy as np
import os
import demucs.separate
import matplotlib.pyplot as plt
import soundfile as sf
import scipy as sp
from distances import *

def preprocess_audio(audio_path):
    # Load the audio file
    audio, sr = librosa.load(audio_path)

    # Convert stereo to mono
    audio_mono = librosa.to_mono(audio)

    # Normalize the audio
    audio_mono = audio_mono - np.mean(audio_mono)
    audio_mono = audio_mono / np.max(np.abs(audio_mono))
    audio_mono = np.hanning(len(audio_mono)) * audio_mono

    return audio_mono, sr

template = np.array(["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"])

song_path = r"C:\Users\theot\Music\Zotify Music\MRS -  ΕΠ21 Spring Semester 2024\Weezer - Buddy Holly.mp3"
output_dir = song_path.replace(".mp3", "")
song_name = os.path.basename(song_path).replace(".mp3", "")

vocal_path = output_dir + "/htdemucs/" + song_name + "/vocals.mp3"
instrumental_path = output_dir + "/htdemucs/" + song_name + "/no_vocals.mp3"

print(vocal_path)
print(instrumental_path)

if not os.path.exists(vocal_path) or not os.path.exists(instrumental_path):
    demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "htdemucs", "-o", output_dir, song_path])

# Compute the chroma features for the vocals
vocals_mono, sr = preprocess_audio(vocal_path)
instrumental_mono, sr = preprocess_audio(instrumental_path)

sf.write("test.mp3", vocals_mono, sr)

# Compute the chroma features
vocals_chroma = librosa.feature.chroma_stft(y=vocals_mono, sr=sr, tuning=0, norm=2, hop_length=512, n_fft=2048)

# Krumhansl - Schmuckler coefficients
major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def autocorrelation(f, W, t, lag):
    print("Calculating autocorrelation...")
    return np.sum(f[t:t+W] *  f[t+lag:t+W+lag])

def DF(f, W, t, max_lag):
    print("Calculating DF...")
    return autocorrelation(f, W, t, 0) + autocorrelation(f, W, t, max_lag) - (2 * autocorrelation(f, W, t, max_lag))

def CMNDF(f, W, t, max_lag):
    print("Calculating CMNDF...")
    if max_lag == 0:
        return 0
    return DF(f, W, t, max_lag) / np.sum([DF(f, W, t, i) for i in range(max_lag)]) * max_lag

def yin_algorithm(f, W, t, sample_rate, bounds):
    print("Running YIN algorithm...")
    CMNDF_vals = [CMNDF(f, W, t, i) for i in range(*bounds)]
    sample = np.argmin(CMNDF_vals) + bounds[0]
    return sample_rate / sample

def calculate_distance(pitch, key, method):
    if method == "Euclidean":
        return Euclidean(pitch, key)
    elif method == "Manhattan":
        return Manhattan(pitch, key)
    elif method == "Cosine":
        return Cosine_Similarity(pitch, key)
    else:
        return 0

def estimate_key(audio, sr, template, template_names):
    print("Estimating key...")
    # Estimate the pitch using YIN algorithm
    window_size = 2048
    hop_size = 512
    pitches = []
    num_of_samples = int (sr * (60-0) / + 1)
    bounds = [20, num_of_samples//2 ]
    pitch = yin_algorithm(audio, window_size, 0, sr, bounds)
    print(pitch)
    
    # # Aggregate pitch to chroma
    # plt.figure(3)
    # chroma = librosa.feature.chroma_stft(y=audio, sr=sr, tuning=0, norm=2, hop_length=512, n_fft=2048)
    # librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Chromagram for Instrumental')
    # plt.show()
    
    # # Calculate distances between estimated pitch chroma and each key in the template
    # distances = []
    # for key in template:
    #     # Calculate distance between pitch chroma and current key
    #     distance = calculate_distance(chroma, key, "Euclidean")
    #     distances.append(distance)
    
    # # Find the key with the minimum distance
    # min_distance_index = np.argmin(distances)
    # return template_names[min_distance_index]
    return pitch



# def find_bpm(audio, sr):
#     # Step 1: Amplitude squaring
#     audio_squared = audio ** 2

#     # Step 2: Windowing
#     window_size = int(0.1 * sr)  # 100ms window
#     window = np.hanning(window_size)
#     windowed_signal = np.convolve(audio_squared, window, mode='same')

#     # Step 3: Differentiation
#     diff_signal = np.diff(windowed_signal)

#     # Step 4: Half wave rectification
#     half_wave_rectified = np.maximum(0, diff_signal)

#     # Step 5: Peak picking
#     peaks, _ = librosa.util.peak_pick(half_wave_rectified, pre_max=sr//2, post_max=sr//2, pre_avg=sr//2, post_avg=sr//2, delta=0.01, wait=0)

#     # Estimate BPM
#     peak_intervals = np.diff(peaks) / sr  # Time between peaks in seconds
#     bpm = 60.0 / np.mean(peak_intervals)  # Convert to BPM

#     return bpm

# Compute the chroma features for the instrumental
# instrumental_chroma = librosa.feature.chroma_stft(y=instrumental_mono, sr=sr, tuning=0, norm=2, hop_length=512, n_fft=2048)

# # # Estimate the keys for vocals and instrumental
# key_major_vocals = estimate_key(vocals_mono, sr, major_template, template)
# key_minor_vocals = estimate_key(vocals_mono, sr, minor_template, template)
# key_major_instrumental = estimate_key(instrumental_mono, sr, major_template, template)
# key_minor_instrumental = estimate_key(instrumental_mono, sr, minor_template, template)

# print("The key of the song (vocals, major):", key_major_vocals)
# print("The key of the song (vocals, minor):", key_minor_vocals)
# print("The key of the song (instrumental, major):", key_major_instrumental)
# print("The key of the song (instrumental, minor):", key_minor_instrumental)
# print("The BPM of the song is:", find_bpm(instrumental_mono, sr))

# # Plot the chromagram for vocals
# plt.figure(2)
# librosa.display.specshow(vocals_chroma, y_axis='chroma', x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Chromagram for Vocals')
# plt.show()

# # Plot the chromagram for instrumental
# plt.figure(3)
# librosa.display.specshow(instrumental_chroma, y_axis='chroma', x_axis='time')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Chromagram for Instrumental')
# plt.show()

instrumental_chroma = librosa.feature.chroma_stft(y=instrumental_mono, sr=sr, tuning=0, norm=2, hop_length=512, n_fft=2048)
chroma_mean = np.mean(instrumental_chroma, axis=1)
correlations = [np.corrcoef(chroma_mean, key_profile)[0, 1] for key_profile in key_profiles]
print(chroma_mean)
