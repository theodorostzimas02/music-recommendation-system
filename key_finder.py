import librosa
import numpy as np
import os
import demucs.separate
import matplotlib.pyplot as plt
import soundfile as sf

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



song_path = r"C:\Users\theot\Music\Zotify Music\MRS -  ΕΠ21 Spring Semester 2024\Turnstile - Gravity.mp3"
output_dir = song_path.replace(".mp3", "")
song_name = os.path.basename(song_path).replace(".mp3", "")
song, sr = sf.read(song_path)

vocal_path = output_dir + "/htdemucs/" + song_name + "/vocals.mp3"
instrumental_path = output_dir + "/htdemucs/" + song_name + "/no_vocals.mp3"

print(vocal_path)
print(instrumental_path)


if not os.path.exists(vocal_path) or not os.path.exists(instrumental_path):
    demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "htdemucs", "-o", output_dir, song_path])

# Compute the chroma features for the vocals
vocals_mono , sr = preprocess_audio(vocal_path)
instrumental_mono , sr = preprocess_audio(instrumental_path)

sf.write("test.mp3", vocals_mono, sr)

# Compute the chroma features
vocals_chroma = librosa.feature.chroma_stft(y=vocals_mono, sr=sr, tuning=0, norm=2, hop_length=512, n_fft=2048)
#Krumhansl - Schmuckler coefficients
major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# # Compute the key
# chroma = np.sum(vocals_chroma, axis=1)
# major_correlation = np.correlate(chroma, major_template)
# minor_correlation = np.correlate(chroma, minor_template)
# key_index_major = np.argmax([major_correlation])
# key_index_minor = np.argmax([minor_correlation])

# key_major= template[key_index_major]
# key_minor= template[key_index_minor]
# print("The key of the song major is:", key_major)
# print("The key of the song minor is:", key_minor)

def estimate_key(chroma, template, template_names):
    chroma = np.sum(chroma, axis=1)
    correlations = np.array([np.correlate(chroma, np.roll(template, i))[0] for i in range(len(template))])
    key_index = np.argmax(correlations)
    return template_names[key_index]




# Compute the chroma features for the instrumental
instrumental_chroma = librosa.feature.chroma_stft(y=instrumental_mono, sr=sr, tuning=0, norm=2, hop_length=512, n_fft=2048)
# # Compute the key
# chroma = np.sum(instrumental_chroma, axis=1)
# major_correlation = np.correlate(chroma, major_template)
# minor_correlation = np.correlate(chroma, minor_template)
# key_index_major = np.argmax([major_correlation])
# key_index_minor = np.argmax([minor_correlation])

# key_major= template[key_index_major]
# key_minor= template[key_index_minor]
# print("The key of the song major is:", key_major)
# print("The key of the song minor is:", key_minor)

# Estimate the keys for vocals and instrumental
key_major_vocals = estimate_key(vocals_chroma, major_template, template)
key_minor_vocals = estimate_key(vocals_chroma, minor_template, template)
key_major_instrumental = estimate_key(instrumental_chroma, major_template, template)
key_minor_instrumental = estimate_key(instrumental_chroma, minor_template, template)

print("The key of the song (vocals, major):", key_major_vocals)
print("The key of the song (vocals, minor):", key_minor_vocals)
print("The key of the song (instrumental, major):", key_major_instrumental)
print("The key of the song (instrumental, minor):", key_minor_instrumental)


def autocorrelation(x, max_lag):
    N = len(x)
    autocorr = np.zeros(max_lag)
    for m in range(max_lag):
        autocorr[m] = np.sum(x[:N-m] * x[m:N])
    return autocorr / N


# Compute the autocorrelation for the separated vocals
max_lag = 100  # Define a reasonable maximum lag
vocals_autocorr = autocorrelation(vocals_mono, max_lag)
# Plot the autocorrelation for vocals


# Plot the chromagram for vocals
plt.figure(2)
librosa.display.specshow(vocals_chroma, y_axis='chroma', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Chromagram for Vocals')
plt.show()

# Plot the chromagram for instrumental
plt.figure(3)
librosa.display.specshow(instrumental_chroma, y_axis='chroma', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Chromagram for Instrumental')
plt.show()

plt.figure(4)
plt.plot(vocals_autocorr)
plt.title('Autocorrelation of Separated Vocals')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.tight_layout()
plt.show()

