import librosa
import numpy as np
import os
import demucs.separate
import matplotlib.pyplot as plt
import soundfile as sf
import scipy as sp
from distances import *
import glob
import pandas as pd

general_path = os.getcwd()

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

# Krumhansl - Schmuckler coefficients
major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def seperation (song_path):
    output_dir = song_path.replace(".mp3", "")
    song_name = os.path.basename(song_path).replace(".mp3", "")

    vocal_path = output_dir + "/htdemucs/" + song_name + "/vocals.mp3"
    instrumental_path = output_dir + "/htdemucs/" + song_name + "/no_vocals.mp3"
    if not os.path.exists(vocal_path) or not os.path.exists(instrumental_path):
        demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "htdemucs", "-o", output_dir, song_path])
    
    return vocal_path, instrumental_path



def key_from_chroma(chroma_distribution, major_template, minor_template):
    # Standardize the chroma distribution
    mean_np = np.mean(chroma_distribution)
    standard_deviation = np.std(chroma_distribution)
    z_scores = (chroma_distribution - mean_np) / standard_deviation

    print("Z scores:", z_scores)
    
    # Standardize the templates
    major_z = (major_template - np.mean(major_template)) / np.std(major_template)
    minor_z = (minor_template - np.mean(minor_template)) / np.std(minor_template)

    print ("Major Z:", major_z)
    print ("Minor Z:", minor_z)

    max_corr_major = float('-inf')
    max_corr_minor = float('-inf')
    best_major_key = None
    best_minor_key = None
    
    for i in range(12):
        rotated_major = np.roll(major_z, i)
        rotated_minor = np.roll(minor_z, i)
        
        corr_major = np.dot(z_scores, rotated_major)
        print ("Correlation major:", corr_major)
        corr_minor = np.dot(z_scores, rotated_minor)
        print ("Correlation minor:", corr_minor)
        
        if corr_major > max_corr_major:
            max_corr_major = corr_major
            best_major_key = i
            print ("Best major key:", best_major_key)
        
        if corr_minor > max_corr_minor:
            max_corr_minor = corr_minor
            best_minor_key = i
            print ("Best minor key:", best_minor_key)

        key_major = template[best_major_key]
        key_minor = template[best_minor_key]

    return key_major, key_minor, max_corr_major, max_corr_minor


def write_key_for_each_song(songs):
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['Name of Song', 'Artist', 'Major Key', 'Minor Key'])
    
    for song in songs:
        song_name, artist_name, vocal_path, instrumental_path = song
        vocal_mono, sr = preprocess_audio(vocal_path)
        instrumental_mono, sr = preprocess_audio(instrumental_path)

        vocal_chroma = librosa.feature.chroma_stft(y=vocal_mono, sr=sr)
        chroma_mean = np.sum(vocal_chroma, axis=1)
        key_major, key_minor, max_corr_major, max_corr_minor = key_from_chroma(chroma_mean, major_template, minor_template)
        
        # Assuming song is a tuple containing (song_name, artist_name, vocal_path, instrumental_path)
        # You need to adjust this based on your actual data structure
        
        new_row = pd.DataFrame([{'Name of Song': song_name, 'Artist': artist_name, 'Major Key': key_major, 'Minor Key': key_minor}])
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Write DataFrame to CSV
    os.chdir(general_path)
    df.to_csv('song_keys.csv', index=False)

song_path = r"C:\Users\theot\Music\Zotify Music\MRS -  ΕΠ21 Spring Semester 2024\The Story So Far - Empty Space.mp3"


dir_path = r"C:\Users\theot\Music\Zotify Music\MRS -  ΕΠ21 Spring Semester 2024"
os.chdir(dir_path)

songs = []
for file in glob.glob("*.mp3"):
    print(file)
    base_name = os.path.basename(file) 
    name_without_extension = os.path.splitext(base_name)[0] 
    artist, song_name = name_without_extension.split(' - ', 1)
    vocal_path, instrumental_path = seperation(file)
    songs.append((song_name, artist, vocal_path, instrumental_path))

write_key_for_each_song(songs)







# Placeholder for distance calculation function
def calculate_distance(pitch, key, method):
    if method == "Euclidean":
        return Euclidean(pitch, key)
    elif method == "Manhattan":
        return Manhattan(pitch, key)
    elif method == "Cosine":
        return Cosine_Similarity(pitch, key)
    else:
        return 0 
    


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



