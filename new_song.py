import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load the trained KMeans model
kmeans = KMeans()
kmeans = joblib.load(r"F:\Σχολή\Μουσική Πληροφορική\MRS\kmeans_model.pkl")  # Adjust filename/path as per your actual saved model

# Load the clustered music data
clustered_data = pd.read_csv('clustered_music_data.csv')

key_mapping = {'C Major': 1, 'C Sharp Major': 2, 'D Flat Major': 2, 'D Major': 3, 'D Sharp Major': 4, 'E Flat Major': 4, 'E Major': 5, 'F Major': 6, 'F Sharp Major': 7, 'G Flat Major': 7, 'G Major': 8, 'G Sharp Major': 9, 'A Flat Major': 9, 'A Major': 10, 'A Sharp Major': 11, 'B Flat Major': 11, 'B Major': 12, 'C Minor': 13, 'C Sharp Minor': 14, 'D Flat Minor': 14, 'D Minor': 15, 'D Sharp Minor': 16, 'E Flat Minor': 16, 'E Minor': 17, 'F Minor': 18, 'F Sharp Minor': 19, 'G Flat Minor': 19, 'G Minor': 20, 'G Sharp Minor': 21, 'A Flat Minor': 21, 'A Minor': 22, 'A Sharp Minor': 23, 'B Flat Minor': 23, 'B Minor': 24}
timbre_mapping = {
    'Upbeat': 1,
    'Energetic': 2,
    'Dynamic': 3,
    'Soothing': 4,
    'Raw': 5,
    'Melancholic': 6,
    'Edgy': 7,
    'Reflective': 8,
    'Mellow': 9,
    'Empowering': 10,
    'Calm': 11,
    'Dramatic': 12,
    'Graceful': 13,
    'Romantic': 14,
    'Ethereal': 15,
    'Heartfelt': 16,
    'Sacred': 17,
    'Powerful': 18,
    'Feel-Good': 19,
    'Bluesy': 20,
    'Folksy': 21,
    'Lively': 22,
    'Soulful': 23,
    'Fierce': 24
}


# Example of a new song (you can replace this with actual data)
new_song = {
    'Song Name': 'New Song',
    'Tempo': 190,
    'Key': 'B Major',
    'Timbre': 'Energetic'
}

# Map key and timbre to labels using predefined mappings
new_song['Key_Label'] = key_mapping[new_song['Key']]
new_song['Timbre_Label'] = timbre_mapping[new_song['Timbre']]

# Prepare new song data for prediction
new_song_data = pd.DataFrame([new_song])

# Extract features for clustering
new_song_features = new_song_data[['Tempo', 'Key_Label', 'Timbre_Label']]

# Predict the cluster for the new song
new_song_cluster = kmeans.predict(new_song_features)[0]
# Find closest neighbors within the same cluster
neighbors = clustered_data[clustered_data['Cluster'] == new_song_cluster]

# Assign cluster label to the new song
new_song_data['Cluster'] = new_song_cluster

# Optionally, you can append the new song to the existing clustered data
updated_clustered_data = pd.concat([clustered_data, new_song_data], ignore_index=True)

# Save the updated data to a new CSV file
updated_clustered_data.to_csv('updated_clustered_music_data.csv', index=False)

print(f"New song '{new_song['Song Name']}' assigned to cluster {new_song_cluster}.")

# Display the closest neighbors
print("Closest neighbors:")
for index, neighbor in neighbors.head(5).iterrows():
    print(neighbor['Song Name'], "by", neighbor['Artist'])

