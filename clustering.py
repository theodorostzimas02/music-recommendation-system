import pandas as pd
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


data = pd.read_csv(r"F:\Σχολή\Μουσική Πληροφορική\MRS\music.CSV", delimiter=',')
print (data.head())


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

data['Key_Label'] = data['Key'].map(key_mapping)
data['Timbre_Label'] = data['Timbre'].map(timbre_mapping)


preprocessed_data = data[['Tempo', 'Key_Label', 'Timbre_Label']]
print(preprocessed_data)


k = 5  
kmeans = KMeans(k)
clusters = kmeans.fit_predict(preprocessed_data)
print(clusters)



data['Cluster'] = clusters
newData = data[['Song Name', 'Artist', 'Tempo', 'Key', 'Timbre', 'Cluster']]


newData.to_csv('clustered_music_data.csv', index=False)
joblib.dump(kmeans, 'kmeans_model.pkl')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data['Tempo'], data['Key_Label'], data['Timbre_Label'], c=clusters, cmap='viridis')
ax.set_xlabel('Tempo')
ax.set_ylabel('Key_Label')
ax.set_zlabel('Timbre_Label')
ax.set_title('K-means Clustering of Songs')

plt.show()


for i, txt in enumerate(data['Song Name']):
    plt.text(data['Tempo'].iloc[i], data['Key_Label'].iloc[i], f"{txt} ({data['Timbre_Label'].iloc[i]})", fontsize=8)

plt.show()

