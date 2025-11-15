import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

class SweatFeatureExtraction:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def load_data(self):
        data = {}
        for sub_dir in ['AA', 'UA']:
            path = os.path.join(self.data_directory, sub_dir)
            data[sub_dir] = self._load_subdirectory(path)
        return data

    def _load_subdirectory(self, path):
        data = []
        for level in range(1, 6):  # Assume L1 to L5 
            level_path = os.path.join(path, f'L{level}')
            for file in os.listdir(level_path):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(level_path, file))
                    data.append(df)
        return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

    def extract_features(self, dataframe):
        pca = PCA(n_components=2)  # Adjust the number of components as needed
        features = pca.fit_transform(dataframe.values)
        return features

    def run(self):
        data = self.load_data()
        for key in data:
            features = self.extract_features(data[key])
            print(f'Features extracted for {key}: {features.shape}')

if __name__ == '__main__':
    data_dir = 'data'  # Update this path as needed
    extractor = SweatFeatureExtraction(data_dir)
    extractor.run()