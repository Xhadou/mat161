import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances

class Eigenfaces:

    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components, whiten=True)

    def train(self, training_data):
        self.pca.fit(training_data)
        self.eigenfaces = self.pca.components_
        self.projected_training_faces = self.pca.transform(training_data)

    def project(self, face):
        return self.pca.transform(face.reshape(1, -1))

    def recognize(self, projected_face, threshold=None):
        distances = euclidean_distances(self.projected_training_faces, projected_face)
        min_distance = np.min(distances)
        closest_face_idx = np.argmin(distances)

        if threshold is not None and min_distance > threshold:
            return None, min_distance

        return closest_face_idx, min_distance
