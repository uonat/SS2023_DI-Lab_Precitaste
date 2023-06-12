import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm.notebook import tqdm

class KNNClassifier:
    def __init__(self, n_neighboors=5, similarity_metric='cosine'):
        allowed_metrics = ['cityblock', 'cosine', 'l1', 'l2', 'euclidean', 'haversine', 'manhattan', 'nan_euclidean']
        if similarity_metric not in allowed_metrics:
             raise ValueError("Given similarity metric is not supported. Please provide one of: {}".format(allowed_metrics))
        self.metric = similarity_metric
        self.n_neigh = n_neighboors
        
        self.knn_model = KNeighborsClassifier(n_neighbors=n_neighboors, metric=similarity_metric)
    
    def load_and_fit_RPC(self, rpc_train_feats_path, n_sample=-1, allowed_folder_names=None):
        folder_names = os.listdir(rpc_train_feats_path)

        if allowed_folder_names is not None:
            folder_names = [fname for fname in folder_names if fname in allowed_folder_names]

        self.class_to_label = {folder_names[i]:i for i in range(len(folder_names))}
        self.label_to_class = {i:folder_names[i] for i in range(len(folder_names))}

        xs = []
        ys = []
        
        for folder_name in tqdm(folder_names):
            folder_label = self.class_to_label[folder_name]
            folder_path = os.path.join(rpc_train_feats_path, folder_name)
            feature_paths = [os.path.join(rpc_train_feats_path, folder_name, fname) for fname in os.listdir(folder_path) if '.npy' in fname]
            
            if n_sample != -1:
                sampled_feature_paths = np.random.choice(feature_paths, n_sample, replace=False).tolist()
            else:
                sampled_feature_paths = feature_paths.copy()
            
            folder_features = []
            for feature_file_path in sampled_feature_paths:
                with open(feature_file_path, 'rb') as npfile:
                    np_feat = np.load(npfile, allow_pickle=True)
                    folder_features.append(np_feat)
        
            xs = xs + folder_features
            ys = ys + [folder_label] * len(folder_features)
            
        self.X = np.concatenate(xs)
        self.Y = np.array(ys)
        
        self.knn_model.fit(self.X, self.Y)
        print("Loadded feature vector with shape: {}, label array with shape: {}".format(self.X.shape, self.Y.shape))
    
    def fit_samples(self, X, Y):
        print("Fitting feature vector with shape: {}, label array with shape: {}".format(X.shape, Y.shape))
        self.X = X
        self.Y = Y
        self.knn_model.fit(X, Y)

    def predict_sample(self, sample_features):
        return self.knn_model.predict(sample_features)
    
    def get_RPC_class_dict(self):
        return self.class_to_label
    
    def get_neighbors(self, x):
        """
        Returns distance to each k neighbors and their indices
        """
        return self.knn_model.kneighbors(x, return_distance=True)
    
    def get_mean_dist_neighbors(self, x, selected_labels=None):
        """
        Returns mean distance of each sample to all of its neighbors.
        If class label is provided returns the distance to the neighbors 
        of that specific class only
        """
        distances, neighbor_indices = self.get_neighbors(x)
        if selected_labels is None:
            return np.mean(distances,  axis=-1)
        
        # A function to get label of each neighbor from their indices
        get_label_of_index = lambda x: self.Y[x]
        labels_of_neighbor = get_label_of_index(neighbor_indices)

        # Mask that selects neighbors of each item in row by their given selected label
        label_mask = labels_of_neighbor == selected_labels[:, np.newaxis]

        # Calculate mean of non zero elements which will be the neighbors belong to selected classes
        sum_distances = np.sum(distances * label_mask, axis=-1)
        count_nonzero_elements = np.sum((distances * label_mask) != 0, axis=-1)
        mean_distances = sum_distances / count_nonzero_elements

        return np.where(np.isnan(mean_distances), -1, mean_distances)        