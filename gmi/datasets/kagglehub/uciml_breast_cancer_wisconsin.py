import kagglehub
import torch
import torchvision
import pandas as pd
import os
import numpy as np

from .base import KaggleHubDataset

kagglehub.dataset_load

class UCIMLBreastCancerWisconsinDataset(KaggleHubDataset):
    def __init__(self, split='train', seed=42, download=False, drop_columns=None):
        assert split in ['train', 'val', 'test', 'all']
        self.split = split
        name = 'uciml/breast-cancer-wisconsin-data'
        super(UCIMLBreastCancerWisconsinDataset, self).__init__(name, download)
        self.data = self._load_data(drop_columns=drop_columns)
    def _load_data(self, drop_columns=None):
        # get the only csv file in the directory
        for file in os.listdir(self.path):
            if file.endswith(".csv"):
                data_path = os.path.join(self.path, file)
                break
        df = pd.read_csv(data_path)
        # drop the id column and diagnosis column
        # df = df.drop(columns=["Unnamed: 32", "diagnosis", "id"])

        if drop_columns is not None:
            df = df.drop(columns=drop_columns)
        # split randomly int 0.70 training, 0.15 validation, 0.15 test
        ordered_indices = np.arange(len(df))
        np.random.seed(42)
        np.random.shuffle(ordered_indices)
        train_indices = ordered_indices[:int(0.7 * len(df))]
        val_indices = ordered_indices[int(0.7 * len(df)):int(0.85 * len(df))]
        test_indices = ordered_indices[int(0.85 * len(df)):]
        if self.split == 'train':
            split_data = df.iloc[train_indices]
        elif self.split == 'val':
            split_data = df.iloc[val_indices]
        elif self.split == 'test':
            split_data = df.iloc[test_indices]
        else:
            split_data = df

        self.df = split_data.copy()

        self.codebook = {}
        # if there are any columns that are strings, make a codebook and convert to integers
        for col in split_data.columns:
            if split_data[col].dtype == 'object':
                self.codebook[col] = {val: i for i, val in enumerate(split_data[col].unique())}
                split_data[col] = split_data[col].map(self.codebook[col])
                
        # convert to torch tensor
        split_tensor = torch.tensor(split_data.values, dtype=torch.float32)
        return split_tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.data[idx],)
    
        



# path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
# print(f"Dataset downloaded to: {path}")
# # get the only csv file in the directory
# for file in os.listdir(path):
#     if file.endswith(".csv"):
#         data_path = os.path.join(path, file)
#         break
# print(f"Data file path: {data_path}")
# import pandas as pd
# data = pd.read_csv(data_path)
# print(data.head())
# print(data.columns)
# print(f"Number of rows: {len(data)}")
# print(f"Number of columns: {len(data.columns)}")
# # save the data to the example directory
# data.to_csv(os.path.join(example_dir, "breast_cancer_data.csv"), index=False)
# print(f"Data saved to: {os.path.join(example_dir, 'breast_cancer_data.csv')}")
# # make a seaborn pairplot of mean radius, mean texture, mean perimeter, mean area, mean smoothness
# # id	diagnosis	radius_mean	texture_mean	perimeter_mean	area_mean	smoothness_mean	compactness_mean	concavity_mean	concave points_mean	symmetry_mean	fractal_dimension_mean	radius_se	texture_se	perimeter_se	area_se	smoothness_se	compactness_se	concavity_se	concave points_se	symmetry_se	fractal_dimension_se	radius_worst	texture_worst	perimeter_worst	area_worst	smoothness_worst	compactness_worst	concavity_worst	concave points_worst	symmetry_worst	fractal_dimension_worst
# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.pairplot(data[["area_mean", "texture_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean"]])
# plt.savefig(os.path.join(example_dir, "breast_cancer_pairplot.png"))
# plt.close()


# # scatterplot of symmetry vs smoothness
# sns.scatterplot(data=data, x="symmetry_mean", y="smoothness_mean")
# plt.savefig(os.path.join(example_dir, "breast_cancer_scatterplot.png"))
# plt.close()