import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

class WineDataset(Dataset):
    def __init__(self, df):
        '''
            csv_data: pandas dataframe consisting of all 13 columns in the Train.csv, Test.csv
        '''
        self.df = df
        self.df = pd.get_dummies(df, columns=['variety'])
        self.df = self.df.fillna(self.df.mean())
        quality_values = sorted(self.df['quality'].unique())
        self.quality_to_index = {quality: idx for idx, quality in enumerate(quality_values)}
        self.X = np.array(self.df.drop('quality', axis=1).values, dtype=np.float64)
        self.y = np.array(self.df['quality'].map(self.quality_to_index).values, dtype=np.float64)

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        self.X_tensor = torch.from_numpy(self.X).type(torch.FloatTensor)
        self.y_tensor = torch.from_numpy(self.y).type(torch.LongTensor)
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        X_tensor = torch.from_numpy(self.X).type(torch.FloatTensor)
        y_tensor = torch.from_numpy(self.y).type(torch.LongTensor)
        return self.X_tensor[idx], self.y_tensor[idx]

def create_dataset(csv_data, test_size=0.2):
    df = pd.read_csv(csv_data)
    # classes = 7
    
    train_df, val_df = train_test_split(df, test_size = test_size, random_state=42)
    train_dataset = WineDataset(train_df)
    val_dataset = WineDataset(val_df)
    return train_dataset, val_dataset

