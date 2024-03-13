
import joblib
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import read_raw_data, read_dict_from_json
from torch.nn.utils.rnn import pad_sequence
from config import (
    UNK_IDX,
    PAD_IDX,
)
from utils import read_raw_data
from sklearn.preprocessing import PolynomialFeatures


def pad_list_collate_fn(batch, seq_len=8):
    trans_category_id_tensor, c_feature_tensor, n_feature_tensor, visit_category_idxs_tensor, target_category_id_tensor, target_category_idx_tensor, label_tensor = zip(*batch)

    c_feature_tensor = torch.stack(c_feature_tensor)
    n_feature_tensor = torch.stack(n_feature_tensor)
    
    # Pad sequences, make length to seq_len, if larger, truncate, if smaller, pad with 0
    visit_category_idxs_tensor = [visit_category_idxs_tensor[i][:seq_len] for i in range(len(visit_category_idxs_tensor))]
    visit_category_idxs_padded = pad_sequence(visit_category_idxs_tensor, batch_first=True, padding_value=PAD_IDX)
    trans_category_id_tensor = torch.stack(trans_category_id_tensor)
    
    target_category_id_tensor = torch.stack(target_category_id_tensor)
    target_category_idx_tensor = torch.stack(target_category_idx_tensor)
    labels_tensor = torch.tensor(label_tensor).long()
    return trans_category_id_tensor, c_feature_tensor, n_feature_tensor, \
            visit_category_idxs_padded, target_category_id_tensor, target_category_idx_tensor, labels_tensor    


class PageDataset(Dataset):
    def __init__(
        self, 
        df,
        category2index,
        numerical_col, 
        categorical_cols, 
        visit_category_id_col,
        target_category_id_col
        ) -> None:
        super().__init__()
        self.visit = df
        self.category2index = category2index
        self.numerical_cols = numerical_col
        self.categorical_cols = categorical_cols
        self.visit_category_id_col = visit_category_id_col
        self.target_category_id_col = target_category_id_col
        
    def get_idx_from_category(self, category_id: int):
        # return UNK_IDX if the category_id is unknown
        return self.category2index.get(category_id, UNK_IDX)
    
    def __len__(self):
        return len(self.visit)
    
    def __getitem__(self, idx):
        row = self.visit.iloc[idx]
        c_feature_tensor = torch.FloatTensor(row[self.categorical_cols].tolist())
        n_feature_tensor = torch.FloatTensor(row[self.numerical_cols].tolist())

        visit_category_idxs_tensor = torch.LongTensor([self.get_idx_from_category(int(cid)) for cid in row[self.visit_category_id_col]])
        # visit_category_idxs_tensor = torch.LongTensor(row[self.visit_category_id_col])
        
        target_category_id_tensor = torch.LongTensor([int(row[self.target_category_id_col])])
        target_category_idx_tensor = torch.LongTensor([self.get_idx_from_category(int(row[self.target_category_id_col]))])

        trans_category_id_tensor = torch.LongTensor([int(row['trans_category_id'])])
        label_tensor = torch.LongTensor([int(row['label'])])
        # return c_feature_tensor, n_feature_tensor, visit_category_idxs_tensor
        return trans_category_id_tensor, c_feature_tensor, n_feature_tensor, visit_category_idxs_tensor, target_category_id_tensor, target_category_idx_tensor, label_tensor
    

class WDNet(nn.Module):
    def __init__(
        self, 
        embeddings, 
        deep_input_size, 
        wide_input_size,
        dropout_rate=0.2
        ):
        super(WDNet, self).__init__()
        self.embeddings = embeddings    
        self.dropout = nn.Dropout(dropout_rate)
        self.deep_net = nn.Sequential(
            nn.Linear(deep_input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.wide_net = nn.Sequential(
            nn.Linear(wide_input_size, 1),
            nn.Sigmoid()
        )
        self.final_net = nn.Linear(2, 1)
        
    def forward(self, x):
        c_feature_tensor, n_feature_tensor, visit_category_idxs_tensor, target_category_idx_tensor = x
        cat_embeddings = self.embeddings(visit_category_idxs_tensor)
        cat_embeddings = cat_embeddings.sum(dim=1)
        
        target_embedding = self.embeddings(target_category_idx_tensor).squeeze(1)
        deep_input = torch.cat((cat_embeddings, target_embedding, n_feature_tensor), dim=1)
        
        deep_out = self.deep_net(deep_input)
        wide_out = self.wide_net(c_feature_tensor)
        out = torch.cat((deep_out, wide_out), dim=1)
        final_out = self.final_net(out).squeeze(1)
        return final_out
        