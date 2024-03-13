
# %%
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from utils import haversine_tensor


def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)


class UserTower(nn.Module):
    def __init__(
            self, 
            embedding,
            wide_input_dim, 
            wide_output_dim, 
            deep_input_dim,
            deep_hidden_dims,
            deep_output_dim,
            p=0.1
        ):
        super(UserTower, self).__init__()
        self.embeddings = embedding
        self.wide_input_dim = wide_input_dim
        self.wide_output_dim = wide_output_dim
        self.dropout = nn.Dropout(p)
        
        # process the categorical (wide) features
        self.wide = nn.Linear(wide_input_dim, wide_output_dim)
        
        # process the numerical and embedding (deep) features
        deep_layers = []
        input_dims = [deep_input_dim] + deep_hidden_dims
        for i in range(len(deep_hidden_dims)):
            deep_layers.append(nn.Linear(input_dims[i], input_dims[i+1]))
            deep_layers.append(nn.BatchNorm1d(input_dims[i+1]))
            deep_layers.append(nn.ReLU())
        deep_layers.append(nn.Linear(deep_hidden_dims[-1], deep_output_dim))
        self.deep = nn.Sequential(*deep_layers)
        self.apply(init_weights)
    
    @staticmethod
    def get_distance(source_lat, source_lon, target_lat, target_lon):
        return haversine_tensor(source_lat, source_lon, target_lat, target_lon)
        
    def forward(self, c_feature, n_feature, visit_category_idxs):
        visit_features = self.embeddings(visit_category_idxs)
        visit_features = torch.sum(visit_features, dim=1)
        
        wide_input = torch.cat([c_feature, n_feature], 1)
        deep_input = torch.cat([n_feature, visit_features], 1)
        
        wide_output = self.wide(wide_input)
        deep_output = self.deep(deep_input)
        combined_output = torch.cat([wide_output, deep_output], 1)
        return combined_output


class CategoryTower(nn.Module):
    def __init__(
            self, 
            embeddings,
            deep_input_dim,
            deep_hidden_dims,
            deep_output_dim,
            p=0.1
        ):
        super(CategoryTower, self).__init__()
        self.embeddings = embeddings
        self.dropout = nn.Dropout(p)

        deep_layers = []
        input_dims = [deep_input_dim] + deep_hidden_dims
        for i in range(len(deep_hidden_dims)):
            deep_layers.append(nn.Linear(input_dims[i], input_dims[i + 1]))
            deep_layers.append(nn.BatchNorm1d(input_dims[i + 1]))
            deep_layers.append(nn.ReLU())
        deep_layers.append(nn.Linear(deep_hidden_dims[-1], deep_output_dim))
        self.deep = nn.Sequential(*deep_layers)
        self.apply(init_weights)

    def forward(self, target_category_idxs):
        target_features = self.embeddings(target_category_idxs).squeeze(1)
        target_output = self.deep(target_features)
        return target_output
    
            
class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, category_tower):
        super(TwoTowerModel, self).__init__()
        self.user_tower = user_tower
        self.category_tower = category_tower
        # self.fc = nn.Linear(64, 32)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.bn1 = nn.BatchNorm1d(32)  # Add batch normalization layer
        self.apply(init_weights)
        
    def forward(self, c_feature, n_feature, visit_category_idxs, target_category_idxs):
        user_output = self.user_tower(c_feature, n_feature, visit_category_idxs)
        category_output = self.category_tower(target_category_idxs)
        
        user_output = self.bn1(user_output)
        category_output = self.bn1(category_output)
        
        concat_output = torch.cat([user_output, category_output], 1)
        u_c_interaction = self.fc1(concat_output)
        u_c_interaction = self.fc2(F.relu(u_c_interaction))
        u_c_interaction = u_c_interaction.squeeze(1)
        return u_c_interaction
    
