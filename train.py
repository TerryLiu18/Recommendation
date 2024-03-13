
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
from config import TestConfig
from utils import read_raw_data
from sklearn.preprocessing import PolynomialFeatures  
from model import WDNet, PageDataset, pad_list_collate_fn


def test_hitrate(model, test_loader):
    model.eval()
    
    hit = 0
    test_count = len(test_loader)
    
    test_target_category_id_list = []
    ground_truth_category_id = []
    
    # TODO: find more efficient approach
    for i, data in enumerate(test_loader):
        trans_category_id_tensor, _, _, _, target_category_id_tensor, _, _ = data 
            
        ground_truth_category_id.append(int(trans_category_id_tensor[0].item()))
        # convert target_category_id_tensor to a list
        target_category_id_list = target_category_id_tensor.reshape(-1).tolist()
        target_category_id_list = [int(cid) for cid in target_category_id_list]
        test_target_category_id_list.append(target_category_id_list)
        
    for bid, data in enumerate(test_loader):
        trans_category_id_tensor, c_feature_tensor, n_feature_tensor, visit_category_idxs_tensor, \
            target_category_id_tensor, target_category_idx_tensor, label_tensor = data 
            
        score = model((c_feature_tensor, n_feature_tensor, visit_category_idxs_tensor, target_category_idx_tensor))
        top_5_bidx = torch.argsort(score, descending=True)[:5].tolist()
        top_5_category_id = [test_target_category_id_list[bid][top_5_bidx[i]] for i in range(5)]
        if ground_truth_category_id[bid] in top_5_category_id:
            hit += 1
            # print(f'ground truth: {ground_truth_category_id} | rec5: {top_5_category_id}')
    print(f'hit: {hit} | test_count: {test_count}')
    model.train()
    return hit / test_count

        
if __name__ == '__main__':
    train_data = read_raw_data('aux_data/final_train_data.csv')
    test_data = read_raw_data('aux_data/final_test_data.csv')
    numerical_cols = ['total_gms', 'avg_gms', 'to_event_time_days_min', 'to_event_time_days_avg',
                    'to_event_distance_min', 'to_event_distance_avg', 'visit_avg_total_gms',
                    'visit_avg_avg_gms']

    attr_onehot = joblib.load('models/attr_onehot.pkl')
    
    train_onehot_columns = attr_onehot.transform(train_data[['category_type_idx', 'category_group_idx', 'top_level_idx']])
    test_onehot_columns = attr_onehot.transform(test_data[['category_type_idx', 'category_group_idx', 'top_level_idx']])
    
    onehot_column_names = attr_onehot.get_feature_names_out(['category_type_idx', 'category_group_idx', 'top_level_idx']).tolist()
    onehot_column_names = attr_onehot.get_feature_names_out(['category_type_idx', 'category_group_idx', 'top_level_idx']).tolist()
    
    train_onehot_df = pd.DataFrame(train_onehot_columns, columns=onehot_column_names).astype(int)
    test_onehot_df = pd.DataFrame(test_onehot_columns, columns=onehot_column_names).astype(int)
    
    train_data = pd.concat([train_data, train_onehot_df], axis=1)
    test_data = pd.concat([test_data, test_onehot_df], axis=1)
    
    ID2IDX = read_dict_from_json('models/categoryid2idx.json')
    
    print(train_data.columns)
    print(train_data.head(4))
    print('---' * 20 + 'test data' + '---' * 20)
    print(test_data.columns)
    print(test_data.head(4))

    train_dataset = PageDataset(train_data, ID2IDX, numerical_cols, onehot_column_names, 'visit_category_id_list', 'target_category_id')
    trainloader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=pad_list_collate_fn)
    
    test_data = PageDataset(test_data, ID2IDX, numerical_cols, onehot_column_names, 'visit_category_id_list', 'target_category_id')
    testloader = DataLoader(test_data, batch_size=TestConfig.candidate_num, shuffle=False, collate_fn=pad_list_collate_fn) 
    
    embeddings = nn.Embedding.from_pretrained(torch.load('models/category_embeddings.pt'), freeze=False)
    
    embed_dim = embeddings.embedding_dim
    n_dim = len(numerical_cols)
    deep_input_size = embed_dim * 2 + n_dim
    wide_input_size = len(onehot_column_names)
    
    epochs = 8
    loss_func = nn.BCEWithLogitsLoss()
    model = WDNet(embeddings, deep_input_size, wide_input_size, 0.1)
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(epochs):
        loss_list = []
        hitrate_list = []
        
        for step, data in enumerate(trainloader):
            trans_category_id_tensor, c_feature_tensor, n_feature_tensor, visit_category_idxs_tensor, \
                target_category_id_tensor, target_category_idx_tensor, label_tensor = data
            # print(trans_category_id_tensor.shape, c_feature_tensor.shape, n_feature_tensor.shape, visit_category_idxs_tensor.shape, target_category_idx_tensor.shape, label_tensor.shape)
            optimizer.zero_grad()
            input_data = (c_feature_tensor, n_feature_tensor, visit_category_idxs_tensor, target_category_idx_tensor)
            score = model(input_data)
            loss = loss_func(score, label_tensor.float())
            
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            
            if step % TestConfig.testing_freq == 0:
                hitrate = test_hitrate(model, testloader)
                print(f'epoch: {epoch}, step: {step}, loss: {loss.item()}')
                hitrate_list.append(hitrate)
        scheduler.step()
    torch.save(model.state_dict(), 'models/wdnet.pt')
            
        
    
    