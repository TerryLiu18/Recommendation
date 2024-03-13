import os
import time
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from haversine import haversine_vector, Unit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from haversine import Unit, haversine_vector
from utils import read_raw_data, save_dict_to_json
from heu_generator import HeuModel
from init_embed import init_embed
from config import TestConfig

# print test config
print("TestConfig: \n")
print(TestConfig.__dict__)

def candidate_generator(category_id: int, heu, candidate_num=TestConfig.candidate_num, freq_threshold=20) -> list:
    candidates = heu.simple_candidate_generate(category_id, topk=candidate_num, threshold=freq_threshold)
    return candidates


def get_top_category_ids(category_df, top=1000):
    category_df = category_df.sort_values(by='total_gms', ascending=False)
    return category_df['category_id'][:top].values.tolist()

def get_bottom_category_ids(category_df, bottom=1000):
    category_df = category_df.sort_values(by='total_gms', ascending=True)
    return category_df['category_id'][:bottom].values.tolist()

def get_top_category_names(category_df, top=1000):
    category_df = category_df.sort_values(by='total_gms', ascending=False)
    return category_df['category_name'][:top].values.tolist()


def create_test_dataset(
    test_pagevisit: pd.DataFrame, 
    generator,
    cand_num: int=50,
    freq_th: int=3
    ) -> pd.DataFrame:
    """ use generator to create candidates for each
        trans_category_id in test_pagevisit
    """    
    candidates_list = []
    for category_ids in tqdm(test_pagevisit['visit_category_id_list'].tolist()):
        candidates_list.append(
            generator.guranatee_k_candidate_generate(
                category_ids, topk=cand_num, threshold=freq_th))
        
    # tile the test_pagevisit dataframe, each row is repeated for cand_num times
    test_pagevisit_with_cand = test_pagevisit.loc[test_pagevisit.index.repeat(cand_num)].reset_index(drop=True)
    flattened_candidates = [item for sublist in candidates_list for item in sublist]
    test_pagevisit_with_cand['target_category_id'] = flattened_candidates
    test_pagevisit_with_cand['label'] = [0] * len(test_pagevisit_with_cand) # placeholder
    return test_pagevisit_with_cand

    
def create_neg_samples(pagevisit_df: pd.DataFrame, neg_candidate_list: list, ratio:int=2) -> pd.DataFrame:
    """ expand the pagevisit_df dataframe by adding negative samples

    Args:
        pagevisit_df (pd.DataFrame): raw pagevisit_df dataframe
        neg_candidate_set (set): negative candidate set
        ratio (int, optional): pos: neg = 1: ratio, defaults to 2.

    Returns:
        pd.DataFrame: expanded pagevisit_df dataframe
    """
    with_negative_df_list = [pagevisit_df]
    for _ in range(ratio):
        temp_df = pagevisit_df.copy()
        temp_df['page_visit_id'] = '-' + temp_df['page_visit_id'].astype(str)
        with_negative_df_list.append(temp_df)
        
    pagevisit_df = pd.concat(with_negative_df_list, axis=0, ignore_index=True)
    pagevisit_df['target_category_id'] = range(len(pagevisit_df))
    pagevisit_df.loc[:len(pagevisit_df) // (ratio + 1), 'target_category_id'] = pagevisit_df.loc[
        :len(pagevisit_df) // (ratio + 1), 'trans_category_id'
    ]
        
    negative_sample = np.random.choice(neg_candidate_list, size=ratio * len(pagevisit_df) // (ratio + 1)) 
    pagevisit_df.loc[len(pagevisit_df) // (ratio + 1): , 'target_category_id'] = negative_sample
    pagevisit_df['label'] = (pagevisit_df['trans_category_id'] == pagevisit_df['target_category_id']).astype(int)
    return pagevisit_df


def scale_gms(category_df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    scaler.fit(category_df[['total_gms', 'avg_gms']].values)
    scaled_columns = scaler.transform(category_df[['total_gms', 'avg_gms']].values)
    category_df[['total_gms', 'avg_gms']] = scaled_columns
    return category_df, scaler


def categorical_encode(category_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    # category_df = category_df.sort_values(by='total_gms', ascending=False)
    category_type_ids = category_df['category_type_id'].unique().tolist()
    category_group_ids = category_df['category_group_id'].unique().tolist()
    top_level_ids = category_df['top_level_id'].unique().tolist()
    
    type_encoder = OrdinalEncoder(encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)
    group_encoder = OrdinalEncoder(encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)
    top_level_encoder = OrdinalEncoder(encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)
    
    type_encoder.fit(np.array(category_type_ids).reshape(-1, 1))
    group_encoder.fit(np.array(category_group_ids).reshape(-1, 1))
    top_level_encoder.fit(np.array(top_level_ids).reshape(-1, 1))
    
    category_df['category_type_idx'] = type_encoder.transform(np.array(category_df['category_type_id']).reshape(-1, 1)).astype(int)
    category_df['category_group_idx'] = group_encoder.transform(np.array(category_df['category_group_id']).reshape(-1, 1)).astype(int)
    category_df['top_level_idx'] = top_level_encoder.transform(np.array(category_df['top_level_id']).reshape(-1, 1)).astype(int)
    
    joblib.dump(type_encoder, 'models/type_encoder.pkl')
    joblib.dump(group_encoder, 'models/group_encoder.pkl')
    joblib.dump(top_level_encoder, 'models/top_level_encoder.pkl')
    return type_encoder, group_encoder, top_level_encoder


def onehot_encode(category_df: pd.DataFrame, cols: list) -> pd.DataFrame:
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoder.fit(category_df[cols])
    onehot_columns = onehot_encoder.transform(category_df[cols])
    onehot_column_names = onehot_encoder.get_feature_names_out(cols).tolist()
    onehot_df = pd.DataFrame(onehot_columns, columns=onehot_column_names).astype(int)
    joblib.dump(onehot_encoder, 'models/attr_onehot.pkl')
    return onehot_encoder, onehot_df


def parse_dist_and_time(train_data):
    train_data['event_datetime'] = pd.to_datetime(train_data['event_datetime'])
    train_data['trans_time'] = pd.to_datetime(train_data['trans_time'])
    train_data['event_created_datetime'] = pd.to_datetime(train_data['event_created_datetime'])
    # time_mask = ((train_data['trans_time'] - train_data['event_datetime']).dt.total_seconds() < 0) \
    #                 & ((train_data['trans_time'] - train_data['event_created_datetime']).dt.total_seconds() > 0)
    # train_data = train_data[time_mask]
    train_data['to_event_time_days'] = (train_data['event_datetime'] - train_data['trans_time']).dt.total_seconds() / (3600 * 24)
    train_data['to_event_time_days'] = train_data['to_event_time_days'].apply(lambda x: 500 if x < 0 else x)
    
    train_data['to_event_distance'] = haversine_vector(
        train_data[['visit_lat', 'visit_lon']].values, 
        train_data[['event_lat', 'event_lon']].values,
        Unit.KILOMETERS)

    train_data_with_dist_and_time = train_data.groupby(['page_visit_id', 'trans_category_id', 'target_category_id']).agg(
        to_event_time_days_min=('to_event_time_days', 'min'),
        to_event_time_days_avg=('to_event_time_days', 'mean'),

        to_event_distance_min=('to_event_distance', 'min'),
        to_event_distance_avg=('to_event_distance', 'mean')
    ).reset_index()
    return train_data_with_dist_and_time


def get_category_attr(pagevisit_df:pd.DataFrame) -> pd.DataFrame:
    # TODO: find some more efficient way to join the gms and lat, lon information

    pagevisit_df['visit_total_gms_list'] = pagevisit_df.apply(
        lambda row: [total_gms_dict.get(int(cid), 0) if cid else 0 for cid in row['visit_category_id_list']], axis=1
    )
    pagevisit_df['visit_avg_gms_list'] = pagevisit_df.apply(
        lambda row: [avg_gms_dict.get(int(cid), 0) if cid else 0 for cid in row['visit_category_id_list']], axis=1
    )
    pagevisit_df['visit_lat_list'] = pagevisit_df.apply(
        lambda row: [event_lat_dict.get(int(eid), 0) for eid in row['visit_event_id_list'] if eid and int(eid) in event_lat_dict], axis=1
    )
    pagevisit_df['visit_lon_list'] = pagevisit_df.apply(
        lambda row: [event_lon_dict.get(int(eid), 0) for eid in row['visit_event_id_list'] if eid and int(eid) in event_lon_dict], axis=1
    )
    return pagevisit_df


def data_merge(result, non_event_data, mode='train', dist_scaler=None, day_scaler=None):
    needed_cols = ['page_visit_id', 'visit_category_id_list', 
                    'visit_total_gms_list', 'visit_avg_gms_list',
                    'trans_category_id', 'target_category_id', 'label', 
                    'total_gms', 'avg_gms', 'category_type_id',
                    'category_group_id', 'top_level_id'] + ['category_type_idx', 'category_group_idx', 'top_level_idx']
    final_data = non_event_data[needed_cols].merge(
        result, on=['page_visit_id', 'trans_category_id', 'target_category_id'], how='inner')

    final_data['visit_avg_total_gms'] = final_data['visit_total_gms_list'].apply(
        lambda x: np.mean([i for i in x if i > 0]) if any(i > 0 for i in x) else 0
    )
    final_data['visit_avg_avg_gms'] = final_data['visit_avg_gms_list'].apply(
        lambda x: np.mean([i for i in x if i > 0]) if any(i > 0 for i in x) else 0
    )
    
    # a bit of hacky here, when train, pass two empty scalers
    if mode == 'train':
        dist_scaler = MinMaxScaler()
        day_scaler = MinMaxScaler()
        dist_scaler.fit(final_data[['to_event_distance_avg']].values)
        day_scaler.fit(final_data[['to_event_time_days_avg']].values)

        joblib.dump(dist_scaler, 'models/dist_scaler.pkl')
        joblib.dump(day_scaler, 'models/day_scaler.pkl')

    final_data[['to_event_distance_avg']] = dist_scaler.transform(final_data[['to_event_distance_avg']].values)
    final_data[['to_event_distance_min']] = dist_scaler.transform(final_data[['to_event_distance_min']].values)
    final_data[['to_event_time_days_avg']] = day_scaler.transform(final_data[['to_event_time_days_avg']].values)
    final_data[['to_event_time_days_min']] = day_scaler.transform(final_data[['to_event_time_days_min']].values)
    final_data['visit_avg_avg_gms'] = final_data['visit_avg_avg_gms'].fillna(0)
    final_data['visit_avg_total_gms'] = final_data['visit_avg_total_gms'].fillna(0)
    final_data.drop(['visit_total_gms_list', 'visit_avg_gms_list'], axis=1, inplace=True)
    return final_data, dist_scaler, day_scaler
    
if __name__ == '__main__':
    start = time.time()
    EMBEDDINGS, IDX2ID, ID2IDX = init_embed(overwrite=False)
    train_pagevisit = read_raw_data('raw_data/pagevisit30.csv')
    test_pagevisit = read_raw_data('raw_data/pagevisit4_test.csv')[:TestConfig.testing_size]
    category_info = read_raw_data('raw_data/all_category_info.csv')
    heu = HeuModel(ID2IDX, IDX2ID, train_pagevisit)
    
    category_info, gms_scaler = scale_gms(category_info) # category_info is shared by train and test
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(gms_scaler, 'models/gms_scaler.pkl')
    type_encoder, group_encoder, top_level_encoder = categorical_encode(category_info, ['category_type_id', 'category_group_id', 'top_level_id'])
    attr_onehot, onehot_df = onehot_encode(category_info, ['category_type_idx', 'category_group_idx', 'top_level_idx'])

    cat_non_event_cols = ['category_name', 'total_gms', 'avg_gms', 
                        'category_type_id', 'category_group_id', 'top_level_id'] + ['category_type_idx', 'category_group_idx', 'top_level_idx']
    cat_event_cols = ['event_id', 'event_name', 'event_lat', 'event_lon', 
                    'event_datetime', 'event_created_datetime']

    category_stats_df = category_info.drop_duplicates(subset=['category_id', 'total_gms', 'avg_gms'])[['category_id'] + cat_non_event_cols]
    total_gms_dict = category_stats_df.set_index('category_id')['total_gms'].to_dict()
    avg_gms_dict = category_stats_df.set_index('category_id')['avg_gms'].to_dict()
    event_lat_dict = category_info.set_index('event_id')['event_lat'].to_dict()
    event_lon_dict = category_info.set_index('event_id')['event_lon'].to_dict()

    train_pagevisit = get_category_attr(train_pagevisit)
    test_pagevisit  = get_category_attr(test_pagevisit)

    hard_sample = get_top_category_ids(category_stats_df, top=300) 
    easy_sample = get_bottom_category_ids(category_stats_df, bottom=800)
    neg_sample_candidate = hard_sample + easy_sample
    train_df = create_neg_samples(train_pagevisit, neg_sample_candidate, ratio=2)
    test_df = create_test_dataset(test_pagevisit, heu, cand_num=TestConfig.candidate_num, freq_th=10)

    non_event_train_data = train_df.merge(category_stats_df, left_on='target_category_id', right_on='category_id', how='inner')
    non_event_test_data = test_df.merge(category_stats_df, left_on='target_category_id', right_on='category_id', how='left')

    train_data = train_df[['page_visit_id', 'trans_category_id', 'target_category_id',
                    'visit_lat', 'visit_lon', 'trans_time']].merge(category_info[['category_id'] + cat_event_cols], left_on='target_category_id', right_on='category_id', how='inner')
    test_data = test_df[['page_visit_id', 'trans_category_id', 'target_category_id',
                    'visit_lat', 'visit_lon', 'trans_time']].merge(category_info[['category_id'] + cat_event_cols], left_on='target_category_id', right_on='category_id', how='left')

    train_with_dist_and_time = parse_dist_and_time(train_data)
    test_with_dist_and_time = parse_dist_and_time(test_data)

    final_train_data, dist_scaler, day_scaler = data_merge(
        train_with_dist_and_time, non_event_train_data, 'train', dist_scaler=None, day_scaler=None)
    final_test_data, _, _ = data_merge(
        test_with_dist_and_time, non_event_test_data, 'test', dist_scaler, day_scaler)

    # save the final train data
    end_time = time.time()
    print(f"Time elapsed: {end_time - start:.2f} seconds")
    
    # final_test_data need to process NaN values
    final_test_data[['total_gms', 'avg_gms', 'category_group_id', 'top_level_id',
                     'category_type_idx', 'category_group_idx', 'top_level_idx']] = \
    final_test_data[['total_gms', 'avg_gms', 'category_group_id', 'top_level_id',
                     'category_type_idx', 'category_group_idx', 'top_level_idx']].fillna(-1)
    
    final_test_data[['to_event_time_days_min', 'to_event_time_days_avg']] = final_test_data[['to_event_time_days_min', 'to_event_time_days_avg']].fillna(2)
    final_test_data[['to_event_distance_min', 'to_event_distance_avg']] = final_test_data[['to_event_distance_min', 'to_event_distance_avg']].fillna(2)
    
    final_train_data.to_csv('aux_data/final_train_data.csv', index=False)
    final_test_data.to_csv('aux_data/final_test_data.csv', index=False)




# gms_scaler = MinMaxScaler()
# gms_scaler.fit(category_info[['total_gms', 'avg_gms']].values)
# scaled_columns = gms_scaler.transform(category_info[['total_gms', 'avg_gms']].values)
# category_info[['total_gms', 'avg_gms']] = scaled_columns
# Save the scaler

# category_df = category_info.sort_values(by='total_gms', ascending=False)
# # category_ids = [0, 1] + category_df['category_id'].unique().tolist()
# category_type_ids = category_df['category_type_id'].unique().tolist()
# category_group_ids = category_df['category_group_id'].unique().tolist()
# top_level_ids = category_df['top_level_id'].unique().tolist()

# ID2IDX = {int(category_id): int(idx) for idx, category_id in enumerate(category_ids)}
# IDX2ID = {int(idx): int(category_id) for category_id, idx in ID2IDX.items()}
# save_dict_to_json(ID2IDX, 'models/categoryid2idx.json')
# save_dict_to_json(IDX2ID, 'models/idx2categoryid.json')

# type_encoder = OrdinalEncoder(encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)
# group_encoder = OrdinalEncoder(encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)
# top_level_encoder = OrdinalEncoder(encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)
# joblib.dump(type_encoder, 'models/type_encoder.pkl')
# joblib.dump(group_encoder, 'models/group_encoder.pkl')
# joblib.dump(top_level_encoder, 'models/top_level_encoder.pkl')

# type_encoder.fit(np.array(category_type_ids).reshape(-1, 1))
# group_encoder.fit(np.array(category_group_ids).reshape(-1, 1))
# top_level_encoder.fit(np.array(top_level_ids).reshape(-1, 1))

# category_info['category_idx'] = category_info['category_id'].apply(lambda x: ID2IDX.get(int(x), 0)).astype(int)
# category_info['category_type_idx'] = type_encoder.transform(np.array(category_info['category_type_id']).reshape(-1, 1)).astype(int)	
# category_info['category_group_idx'] = group_encoder.transform(np.array(category_info['category_group_id']).reshape(-1, 1)).astype(int)
# category_info['top_level_idx'] = top_level_encoder.transform(np.array(category_info['top_level_id']).reshape(-1, 1)).astype(int)



# attr_onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# attr_onehot.fit(category_info[['category_type_idx', 'category_group_idx', 'top_level_idx']])
# onehot_columns = attr_onehot.transform(category_info[['category_type_idx', 'category_group_idx', 'top_level_idx']])
# onehot_column_names = attr_onehot.get_feature_names_out(['category_type_idx', 'category_group_idx', 'top_level_idx']).tolist()
# onehot_df = pd.DataFrame(onehot_columns, columns=onehot_column_names).astype(int)
# category_info = pd.concat([category_info, onehot_df], axis=1)
# joblib.dump(attr_onehot, 'models/attr_onehot.pkl')







