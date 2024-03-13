UNK_IDX = 0
PAD_IDX = 1
page_visit_cols = [
    'page_visit_id', 'visit_category_id_list', 'trans_time',
    'visit_lat', 'visit_lon', 'trans_category_id'
]
# numerical_feature_cols = ['min_distance', 'gms', 'max_visit_gms', 'avg_visit_gms', 'next_event_time_diff', 'min_dist_time_diff']
numerical_feature_cols = ['min_distance', 'gms']
# numerical_feature_cols = ['min_distance', 'gms', 'max_visit_gms', 'avg_visit_gms', 'next_event_time_diff', 'min_dist_time_diff']
categorical_feature_cols = ['category_type_id', 'top_level_id']
interaction_feature_cols = ['visit_category_id_list']
fill_na_dict = {'min_distance': 5000, 'gms': 0, 'time_hours_diff': 50000}


class TestConfig:
    candidate_num = 200
    freq_threshold = 20
    testing_size = 500
    testing_freq = 200
    
    
    

