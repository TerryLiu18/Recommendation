# this is used for a new version of pagevisit data 

import argparse
from utils import read_raw_data

def parse_raw_data(data_path, save_path):
    def parse_row(r):
        trans_category_id, visit_category_id_list = r['trans_category_id'], r['visit_category_id_list'] 
        visit_category_id_list = visit_category_id_list[:visit_category_id_list.index(trans_category_id)]
        return visit_category_id_list
    
    train_data = read_raw_data(args.data_path)
    train_data['visit_category_id_list'] = train_data.apply(parse_row, axis=1)
    train_data = train_data[train_data['visit_category_id_list'].apply(len) > 0]
    train_data['trans_event_id'] = train_data['trans_event_id'].fillna(0)
    train_data['trans_event_id'] = train_data['trans_event_id'].astype(int)
    train_data.to_csv(save_path, index=False)
    print('process done')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='raw_data/pagevisit_final_version3.csv')
    parser.add_argument('--save_path', type=str, default='raw_data/pagevisit_new.csv')
    args = parser.parse_args()
    parse_raw_data(args.data_path, args.save_path)
    
    # train_data = read_raw_data(args.data_path)
    # train_data['visit_category_id_list'] = train_data.apply(parse_row, axis=1)
    # train_data = train_data[train_data['visit_category_id_list'].apply(len) > 0]
    # train_data['trans_event_id'] = train_data['trans_event_id'].fillna(0)
    # train_data['trans_event_id'] = train_data['trans_event_id'].astype(int)
    # train_data.to_csv('raw_data/pagevisit_new.csv', index=False)