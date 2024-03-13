# %%
import os
import pandas as pd
import numpy as np
from utils import read_raw_data
from functools import partial
from haversine import haversine, haversine_vector, Unit
# read the data
    
def haversine_numpy(source_lats, source_lons, target_lats, target_lons):
    R = 6373.0
    source_lats = np.radians(source_lats)
    source_lons = np.radians(source_lons)
    target_lats = np.radians(target_lats[:, np.newaxis])
    target_lons = np.radians(target_lons[:, np.newaxis])
    dlat = target_lats - source_lats
    dlon = target_lons - source_lons
    a = np.sin(dlat / 2)**2 + np.cos(source_lats) * np.cos(target_lats) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def simply_distance(source_lats, source_lons, target_lats, target_lons):
    source_lats = np.radians(source_lats)
    source_lons = np.radians(source_lons)
    target_lats = np.radians(target_lats)
    target_lons = np.radians(target_lons)
    dlat = target_lats - source_lats
    dlon = target_lons - source_lons
    simplified_distances_deg = np.sqrt(dlat**2 + dlon**2)
    simplified_distances_km = simplified_distances_deg * 111
    return simplified_distances_km


class CandidateGenerator:
    @staticmethod
    def get_category_list(df):
        return df['category_id'].unique().tolist()
    
    @staticmethod
    def event_filter_by_time(df, start_time='2000-01-01', end_time='2099-12-31') -> pd.DataFrame:
        """
        Filter events based on a time window.
        """
        return df[(df['event_datetime'] >= start_time) & (df['event_datetime'] <= end_time)]

    @staticmethod
    def event_filter_by_gms(df, total_gms_range=[None, None], avg_gms_range=[None, None]) -> pd.DataFrame:
        """
        Filter events based on GMS thresholds.
        """
        min_total_gms, max_total_gms = total_gms_range
        min_avg_gms, max_avg_gms = avg_gms_range
        
        default_mask = np.ones(len(df), dtype=bool)
        min_total_gms_mask = max_total_gms_mask = min_avg_gms_mask = max_avg_gms_mask = default_mask
        if min_total_gms:
            min_total_gms_mask = df['total_gms'] >= min_total_gms
        if max_total_gms:
            max_total_gms_mask = df['total_gms'] <= max_total_gms
        if min_avg_gms:
            min_avg_gms_mask = df['avg_gms'] >= min_avg_gms
        if max_avg_gms:
            max_avg_gms_mask = df['avg_gms'] <= max_avg_gms
        return df[min_total_gms_mask & max_total_gms_mask & min_avg_gms_mask & max_avg_gms_mask]    
    
    @staticmethod
    def event_filter_by_category_type(df, category_type_ids=None):
        """
        Filter events based on category type IDs.
        """
        if category_type_ids is not None:
            return df[df['category_type_id'].isin(category_type_ids)]
        return df
    
    @staticmethod
    def category_dynamic_filter_by_time(df, start_time_list, end_time_list, stop_criterion=1000) -> list:
        candidate_num = 0
        assert len(start_time_list) == len(end_time_list), "The length of start_time_list and end_time_list should be the same."
        if len(start_time_list) == 1:
            return CandidateGenerator.category_filter_by_time(df, start_time_list[0], end_time_list[0])
        
        for i in range(len(start_time_list)):
            filted_event_df = CandidateGenerator.event_filter_by_time(df, start_time_list[i], end_time_list[i])            
            category_list = filted_event_df['category_id'].unique().tolist()
            candidate_num = max(candidate_num, len(category_list))
            if candidate_num >= stop_criterion: break
        return category_list
    
    @staticmethod
    def event_filter_pipeline(df, filters):
        """
        Apply a series of filters to refine the candidate category IDs.
        
        :param filters: A list of functions (filters) to apply to the dataframe.
        """
        for filter_func in filters:
            df = filter_func(df)
        return df
    
    @staticmethod
    def filter_by_distance(df, input_lats, input_lons, max_distance_km=500, method='simple'):
        if method == 'simple':
            return CandidateGenerator.filter_by_distance_simple(df, input_lats, input_lons, max_distance_km)
        elif method == 'haversine':
            return CandidateGenerator.filter_by_distance_haversine(df, input_lats, input_lons, max_distance_km)
        else:
            raise ValueError(f"Invalid method: {method}. Use 'simple' or 'accurate'.")

    @staticmethod
    def event_filter_by_distance(df, input_lat, input_lon, max_distance_km=500, method='simple'):
        input_loc = np.array([input_lat, input_lon])
        target_locs = df[['lat', 'lon']].to_numpy()
        distances = haversine_vector(input_loc, target_locs, comb=True)
        distance_mask = distances <= max_distance_km
        return df[distance_mask]
        
    @staticmethod
    def filter_by_distance_simple(df, input_lats, input_lons, max_distance_km=500):
        simplified_distances_km = simply_distance(input_lats, input_lons, df['lat'].to_numpy(), df['lon'].to_numpy()) 
        within_distance_mask = simplified_distances_km <= max_distance_km
        categories_per_input_loc = []
        
        for mask in within_distance_mask:
            # Use the mask to filter the DataFrame and get unique category_ids
            category_ids = df[mask]['category_id'].unique().tolist()
            categories_per_input_loc.append(category_ids)
        return categories_per_input_loc
    
    @staticmethod
    def filter_by_distance_haversine(df, input_lats, input_lons, max_distance_km=500):
        """
        Filter events based on location proximity and return a list of available category_ids
        for each input location.

        :param input_lats: List of latitudes for input locations.
        :param input_lons: List of longitudes for input locations.
        :param max_distance_km: Maximum distance in kilometers for location-based filtering.
        :return: A list of lists, where each inner list contains the available category_ids
                 within max_distance_km for the corresponding input location.
        """
        distances = haversine_numpy(np.array(list(input_lats)), 
                                    np.array(list(input_lons)),
                                    df['lat'].to_numpy(),
                                    df['lon'].to_numpy())
        within_distance_mask = distances <= max_distance_km
        categories_per_input_loc = []
        t1 = time()
        
        for mask in within_distance_mask.T:
            # Use the mask to filter the DataFrame and get unique category_ids
            category_ids = df[mask]['category_id'].unique().tolist()
            categories_per_input_loc.append(category_ids)
        return categories_per_input_loc

    
from time import time
if __name__ == "__main__":
    c_df = read_raw_data('raw_data/all_category_info.csv')
    cg = CandidateGenerator()
    
    time_strategy = partial(cg.event_filter_by_time, start_time='2023-11-03', end_time='2023-12-31')
    gms_strategy = partial(cg.event_filter_by_gms, total_gms_range=[None, None], avg_gms_range=[1000, None])    
    
    irvine = (33.6846, -117.8265)
    dist_strategy = partial(cg.event_filter_by_distance, input_lat=irvine[0], input_lon=irvine[1], max_distance_km=100)
    
    t1 = time()
    for _ in range(1000):
        res1 = cg.event_filter_pipeline(c_df, [time_strategy, gms_strategy, dist_strategy])
    t2 = time()
    print(f"Pipeline method: {t2 - t1} seconds")
    print(res1.shape)
    
    
    #randomize 1000 input locations
    # repeat = 5000
    # input_lats = np.random.uniform(33, 42, repeat)
    # input_lons = np.random.uniform(-118, -112, repeat)

    # t0 = time()
    # res1 = cg.filter_by_distance(input_lats, input_lons, max_distance_km=100, method='simple')
    # t1 = time()
    # res2 = cg.filter_by_distance(input_lats, input_lons, max_distance_km=100, method='haversine')
    # t2 = time()
    # print(f"Simple method: {t1 - t0} seconds")
    # print(f"Accurate method: {t2 - t1} seconds")

# # compare the difference between the two methods, res are list of list of category_id, overlap divided by total distanct as percentage metric, calculate the average

# accs = []
# for i, (r1, r2) in enumerate(zip(res1, res2)):
#     overlap = len(set(r1) & set(r2))
#     all_category = len(set(r1) | set(r2))
#     if all_category == 0:
#         continue
#     else:
#         accuracy = overlap / all_category
#         accs.append(accuracy)
# print(np.mean(accs))
    
# random 1000 input times
# input_time_list, end_time_list = [], []
# for i in range(1000):
#     input_time_list.append(np.random.choice(c_df['event_datetime'].values))
#     end_time_list.append(np.random.choice(c_df['event_datetime'].values))

# res3 = cg.event_filter_by_time(c_df, start_time='2023-11-03', end_time='2023-12-31')
    

# %%
