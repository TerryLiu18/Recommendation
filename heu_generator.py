# %%
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from base_generator import BaseGenerator

class HeuModel(BaseGenerator):
    def __init__(self, ID2IDX, IDX2ID, pagevisit):
        self.ID2IDX = ID2IDX
        self.IDX2ID = IDX2ID
        self.pagevisit = pagevisit
        self.occur_matrix = self.__get_cooccur_matrix()

    def __get_cooccur_matrix(self):
        occur_matrix = np.zeros((len(self.ID2IDX), len(self.ID2IDX)))
        for _, row in tqdm(self.pagevisit.iterrows(), total=self.pagevisit.shape[0]):
            visit_category_id_list, trans_category_id = row['visit_category_id_list'], row['trans_category_id']
            vis_cat_idxs = [self.ID2IDX.get(int(cat_id), 0) for cat_id in visit_category_id_list]
            trs_cat_idx = self.ID2IDX.get(trans_category_id, 0)
            for vis_cat_idx in vis_cat_idxs:
                if vis_cat_idx > 0 and trs_cat_idx > 0:  # Assuming you meant index 0 as invalid
                    occur_matrix[vis_cat_idx, trs_cat_idx] += 1
        return occur_matrix
    
    def generate_baseline_candidates(self, candidate_num):
        purchase_appear_times = self.occur_matrix.sum(axis=0)
        top_N_category_idx = purchase_appear_times.argsort()[-candidate_num:][::-1]
        top_candidate_category_id = [self.IDX2ID.get(idx, 0) for idx in top_N_category_idx]
        top_candidate_category_id = [cand for cand in top_candidate_category_id if cand > 0]
        return top_candidate_category_id
    
    def generate_candidates(self, category_ids, candidate_num, freq_threshold=20):
        id2score = defaultdict(int)
        for category_id in category_ids:
            idx = self.ID2IDX.get(category_id, 0)
            top_candidate_idxs = np.argsort(self.occur_matrix[idx])[::-1][:candidate_num]
            top_candidate_idxs = [cand for cand in top_candidate_idxs 
                                if self.occur_matrix[idx][cand] > freq_threshold]
            top_candidate_ids = [self.IDX2ID.get(int(idx), 0) for idx in top_candidate_idxs]
            top_candidate_ids = [cand for cand in top_candidate_ids if cand > 0]
            for cand in top_candidate_ids:
                id2score[cand] += 1
                
        return_candidate_category_ids = [
            category_id for category_id, _ in sorted(
                id2score.items(), key=lambda x: x[1], reverse=True)[:candidate_num]
        ]
        return return_candidate_category_ids
        
    def guranatee_k_candidate_generate(self, category_id, topk=50, threshold=20):
        """ guranatee to generate topK candidates by backfilling with baselines 
        """
        input_topK_category_id = self.generate_candidates(category_id, topk, threshold)
        len_candidate = len(input_topK_category_id)
        input_candidate_set = set(input_topK_category_id)
        top_N_category_id = self.generate_baseline_candidates(topk)
        while len_candidate < topk:
            for cid in top_N_category_id:
                if cid not in input_candidate_set:
                    input_topK_category_id.append(cid)
                    input_candidate_set.add(cid)
                    len_candidate += 1
                if len_candidate >= topk:
                    break
        return input_topK_category_id
