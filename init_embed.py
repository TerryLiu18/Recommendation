# https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial

import os
import logging  
import argparse
import torch
import numpy as np
import torch.nn as nn
import multiprocessing
from gensim.models import Word2Vec
from time import time
from config import UNK_IDX, PAD_IDX
from utils import (
    str2bool, 
    save_dict_to_json, 
    read_dict_from_json, 
    read_raw_data
)

logging.basicConfig(format="%(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


def remove_repeats(arr):
    return [arr[i] for i in range(len(arr)) if i == 0 or arr[i] != arr[i - 1]] 

def train_model(sequences, vec_size=128):
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=vec_size,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
    t = time()
    w2v_model.build_vocab(sequences, progress_per=1000000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    
    t = time()
    w2v_model.train(sequences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=10)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    return w2v_model

def generate_embeddings(model, vec_size=128):
    """ generate embeddings for category_id, with idx 0 and 1 stands for UNK and PAD
    """
    pad_category_embedding = np.zeros((1, vec_size))
    unk_category_embedding = np.zeros((1, vec_size))
    embeddings = np.vstack((pad_category_embedding, unk_category_embedding, model.wv.vectors))
    idx2categoryid = dict(enumerate([UNK_IDX, PAD_IDX] + model.wv.index_to_key))
    catergoryid2idx = {int(v): int(k) for k, v in idx2categoryid.items()}
    return embeddings, idx2categoryid, catergoryid2idx


def get_train_data(data='raw_data/pagevisit30.csv'):
    pagevisit = read_raw_data(data)
    pagevisit['trans_category_id_list'] = pagevisit['trans_category_id'].apply(lambda x: [x])
    pagevisit['seq'] = pagevisit['visit_category_id_list'] + pagevisit['trans_category_id_list']
    # filter out seq with len smaller than 2
    # pagevisit = pagevisit[pagevisit['seq'].apply(len) > 2]
    train_data = pagevisit['seq']
    return train_data

def init_embed(overwrite=False):
    if os.path.exists('models/category_embeddings.pt') and not overwrite:
        # logging.info('use existing embeddings')
        EMBEDDINGS = nn.Embedding.from_pretrained(torch.load('models/category_embeddings.pt'), freeze=False)
        IDX2CATEGORYID = read_dict_from_json('models/idx2categoryid.json')
        CATEGORYID2IDX = {int(v): int(k) for k, v in IDX2CATEGORYID.items()}
    else:
        logging.info('start new init embeddings')
        c2v_train_data = get_train_data()
        model = train_model(c2v_train_data)
        embeddings_init, IDX2CATEGORYID, CATEGORYID2IDX = generate_embeddings(model)
        embeddings_init = torch.tensor(embeddings_init).float()
        torch.save(embeddings_init, 'models/category_embeddings.pt')
        EMBEDDINGS = nn.Embedding.from_pretrained(embeddings_init, freeze=False)
        save_dict_to_json(IDX2CATEGORYID, 'models/idx2categoryid.json')
        save_dict_to_json(CATEGORYID2IDX, 'models/categoryid2idx.json')
    # logging.info('category embeddings and index loaded')
    return EMBEDDINGS, IDX2CATEGORYID, CATEGORYID2IDX

# pass overwrite or not as argument
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', type=str2bool, default=False)
    args = parser.parse_args()
    EMBEDDINGS, IDX2CATEGORYID, CATEGORYID2IDX = init_embed(args.overwrite)
    logging.info('length of embeddings: {}'.format(len(IDX2CATEGORYID)))
    print('done')