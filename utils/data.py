import networkx as nx
import numpy as np
import scipy
import pickle
import torch

def load_data(prefix='data/preprocessed/ACM_processed'):
    in_file = open(prefix + '/0-1-0.adjlist', 'r')
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00[3:]
    in_file.close()
    in_file = open(prefix + '/0-2-0.adjlist', 'r')
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01[3:]
    in_file.close()
    in_file = open(prefix + '/0-1-0_idx.pickle', 'rb')
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + '/0-2-0_idx.pickle', 'rb')
    idx01 = pickle.load(in_file)
    in_file.close()

    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [adjlist00, adjlist01], \
           [idx00, idx01], \
           [features_0, features_1, features_2], \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx

