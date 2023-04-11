import time
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from utils.tools import index_generator, evaluate_results_nc, parse_minibatch, parse_mask
from model.HetReGAT_FC import HeReGAT_nc_FC
import dgl
from scipy import sparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser(description='HetReGAT-FC')
ap.add_argument('--hidden-dim', type=int, default=64)
ap.add_argument('--num-heads', type=int, default=8)
ap.add_argument('--num_layer_1', type=int, default=2)
ap.add_argument('--num_layer_3', type=int, default=1)
ap.add_argument('--epoch', type=int, default=100)
ap.add_argument('--patience', type=int, default=5)
ap.add_argument('--batch-size', type=int, default=500)
ap.add_argument('--samples', type=int, default=100)
ap.add_argument('--save-postfix', default='ACM')
ap.add_argument('--slope', type=float, default=0.05)
ap.add_argument('--res', default=False)
ap.add_argument('--feats-opt', type=str, default='011')

args = ap.parse_args()
print(args)

hidden_dim = args.hidden_dim
num_heads = args.num_heads
num_epochs = args.epoch
patience = args.patience
batch_size = args.batch_size
neighbor_samples = args.samples
save_postfix = args.save_postfix
feats_opt = args.feats_opt
num_layer_1 = args.num_layer_1
num_layer_2 = args.num_layer_3
slope = args.slope
res = args.res

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

feats_opt = list(feats_opt)
feats_opt = list(map(int, feats_opt))
print('feats_opt: {}'.format(feats_opt))
etypes_list = [[0, 1], [2, 3]]
num_edge_type = 4
src_node_type = 0
num_class = 3
dropout_rate = 0.5
f_drop = 0.5
att_drop = 0.5

activation = F.elu
device = torch.device('cpu')
adjlists, edge_metapath_indices_list, features, adjM, type_mask, labels, train_val_test_idx = load_data()
features_list = [torch.FloatTensor(feature) for feature in features]
onehot_feature_list = [torch.FloatTensor(feature) for feature in features]
in_dim_3 = [features.shape[1] for features in features_list]
labels = torch.LongTensor(labels)
train_idx = train_val_test_idx['train_idx']
train_idx = np.sort(train_idx)
val_idx = train_val_test_idx['val_idx']
val_idx = np.sort(val_idx)
test_idx = train_val_test_idx['test_idx']
test_idx = np.sort(test_idx)

node_type_feature = [[0 for c in range(1)] for r in range(len(features_list))]
node_type_feature_init = F.one_hot(torch.arange(0, len(features_list)), num_classes=len(features_list))

for i in range(0, len(features_list)):
    node_type_feature[i] = node_type_feature_init[i].expand(features_list[i].shape[0], len(node_type_feature_init)).to(
        device).type(torch.FloatTensor)
in_dim_2 = [features.shape[1] for features in node_type_feature]

in_dim_1 = [features.shape[0] for features in onehot_feature_list]
for i in range(0, len(onehot_feature_list)):
    dim = onehot_feature_list[i].shape[0]
    indices = np.vstack((np.arange(dim), np.arange(dim)))
    indices = torch.LongTensor(indices)
    values = torch.FloatTensor(np.ones(dim))
    onehot_feature_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)


adjm = sparse.csr_matrix(adjM)
adjM = torch.FloatTensor(adjM).to(device)
g = dgl.DGLGraph(adjm + (adjm.T))
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
g = g.to(device)


print('Data loading completed!')
net = HeReGAT_nc_FC(g, in_dim_1, in_dim_2, in_dim_3, hidden_dim, num_class,
                num_layer_1, num_layer_2, num_heads, f_drop, att_drop, activation,
                 slope, res, dropout_rate, False, feats_opt)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.000)
print('model init finish\n')

net.train()
early_stopping = EarlyStopping(patience=patience, verbose=True,save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
train_idx_generator = index_generator(batch_size=batch_size, indices=train_idx)
val_idx_generator = index_generator(batch_size=batch_size, indices=val_idx, shuffle=False)
for epoch in range(num_epochs):
    t = time.time()
    net.train()
    train_loss_avg = 0
    for iteration in range(train_idx_generator.num_iterations()):
        train_idx_batch = train_idx_generator.next()
        train_idx_batch.sort()
        train_g_list, train_indices_list, train_idx_batch_mapped_list = parse_minibatch(
            adjlists, edge_metapath_indices_list, train_idx_batch, device, neighbor_samples)
        mask_list, feat_keep_idx, feat_drop_idx = parse_mask(
            indices_list=train_indices_list, type_mask=type_mask, num_classes=num_class,
            src_type=src_node_type, rate=0.01, device=device)
        _, logits, _, _ = net((onehot_feature_list, node_type_feature),
            (adjM, features_list, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
            (train_g_list, type_mask, train_indices_list, train_idx_batch_mapped_list))
        logp = F.log_softmax(logits, 1)
        loss_classification = F.nll_loss(logp[train_idx], labels[train_idx])
        train_loss = loss_classification
        train_loss_avg += loss_classification.item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    train_loss_avg /= train_idx_generator.num_iterations()
    train_time = time.time() - t

    t = time.time()
    net.eval()
    val_logp = []
    val_loss_avg = 0
    with torch.no_grad():
        for iteration in range(val_idx_generator.num_iterations()):
            val_idx_batch = val_idx_generator.next()
            val_g_list, val_indices_list, val_idx_batch_mapped_list = parse_minibatch(
                adjlists, edge_metapath_indices_list, val_idx_batch, device, neighbor_samples)
            mask_list, feat_keep_idx, feat_drop_idx = parse_mask(
                indices_list=val_indices_list, type_mask=type_mask, num_classes=num_class,
                src_type=src_node_type, rate=0.01, device=device)

            _, logits, _, _ = net((onehot_feature_list, node_type_feature),
                (adjM, features_list, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
                (val_g_list, type_mask, val_indices_list, val_idx_batch_mapped_list))
            logp = F.log_softmax(logits, 1)
            val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            val_loss_avg += val_loss.item()
        val_loss_avg /= val_idx_generator.num_iterations()
    val_time = time.time() - t
    print('Epoch {:05d} | Train_Loss {:.4f} | Train_Time(s) {:.4f} | Val_Loss {:.4f} | Val_Time(s) {:.4f}'.format(
            epoch, train_loss_avg, train_time, val_loss_avg, val_time))

    early_stopping(val_loss_avg, net)
    if early_stopping.early_stop:
        print('Early stopping!')
        break

print('\ntesting...')
test_idx_generator = index_generator(batch_size=batch_size, indices=test_idx, shuffle=False)
net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
net.eval()
test_embeddings = []
with torch.no_grad():
    for iteration in range(test_idx_generator.num_iterations()):
        test_idx_batch = test_idx_generator.next()
        test_g_list, test_indices_list, test_idx_batch_mapped_list = parse_minibatch(
            adjlists, edge_metapath_indices_list, test_idx_batch, device, neighbor_samples)
        mask_list, feat_keep_idx, feat_drop_idx = parse_mask(
            indices_list=test_indices_list, type_mask=type_mask, num_classes=num_class,
            src_type=src_node_type, rate=0.01, device=device)

        embs, _, embeddings, transformed_features = net((onehot_feature_list, node_type_feature),
            (adjM, features_list, mask_list, feat_keep_idx, feat_drop_idx, src_node_type),
            (test_g_list, type_mask, test_indices_list, test_idx_batch_mapped_list))
        test_embeddings.append(embeddings)

    test_embeddings = torch.cat(test_embeddings, 0)
    embeddings = test_embeddings.detach().cpu().numpy()
    transformed_features = transformed_features.cpu().numpy()

    # #vis
    # Y=labels[test_idx].cpu().numpy()
    # ml=TSNE(n_components=2)
    # node_pos = ml.fit_transform(embeddings[test_idx])
    # color_idx = {}
    # color_idx = {}
    # for i in range(3219):
    #     color_idx.setdefault(Y[i], [])
    #     color_idx[Y[i]].append(i)
    # for c, idx in color_idx.items():#c是类型数，idx是索引
    #     if str(c)=='1':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1],c='#DAA520', s=15, alpha=1)
    #     elif str(c)=='2':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1],c='#8B0000', s=15, alpha=1)
    #     elif str(c) == '0':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
    # plt.legend()
    # plt.savefig("HGAT-AC_ACM"+str(cur_repeat)+".png", dpi=1000, bbox_inches='tight')
    # plt.show()
    #=======================================================

    svm_macro, svm_micro, nmi, ari = evaluate_results_nc(embeddings[test_idx], labels[test_idx].cpu().numpy(), num_class)#使用SVM评估节点

with open('log-statistics.txt', 'a+') as f:
    f.writelines('\n' + 'Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro]) + '\n' +
                 'Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro]) + '\n')
