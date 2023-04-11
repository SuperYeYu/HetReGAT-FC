import torch.nn as nn
from dgl.nn.pytorch import edge_softmax, GATConv
import torch
import torch.nn.functional as F
import numpy as np
from model.FC import FC

class HGAT(nn.Module):
    def __init__(self,g,in_dims,in_dims_2,num_hidden,num_classes,num_layers,heads,activation,feat_drop,attn_drop,negative_slope,residual):
        super(HGAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.hgat_layers = nn.ModuleList()
        self.activation = activation

        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])

        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.ntfc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims_2])
        for ntfc in self.ntfc_list:
            nn.init.xavier_normal_(ntfc.weight, gain=1.414)

        self.hgat_layers.append(GATConv(num_hidden*2, num_hidden, heads[0],feat_drop, attn_drop, negative_slope, False, self.activation))

        for l in range(1, num_layers):
            self.hgat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l],feat_drop, attn_drop, negative_slope, residual, self.activation))

        self.hgat_layers.append(GATConv(num_hidden * heads[-2], num_hidden, heads[-1],feat_drop, attn_drop, negative_slope, residual, None))
        self.lines=nn.Linear(num_hidden,num_classes,bias=True)
        nn.init.xavier_normal_(self.lines.weight, gain=1.414)

    def forward(self, features_list,node_type_feature):
        h = []
        h2 = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        for ntfc, feature in zip(self.ntfc_list, node_type_feature):
            h2.append(ntfc(feature))
        h2 = torch.cat(h2, 0)
        h = torch.cat((h,h2),1)
        for l in range(self.num_layers):
            h = self.hgat_layers[l](self.g, h).flatten(1)
        h = self.hgat_layers[-1](self.g, h).mean(1)
        logits = self.lines(h)
        return logits, h


class HeReGAT_nc_FC(nn.Module):
    def __init__(self,g,in_dim_1,in_dim_2,in_dim_3,hidden_dim,num_class,
                 num_layer_1,num_layer_2,num_heads,f_drop,att_drop,activation,
                 slope,res,dropout_rate=0.5,cuda=False,feat_opt=None):
        super(HeReGAT_nc_FC, self).__init__()

        self.feat_opt = feat_opt
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in in_dim_3])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        heads = [num_heads] * num_layer_1 + [1]
        self.layer1 = HGAT(g,in_dim_1,in_dim_2,hidden_dim,num_class,
                           num_layer_1,heads,activation,f_drop,att_drop,slope,res)

        self.hgn_FC = FC(in_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout_rate,
                               activation=F.elu, num_heads=num_heads, cuda=cuda)

        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x

        heads = [num_heads] * num_layer_2 + [1]
        in_dim_4 = [hidden_dim for num in range(num_class)]
        self.layer3 = HGAT(g,in_dim_4,in_dim_2,hidden_dim,num_class,
                           num_layer_2,heads, activation,f_drop,att_drop,slope,res)

    def forward(self, inputs1, inputs2, inputs3):

        onehot_feature_list, node_type_feature = inputs1

        adj, feat_list, mask_list, feat_keep_idx, feat_drop_idx, node_type_src = inputs2

        g_list, type_mask, edge_metapath_indices_list, target_idx_list = inputs3

        logits_1, emb = self.layer1(onehot_feature_list,node_type_feature)

        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=adj.device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(feat_list[i])
        feat_src = transformed_features

        for i, opt in enumerate(self.feat_opt):
            if opt == 1:
                feat_ac = self.hgn_FC(adj[mask_list[i]][:, mask_list[node_type_src]],
                                       emb[mask_list[i]], emb[mask_list[node_type_src]],
                                       feat_src[mask_list[node_type_src]])
                transformed_features[mask_list[i]] = feat_ac
        transformed_features = self.feat_drop(transformed_features)

        node_len = []
        transformed_feature = []
        for i in range(len(onehot_feature_list)):
            node_len.append(len(onehot_feature_list[i]))
        a, b, c = transformed_features.split(node_len,dim=0)
        transformed_feature.append(a)
        transformed_feature.append(b)
        transformed_feature.append(c)
        logits_2, h_representation = self.layer3(transformed_feature, node_type_feature)

        return emb,logits_2, h_representation, transformed_features