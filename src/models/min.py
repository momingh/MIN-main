# coding: utf-8
#
# user-graph need to be generated by the following script
# tools/generate-u-u-matrix.py
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric
import pdb
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization

class MIN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MIN, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        # self.construction = 'weighted_max'
        self.reg_weight = config['reg_weight']
        self.pvn_weight = config['pvn_weight']
        self.cl_weight = config['cl_weight']
        self.epsilon = config['epsilon']
        self.lambda1 = config['lambda1']
        self.temp = config['temp']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.id_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.mm_adj = None

        self.mlp = nn.Linear(2*dim_x, 2*dim_x)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()


        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)


        if self.v_feat is not None:
            indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.v_mm_adj = image_adj
        if self.t_feat is not None:
            indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.t_mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            del text_adj
            del image_adj



        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        _, self.uu_index = self.get_knn_uu_mat(self.edge_index)

        # pdb.set_trace()
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.weight_u2 = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u2.data = F.softmax(self.weight_u, dim=1)

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.v_mlp = MLP_layer(dim_latent=64, features=self.v_feat)
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
            self.ori_v_gcn = GCN_ori(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
            self.v_gcn_n1 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
            self.v_gcn_n2 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                                device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_mlp = MLP_layer(dim_latent=64, features=self.t_feat)
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)
            self.ori_t_gcn = GCN_ori(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)
            self.t_gcn_n1 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)
            self.t_gcn_n2 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                                device=self.device, features=self.t_feat)


        self.f_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                         device=self.device, features=self.t_feat)
        self.ori_f_gcn = GCN_ori(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                                 device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                          dim_latent=64, device=self.device, features=self.id_feat)

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)


    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def get_knn_uu_mat(self, edge_index):
        values = torch.ones(edge_index.size(1)).to(self.device)
        adjacency_matrix = torch.sparse_coo_tensor(edge_index, values, (self.num_user + self.num_item, self.num_user + self.num_item)).coalesce()
        user_item_matrix = adjacency_matrix.to_dense()[:self.num_user, self.num_user:]
        context_norm = user_item_matrix.div(torch.norm(user_item_matrix, p=2, dim=-1, keepdim=True))

        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, 2 * self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, 2 * self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def compute_normalized_laplacian_1(self, indices, n_users, n_items):
        adj_size = (indices.max().item() + 1, indices.max().item() + 1)
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, dim=-1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        normalized_laplacian = torch.sparse.FloatTensor(indices, values, adj_size)

        return normalized_laplacian

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users
        self.u_idx = user_nodes
        self.v_idx = pos_item_nodes

        self.vv_feat = self.v_mlp(self.v_feat)
        self.tt_feat = self.t_mlp(self.t_feat)

        self.vv_feat = self.ori_v_gcn(self.edge_index_dropv, self.v_mm_adj.coalesce().indices(), self.vv_feat)
        self.tt_feat = self.ori_t_gcn(self.edge_index_dropt, self.t_mm_adj.coalesce().indices(), self.tt_feat)

        # GCN for id, v, t modalities
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.vv_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.tt_feat)


        # v, v, id, and vt modalities
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)
        v_representation = torch.cat((self.v_rep, self.v_rep), dim=1)
        t_representation = torch.cat((self.t_rep, self.t_rep), dim=1)

        self.v_rep = torch.unsqueeze(self.v_rep, 2)
        self.t_rep = torch.unsqueeze(self.t_rep, 2)

        user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)


        item_rep = representation[self.num_user:]

        v_item_rep = v_representation[self.num_user:]
        t_item_rep = t_representation[self.num_user:]
        fuse_item_rep = v_item_rep + t_item_rep


        # build item-item graph
        h = self.buildItemGraph(self.mm_adj, item_rep)
        u = self.buildItemGraph(self.uu_index, user_rep)

        user_rep = user_rep + u
        item_rep = item_rep + h

        # build multimodal result embedding
        self.result_embed1 = torch.cat((user_rep, item_rep), dim=0)

        self.result_embed1 = self.ItemGraph(self.compute_normalized_laplacian_1(self.edge_index, self.num_user, self.num_item), self.result_embed1)
        self.perturbed_embeddings1 = self.ItemGraph(self.compute_normalized_laplacian_1(self.edge_index, self.num_user, self.num_item), self.result_embed1, perturbed=True)
        self.perturbed_embeddings2 = self.ItemGraph(self.compute_normalized_laplacian_1(self.edge_index, self.num_user, self.num_item), self.result_embed1, perturbed=True)

        # calculate pos and neg scores
        user_tensor1 = self.result_embed1[user_nodes]
        pos_item_tensor1 = self.result_embed1[pos_item_nodes]
        neg_item_tensor1 = self.result_embed1[neg_item_nodes]
        pos_scores1 = torch.sum(user_tensor1 * pos_item_tensor1, dim=1)
        neg_scores1 = torch.sum(user_tensor1 * neg_item_tensor1, dim=1)
        pos_neg_dis = torch.sum(pos_item_tensor1 * neg_item_tensor1, dim=1)

        pos_scores = pos_scores1
        neg_scores = neg_scores1

        closs = self.InfoNCE(user_tensor1, pos_item_tensor1, 0.2)
        return pos_scores, neg_scores, pos_neg_dis, self.cl_weight * closs

    def buildItemGraph(self, adj, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(adj, h)
        return h

    def ItemGraph(self, adj, all_embeddings, perturbed=False):
        embeddings_list = [all_embeddings]
        if adj.dtype != all_embeddings.dtype:
            adj = adj.to(h.dtype)

        for i in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            if perturbed == True:
                random_noise = torch.rand_like(all_embeddings)
                all_embeddings += torch.sign(all_embeddings) * F.normalize(random_noise, p=2, dim=-1) * self.epsilon
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings


    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores, pos_neg_dis, closs = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        L2_value = torch.mean(torch.log2(torch.sigmoid(torch.clamp(pos_neg_dis - pos_scores, 0))))

        # reg
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()

        return loss_value + reg_loss + closs + self.pvn_weight * L2_value + self.calc_cl_loss(self.perturbed_embeddings1, self.perturbed_embeddings2)

    def calc_cl_loss(self, perturbed_embeddings1, perturbed_embeddings2):
        unique_u_idx = torch.unique(self.u_idx)
        unique_v_idx = torch.unique(self.v_idx)
        p_user_emb1 = perturbed_embeddings1[unique_u_idx]
        p_user_emb2 = perturbed_embeddings2[unique_u_idx]
        p_item_emb1 = perturbed_embeddings1[unique_v_idx]
        p_item_emb2 = perturbed_embeddings2[unique_v_idx]

        normalize_emb_user1 = F.normalize(p_user_emb1, p=2, dim=1)
        normalize_emb_user2 = F.normalize(p_user_emb2, p=2, dim=1)
        normalize_emb_item1 = F.normalize(p_item_emb1, p=2, dim=1)
        normalize_emb_item2 = F.normalize(p_item_emb2, p=2, dim=1)

        pos_score_u = (normalize_emb_user1 * normalize_emb_user2).sum(dim=-1)
        pos_score_i = (normalize_emb_item1 * normalize_emb_item2).sum(dim=-1)

        ttl_score_u = torch.matmul(normalize_emb_user1, normalize_emb_user2.transpose(0, 1))
        ttl_score_i = torch.matmul(normalize_emb_item1, normalize_emb_item2.transpose(0, 1))

        pos_score_u = torch.exp(pos_score_u / 0.2)
        ttl_score_u = torch.exp(ttl_score_u / 0.2).sum(dim=1)

        pos_score_i = torch.exp(pos_score_i / 0.2)
        ttl_score_i = torch.exp(ttl_score_i / 0.2).sum(dim=1)

        cl_loss = - (torch.log(pos_score_u / ttl_score_u).mean() + torch.log(pos_score_i / ttl_score_i).mean())
        
        return self.lambda1 * cl_loss

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def full_sort_predict(self, interaction):
        user_tensor1 = self.result_embed1[:self.n_users]
        item_tensor1 = self.result_embed1[self.n_users:]

        temp_user_tensor1 = user_tensor1[interaction[0], :]
        score_matrix1 = torch.matmul(temp_user_tensor1, item_tensor1.t())
        return score_matrix1

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix


class MLP_layer(torch.nn.Module):
    def __init__(self, dim_latent, features=None):
        super(MLP_layer, self).__init__()
        self.dim_latent = dim_latent
        self.dim_feat = features.size(1)
        self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, features):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))
        return temp_features

class MLP(nn.Module):
    def __init__(self, dim_latent, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        # self.activate = torch.nn.Tanh()
        self.activate = torch.nn.LeakyReLU()
        self.mlp_1 = nn.Linear(2 * dim_latent, 4 * dim_latent)
        self.mlp_2 = nn.Linear(4 * dim_latent, 2 * dim_latent)

    def forward(self, x):
        x = self.dropout(self.activate(self.mlp_1(x)))
        x = self.dropout(self.activate(self.mlp_2(x)))
        return x


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features, perturbed=False):
        # temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        temp_features = features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h += torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h_1 += torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1
        # h_2 = self.conv_embed_1(h_1, edge_index)

        x_hat = x + h + h_1
        return x_hat, self.preference

class GCN_ori(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(GCN_ori, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device

        if self.dim_latent:
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features, perturbed=False):
        # temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        temp_features = features
        x = temp_features
        x = F.normalize(x).to(self.device)

        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h += torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h_1 += torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1
        # h_2 = self.conv_embed_1(h_1, edge_index)

        x_hat = x + h + h_1
        return x_hat


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


