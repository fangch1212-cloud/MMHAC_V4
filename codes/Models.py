import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from sklearn.cluster import KMeans
from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph

args = parse_args()


class HyperConv(nn.Module):
    """
    封装 MMHCL 原有的超图卷积逻辑：
    对输入的 Embedding 在超图邻接矩阵上进行 L 层传播。
    """

    def __init__(self, layers):
        super(HyperConv, self).__init__()
        self.layers = layers

    def forward(self, embedding, adj):
        emb = embedding
        for _ in range(self.layers):
            emb = torch.sparse.mm(adj, emb)
        # [回滚] 只返回最后一层，保证高阶语义的纯度
        return emb


class ModalityAttention(nn.Module):
    """
    [MM-HAC v2.0 核心组件] 模态注意力层
    作用：动态计算 Visual, Textual, Acoustic 等模态的权重，替代简单的 "相加" 操作。
    原理：让模型自动学会“对于这个物品，图片更重要还是文字更重要”。
    """

    def __init__(self, dim):
        super(ModalityAttention, self).__init__()
        # 使用一个简单的线性层来计算每个模态的“重要性分数”
        self.att_layer = nn.Linear(dim, 1)

    def forward(self, embeddings_list):
        """
        输入: embeddings_list (list of Tensors), 每个 Tensor 形状为 [N, Dim]
        输出: 融合后的 Tensor [N, Dim], 以及权重
        """
        if not embeddings_list:
            return None, None

        # 1. 堆叠模态特征: [N, Num_Modalities, Dim]
        # 例如: [Batch, 3, 64]
        stack = torch.stack(embeddings_list, dim=1)

        # 2. 计算原始分数: [N, Num_Modalities, 1]
        # 通过线性层将维度映射为1，代表该模态的打分
        raw_scores = self.att_layer(stack)

        # 3. Softmax 归一化: [N, Num_Modalities, 1]
        # 在模态维度(dim=1)上做 Softmax，保证权重之和为 1
        weights = F.softmax(raw_scores, dim=1)

        # 4. 加权求和: [N, Dim]
        # 利用广播机制: [N, 3, 64] * [N, 3, 1] -> Sum -> [N, 64]
        output = torch.sum(stack * weights, dim=1)

        return output, weights


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(args.UI_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings


class MM_HAC(nn.Module):
    def __init__(self, config, dataset):
        super(MM_HAC, self).__init__()
        self.config = config
        self.dataset = dataset
        self.latent_dim = config['recdim']

        # 参数设置
        self.u2u_layers = config.get('u2u_layers', 2)
        self.i2i_layers = 2
        self.ui_layers = 2  # LightGCN 骨架层数
        self.n_clusters = config.get('n_clusters', 1000)  # [新增] 聚类簇数量，Sports数据较大，设为200-500

        # 1. Embeddings
        self.embedding_user = nn.Embedding(self.dataset.n_users, self.latent_dim)
        self.embedding_item = nn.Embedding(self.dataset.n_items, self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

        # 2. 物品侧模块
        self.v_gcn = HyperConv(self.i2i_layers)
        self.t_gcn = HyperConv(self.i2i_layers)
        self.a_gcn = HyperConv(self.i2i_layers)
        self.modality_attention = ModalityAttention(self.latent_dim)

        # 3. 用户侧模块
        self.direct_interest_layer = DirectInterestLayer(self.latent_dim)
        self.common_interest_layer = HyperConv(self.u2u_layers)
        self.fusion_layer = GatedFusionLayer(self.latent_dim)

        # 4. 损失参数
        self.tau = config['temperature']
        self.ssl_reg = config['ssl_reg']
        self.modal_align_reg = config.get('modal_align_reg', 0.01)

        # [修改] 大幅降低权重，或者使用 args 传入
        self.proto_reg = config.get('proto_reg', 1e-4)  # 建议设为 0.0001 或 0.0005

        # 5. 图数据加载 (保持不变)
        self.Graph_Comb = self.dataset.get_UI_mat().cuda()
        self.UserItemNet = self.dataset.sparse_mx_to_torch_sparse_tensor(self.dataset.R).cuda()
        self.u2u_graph = self.dataset.get_U2U_mat().cuda()

        # -----------------------------------------------------------
        # [MM-HAC v3.0 核心组件] 在线原型聚类参数
        # -----------------------------------------------------------
        self.consistency_threshold = 0.5  # 一致性筛选阈值
        self.ema_momentum = 0.99  # 原型更新动量

        # 获取权重 (注意：这里使用 get 防止报错，默认值设小一点)
        self.proto_reg = config.get('proto_reg', 1e-4)

        # [关键] 注册缓冲区 (不参与反向传播)
        self.register_buffer("centroids", torch.zeros(self.n_clusters, self.latent_dim))

        # [关键] 必须初始化！否则计算 Cosine 时除以 0 会报错
        nn.init.xavier_uniform_(self.centroids)

        try:
            graphs = self.dataset.get_I2I_single_mat()
            if len(graphs) == 3:
                self.v_graph, self.t_graph, self.a_graph = graphs
                if isinstance(self.a_graph, str): self.a_graph = None
            else:
                self.v_graph, self.t_graph = graphs[:2]
                self.a_graph = None
        except Exception:
            self.v_graph = self.dataset.get_I2I_Hypergraph_mul_mat().cuda()
            self.t_graph = self.v_graph
            self.a_graph = None

        if self.v_graph is not None: self.v_graph = self.v_graph.cuda()
        if self.t_graph is not None: self.t_graph = self.t_graph.cuda()
        if self.a_graph is not None: self.a_graph = self.a_graph.cuda()

        # [新增] 聚类中心存储器
        self.cluster_centroids_struct = None
        self.cluster_centroids_modal = None
        self.item2cluster_struct = None  # 物品所属的结构簇ID
        self.item2cluster_modal = None  # 物品所属的模态簇ID

    # [步骤一] 混合聚类更新 (需要在 main.py 每个 epoch 开始时调用)
    def update_clusters(self):
        with torch.no_grad():
            # 1. 获取当前特征
            # 结构特征: LightGCN 卷积后的特征 (捕捉协同信息)
            # 模态特征: MM-HAC 融合后的特征 (捕捉语义信息)
            # 为了速度，这里简单使用 Embedding 进行聚类，或者做一次快速卷积
            struct_emb = torch.sparse.mm(self.Graph_Comb,
                                         torch.cat([self.embedding_user.weight, self.embedding_item.weight]))
            _, i_struct = torch.split(struct_emb, [self.dataset.n_users, self.dataset.n_items])

            # 获取模态特征 (需要跑一次 forward 的部分逻辑)
            modal_emb = self.embedding_item.weight
            modal_list = []
            if self.v_graph is not None: modal_list.append(self.v_gcn(modal_emb, self.v_graph))
            if self.t_graph is not None: modal_list.append(self.t_gcn(modal_emb, self.t_graph))
            if len(modal_list) > 0:
                modal_emb, _ = self.modality_attention(modal_list)

            # 转为 numpy 用于 sklearn
            i_struct_np = F.normalize(i_struct, p=2, dim=1).cpu().numpy()
            modal_emb_np = F.normalize(modal_emb, p=2, dim=1).cpu().numpy()

            # 2. 执行 K-Means (混合聚类)
            kmeans_s = KMeans(n_clusters=self.n_clusters, n_init=1).fit(i_struct_np)
            kmeans_m = KMeans(n_clusters=self.n_clusters, n_init=1).fit(modal_emb_np)

            # 3. 存储结果到 GPU
            self.cluster_centroids_struct = torch.tensor(kmeans_s.cluster_centers_).cuda()
            self.cluster_centroids_modal = torch.tensor(kmeans_m.cluster_centers_).cuda()
            self.item2cluster_struct = torch.tensor(kmeans_s.labels_).cuda()
            self.item2cluster_modal = torch.tensor(kmeans_m.labels_).cuda()

            # print("  [Clustering] Updated hybrid clusters.")

    def forward(self):
        # 1. LightGCN Backbone (结构视图)
        ego_embeddings = torch.cat((self.embedding_user.weight, self.embedding_item.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for k in range(self.ui_layers):
            ego_embeddings = torch.sparse.mm(self.Graph_Comb, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_cf, i_g_cf = torch.split(all_embeddings, [self.dataset.n_users, self.dataset.n_items], dim=0)

        # 2. MM-HAC Enhancement (模态视图)
        modal_embeddings_list = []
        if self.v_graph is not None: modal_embeddings_list.append(self.v_gcn(self.embedding_item.weight, self.v_graph))
        if self.t_graph is not None: modal_embeddings_list.append(self.t_gcn(self.embedding_item.weight, self.t_graph))
        if self.a_graph is not None and hasattr(self.a_graph, 'shape'): modal_embeddings_list.append(
            self.a_gcn(self.embedding_item.weight, self.a_graph))

        if len(modal_embeddings_list) > 0:
            h_i_hac, _ = self.modality_attention(modal_embeddings_list)
        else:
            h_i_hac = self.embedding_item.weight

        # 3. User Interest
        item_input = h_i_hac + self.embedding_item.weight
        h_u_direct = self.direct_interest_layer(self.embedding_user.weight, item_input, self.UserItemNet)
        h_u_common = self.common_interest_layer(self.embedding_user.weight, self.u2u_graph)
        h_u_hac = self.fusion_layer(h_u_direct, h_u_common)

        # 4. Final Fusion
        final_u = u_g_cf + F.normalize(h_u_hac, p=2, dim=1)
        final_i = i_g_cf + F.normalize(h_i_hac, p=2, dim=1)

        return final_u, final_i, u_g_cf, i_g_cf, h_i_hac


    def calc_loss(self, users, pos_items, neg_items):
        """
        [Fix] 修复 RuntimeError: inplace operation 报错
        核心修正：在计算 loss 前 detach 原型中心，防止 EMA 更新干扰反向传播。
        """
        device = self.embedding_user.weight.device

        # 1. 数据转换
        if not torch.is_tensor(users): users = torch.LongTensor(users).to(device)
        if not torch.is_tensor(pos_items): pos_items = torch.LongTensor(pos_items).to(device)
        if not torch.is_tensor(neg_items): neg_items = torch.LongTensor(neg_items).to(device)

        # 2. 前向传播
        final_u, final_i, u_cf, i_cf, h_i_mm_raw = self.forward()
        h_i_mm = F.normalize(h_i_mm_raw, p=2, dim=1)

        users_unique = torch.unique(users)
        items_unique = torch.unique(pos_items)

        # --- A: 结构-模态一致性计算 ---
        batch_struct_emb = F.normalize(i_cf[items_unique], p=2, dim=1)
        batch_modal_emb = h_i_mm[items_unique]
        consistency = F.cosine_similarity(batch_struct_emb, batch_modal_emb, dim=1)
        conf_mask = (consistency > self.consistency_threshold).float()
        soft_weights = torch.sigmoid(consistency * 5.0).detach()

        # --- B: 在线原型聚类与更新 ---

        # [关键修复 1] 必须 detach!
        # 告诉 PyTorch: "这里用的 centroids 是常量，不需要对它求导，也不要追踪它的版本"
        curr_centroids = F.normalize(self.centroids.detach(), p=2, dim=1)

        sim_to_centroids = torch.matmul(batch_modal_emb, curr_centroids.T)
        _, cluster_ids = sim_to_centroids.max(dim=1)

        # [关键修复 2] EMA 更新放入 no_grad，防止任何意外的梯度追踪
        if self.training:
            with torch.no_grad():
                valid_indices = torch.nonzero(conf_mask).squeeze()
                # 兼容 valid_indices 只有一个元素变成 0-d tensor 的情况
                if valid_indices.dim() == 0 and valid_indices.numel() == 1:
                    valid_indices = valid_indices.unsqueeze(0)

                if valid_indices.numel() > 0:
                    valid_items_emb = batch_modal_emb[valid_indices]
                    valid_cluster_ids = cluster_ids[valid_indices]

                    unique_c_ids = torch.unique(valid_cluster_ids)
                    for c_id in unique_c_ids:
                        selected = (valid_cluster_ids == c_id)
                        if selected.sum() > 0:
                            cluster_mean = valid_items_emb[selected].mean(dim=0)
                            # In-place 修改: 现在安全了，因为前面已经 detach 了
                            self.centroids[c_id] = self.ema_momentum * self.centroids[c_id] + \
                                                   (1 - self.ema_momentum) * cluster_mean

        # --- C: 损失计算 ---

        # 1. BPR
        batch_u = final_u[users]
        batch_pos = final_i[pos_items]
        batch_neg = final_i[neg_items]
        pos_scores = torch.mul(batch_u, batch_pos).sum(dim=1)
        neg_scores = torch.mul(batch_u, batch_neg).sum(dim=1)
        loss_bpr = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        # 2. Proto Loss
        pos_proto_sim = sim_to_centroids[torch.arange(len(items_unique)), cluster_ids]
        all_proto_sim = torch.logsumexp(sim_to_centroids / self.tau, dim=1)
        loss_proto_per_item = all_proto_sim - (pos_proto_sim / self.tau)

        if conf_mask.sum() > 0:
            loss_proto = (loss_proto_per_item * conf_mask).sum() / (conf_mask.sum() + 1e-9)
        else:
            loss_proto = torch.tensor(0.0).to(device)
        loss_proto = loss_proto * self.proto_reg

        # 3. SCL Loss
        loss_cl_i = self.cal_cl_loss([i_cf[items_unique], final_i[items_unique]], weights=soft_weights)
        loss_cl_u = self.cal_cl_loss([u_cf[users_unique], final_u[users_unique]])
        loss_scl = self.config['ssl_reg'] * (loss_cl_u + loss_cl_i)

        # 4. Modal Loss
        loss_modal = self.cal_cl_loss([i_cf[items_unique], h_i_mm[items_unique]], weights=soft_weights)
        loss_modal = loss_modal * self.modal_align_reg

        # 5. Reg Loss
        loss_reg = self.config['reg'] * (1 / 2) * (self.embedding_user.weight.norm(2).pow(2) +
                                                   self.embedding_item.weight.norm(2).pow(2)) / float(len(users))

        total_loss = loss_bpr + loss_scl + loss_modal + loss_proto + loss_reg

        return total_loss, loss_bpr, loss_scl, loss_proto, loss_reg



    def cal_cl_loss(self, views, weights=None):
        z1, z2 = views
        norm_z1 = F.normalize(z1, p=2, dim=1)
        norm_z2 = F.normalize(z2, p=2, dim=1)
        pos_score = (norm_z1 * norm_z2).sum(dim=1)
        ttl_score = torch.matmul(norm_z1, norm_z2.transpose(0, 1))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.sum(torch.exp(ttl_score / self.tau), dim=1)
        loss = -torch.log(pos_score / ttl_score)

        if weights is not None:
            loss = loss * weights

        return loss.mean()

    # def cal_cl_loss(self, views, weights=None):
    #     z1, z2 = views
    #     norm_z1 = F.normalize(z1, p=2, dim=1)
    #     norm_z2 = F.normalize(z2, p=2, dim=1)
    #     pos_score = (norm_z1 * norm_z2).sum(dim=1)
    #     ttl_score = torch.matmul(norm_z1, norm_z2.transpose(0, 1))
    #     pos_score = torch.exp(pos_score / self.tau)
    #     ttl_score = torch.sum(torch.exp(ttl_score / self.tau), dim=1)
    #     loss = -torch.log(pos_score / ttl_score)
    #     if weights is not None:
    #         loss = loss * weights
    #     return loss.mean()


# class MM_HAC(nn.Module):
#     def __init__(self, config, dataset):
#         super(MM_HAC, self).__init__()
#         self.config = config
#         self.dataset = dataset
#         self.latent_dim = config['recdim']

#         self.u2u_layers = config.get('layer', 2)
#         self.i2i_layers = 2
#         # [新增] LightGCN 骨架层数 (通常 2 或 3)
#         self.ui_layers = config.get('ui_layers', 2)

#         # 1. 初始化 Embeddings
#         self.embedding_user = nn.Embedding(self.dataset.n_users, self.latent_dim)
#         self.embedding_item = nn.Embedding(self.dataset.n_items, self.latent_dim)
#         nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#         nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)

#         # 2. 物品侧创新 (HAC)
#         self.v_gcn = HyperConv(self.i2i_layers)
#         self.t_gcn = HyperConv(self.i2i_layers)
#         self.a_gcn = HyperConv(self.i2i_layers)
#         self.modality_attention = ModalityAttention(self.latent_dim)

#         # 3. 用户侧创新 (HAC)
#         self.direct_interest_layer = DirectInterestLayer(self.latent_dim)
#         self.common_interest_layer = HyperConv(self.u2u_layers)
#         self.fusion_layer = GatedFusionLayer(self.latent_dim)

#         # 4. 参数
#         self.tau = config['temperature']
#         self.modal_align_reg = config.get('modal_align_reg', 0.01)

#         # 5. [关键修复] 加载数据
#         # (A) LightGCN 需要的 N+M 大图 (用于协同过滤骨架)
#         # 您之前修复过 get_UI_mat 的 OOM 问题，这里直接调用
#         self.Graph_Comb = self.dataset.get_UI_mat().cuda()

#         # (B) MM-HAC 需要的组件图
#         self.UserItemNet = self.dataset.sparse_mx_to_torch_sparse_tensor(self.dataset.R).cuda()
#         self.u2u_graph = self.dataset.get_U2U_mat().cuda()


#         try:
#             graphs = self.dataset.get_I2I_single_mat()
#             if len(graphs) == 3:
#                 self.v_graph, self.t_graph, self.a_graph = graphs
#                 if isinstance(self.a_graph, str): self.a_graph = None
#             else:
#                 self.v_graph, self.t_graph = graphs[:2]
#                 self.a_graph = None
#         except Exception:
#             # Fallback
#             self.v_graph = self.dataset.get_I2I_Hypergraph_mul_mat().cuda()
#             self.t_graph = self.v_graph
#             self.a_graph = None

#         if self.v_graph is not None: self.v_graph = self.v_graph.cuda()
#         if self.t_graph is not None: self.t_graph = self.t_graph.cuda()
#         if self.a_graph is not None: self.a_graph = self.a_graph.cuda()

#     def forward(self):
#         # ================= 1. LightGCN Backbone (找回缺失的脊梁) =================
#         # 这是捕捉协同信号的关键，之前的版本缺失了这部分
#         ego_embeddings = torch.cat((self.embedding_user.weight, self.embedding_item.weight), dim=0)
#         all_embeddings = [ego_embeddings]

#         for k in range(self.ui_layers):
#             ego_embeddings = torch.sparse.mm(self.Graph_Comb, ego_embeddings)
#             all_embeddings.append(ego_embeddings)

#         # Mean Pooling 聚合
#         all_embeddings = torch.stack(all_embeddings, dim=1)
#         all_embeddings = torch.mean(all_embeddings, dim=1)

#         # 分离出 User 和 Item 的 CF 特征
#         u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.dataset.n_users, self.dataset.n_items],
#                                                      dim=0)

#         # ================= 2. MM-HAC Feature Enhancement (创新增强) =================

#         # --- Item Part ---
#         modal_embeddings_list = []
#         if self.v_graph is not None:
#             modal_embeddings_list.append(self.v_gcn(self.embedding_item.weight, self.v_graph))
#         if self.t_graph is not None:
#             modal_embeddings_list.append(self.t_gcn(self.embedding_item.weight, self.t_graph))
#         if self.a_graph is not None and hasattr(self.a_graph, 'shape'):
#             modal_embeddings_list.append(self.a_gcn(self.embedding_item.weight, self.a_graph))

#         if len(modal_embeddings_list) > 0:
#             h_i_hac, _ = self.modality_attention(modal_embeddings_list)
#         else:
#             h_i_hac = torch.zeros_like(self.embedding_item.weight)

#         # --- User Part ---
#         # 注意：这里我们使用 h_i_hac (多模态特征) 来增强用户表示
#         # 同时也传入 h_i_hac + id 确保信息流
#         item_input_for_user = h_i_hac + self.embedding_item.weight
#         h_u_direct = self.direct_interest_layer(self.embedding_user.weight, item_input_for_user, self.UserItemNet)
#         h_u_common = self.common_interest_layer(self.embedding_user.weight, self.u2u_graph)
#         h_u_hac = self.fusion_layer(h_u_direct, h_u_common)

#         # ================= 3. Final Fusion (骨架 + 增强) =================
#         # 将 LightGCN 的协同特征 与 MM-HAC 的语义/结构特征 相加

#         final_u = u_g_embeddings + F.normalize(h_u_hac, p=2, dim=1)
#         final_i = i_g_embeddings + F.normalize(h_i_hac, p=2, dim=1)

#         # 返回: 用户终态, 物品终态, 用户ID(用于CL), 物品ID(用于CL), 模态列表(用于CL)
#         return final_u, final_i, u_g_embeddings, i_g_embeddings, modal_embeddings_list

#     def calc_loss(self, users, pos_items, neg_items):
#         # 1. 数据准备
#         device = self.embedding_user.weight.device
#         if not torch.is_tensor(users): users = torch.LongTensor(users).to(device)
#         if not torch.is_tensor(pos_items): pos_items = torch.LongTensor(pos_items).to(device)
#         if not torch.is_tensor(neg_items): neg_items = torch.LongTensor(neg_items).to(device)

#         # 2. 获取【纯净】特征用于 BPR (主任务)
#         final_u, final_i, u_g_cf, i_g_cf, modal_list = self.forward()

#         # BPR Loss Calculation
#         batch_u = final_u[users]
#         batch_pos = final_i[pos_items]
#         batch_neg = final_i[neg_items]
#         pos_scores = torch.mul(batch_u, batch_pos).sum(dim=1)
#         neg_scores = torch.mul(batch_u, batch_neg).sum(dim=1)
#         loss_bpr = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

#         # 3. 获取【加噪】特征用于 Contrastive Learning (辅助任务)
#         # [核心改进] SimGCL 策略: 在 Embedding 上加均匀分布噪声
#         noise_eps = self.config.get('eps', 0.1)

#         def perturb(emb):
#             noise = torch.rand_like(emb).to(device) # [0, 1)
#             noise = (noise * 2 - 1) * noise_eps     # [-eps, eps]
#             return emb + noise

#         # 对用于对比的特征加噪声
#         # 注意：这里我们简单地对最终输出加噪声，这是一种高效的近似
#         final_u_noise = perturb(final_u)
#         final_i_noise = perturb(final_i)

#         # 4. Structural SCL (User-side & Item-side)
#         users_unique = torch.unique(users)
#         items_unique = torch.unique(pos_items)

#         # 对比：加噪后的融合特征 vs 纯净的ID特征 (或者也给ID加噪，这里保持简单)
#         loss_cl_u = self.cal_cl_loss([u_g_cf[users_unique], final_u_noise[users_unique]])
#         loss_cl_i = self.cal_cl_loss([i_g_cf[items_unique], final_i_noise[items_unique]])
#         loss_cl_struct = self.config['ssl_reg'] * (loss_cl_u + loss_cl_i)

#         # 5. Modal SCL (跨模态对比)
#         loss_modal = torch.tensor(0.0).to(device)
#         if len(modal_list) >= 2:
#             # 给模态特征也加噪声，增加难度
#             m1 = perturb(modal_list[0][items_unique])
#             m2 = perturb(modal_list[1][items_unique])
#             loss_modal = self.modal_align_reg * self.cal_cl_loss([m1, m2])

#         # 6. Regularization
#         reg_loss = (1/2)*(self.embedding_user.weight.norm(2).pow(2) +
#                           self.embedding_item.weight.norm(2).pow(2))/float(len(users))
#         loss_reg = self.config['reg'] * reg_loss

#         loss_cl_total = loss_cl_struct + loss_modal
#         total_loss = loss_bpr + loss_cl_total + loss_reg

#         return total_loss, loss_bpr, loss_cl_total, loss_modal, loss_reg

#     def cal_cl_loss(self, views):
#         z1, z2 = views
#         norm_z1 = F.normalize(z1, p=2, dim=1)
#         norm_z2 = F.normalize(z2, p=2, dim=1)
#         pos_score = (norm_z1 * norm_z2).sum(dim=1)
#         ttl_score = torch.matmul(norm_z1, norm_z2.transpose(0, 1))
#         pos_score = torch.exp(pos_score / self.tau)
#         ttl_score = torch.sum(torch.exp(ttl_score / self.tau), dim=1)
#         loss = -torch.log(pos_score / ttl_score)
#         return loss.mean()

class MMHCL(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super(MMHCL, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embeddings_dim = embedding_dim

        self.user_ui_embedding = nn.Embedding(n_users, self.embeddings_dim)  # 用户UI嵌入
        self.item_ui_embedding = nn.Embedding(n_items, self.embeddings_dim)  # 物品UI嵌入

        self.uu_embedding = nn.Embedding(n_users, self.embeddings_dim)  # 用户超图嵌入
        self.ii_embedding = nn.Embedding(n_items, self.embeddings_dim)  # 物品超图嵌入

        if args.cf_model == 'NGCF':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(args.UI_layers):
                self.GC_Linear_list.append(nn.Linear(eval(args.weight_size)[i], eval(args.weight_size)[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(eval(args.weight_size)[i], eval(args.weight_size)[i + 1]))
                self.dropout_list.append(nn.Dropout(0.1))

        nn.init.xavier_uniform_(self.user_ui_embedding.weight)
        nn.init.xavier_uniform_(self.item_ui_embedding.weight)
        nn.init.xavier_uniform_(self.uu_embedding.weight)
        nn.init.xavier_uniform_(self.ii_embedding.weight)

        self.tau = args.temperature

    def forward(self, UI_mat, I2I_mat, U2U_mat):

        ''' 1.超图嵌入传播 '''
        ii_emb = self.ii_embedding.weight
        uu_emb = self.uu_embedding.weight

        if args.item_loss_ratio != 0:
            # 物品超图卷积
            for i in range(args.Item_layers):
                ii_emb = torch.sparse.mm(I2I_mat, ii_emb)

        if args.user_loss_ratio != 0:
            # 用户超图卷积
            for i in range(args.User_layers):
                uu_emb = torch.sparse.mm(U2U_mat, uu_emb)

        if args.cf_model == 'LightGCN':
            ''' 2.主干网络（LightGCN）嵌入传播 '''
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(args.UI_layers):
                side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        elif args.cf_model == 'NGCF':
            ego_embeddings = torch.cat((self.user_ui_embedding.weight, self.item_ui_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(args.UI_layers):
                side_embeddings = torch.sparse.mm(UI_mat, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_ui_emb, i_ui_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        elif args.cf_model == 'MF':
            u_ui_emb, i_ui_emb = self.user_ui_embedding.weight, self.item_ui_embedding.weight

        ''' 3.融合：一阶信号 + 二阶语义 '''
        if args.item_loss_ratio != 0:
            i_ui_emb = i_ui_emb + F.normalize(ii_emb, p=2, dim=1)

        if args.user_loss_ratio != 0:
            u_ui_emb = u_ui_emb + F.normalize(uu_emb, p=2, dim=1)

        return u_ui_emb, i_ui_emb, ii_emb, uu_emb

    ''' 对比学习损失 '''

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        # z1：超图嵌入； z2：融合嵌入（或反之）
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]  同视图相似度
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]  跨视图相似度

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


class DirectInterestLayer(nn.Module):
    """
    改进版：移除易过拟合的 Attention 参数，改用 GCN 归一化聚合。
    这在稀疏数据集（如 Tiktok）上更稳健。
    """

    def __init__(self, dim):  # 移除了 n_heads 参数
        super(DirectInterestLayer, self).__init__()
        self.dim = dim
        # [修改] 移除可学习的 W 和 a，减少参数量
        # self.W = nn.Linear(dim, dim)
        # self.a = nn.Linear(2 * dim, 1)

        # 可选：仅保留一个线性变换用于特征对齐
        self.trans = nn.Linear(dim, dim)

    def forward(self, user_emb, item_emb, adj_matrix):
        """
        user_emb: [n_users, dim]
        item_emb: [n_items, dim] (多模态增强后的物品表示)
        adj_matrix: [n_users, n_items] 原始交互矩阵 (Sparse Tensor)
        """
        # 1. 特征变换 (可选，也可以去掉)
        i_feat = self.trans(item_emb)

        # 2. GCN 风格聚合： h_u = Sum( A_ui * h_i ) / sqrt(D_u * D_i)
        # 由于我们传入的 adj_matrix 是 R (0/1矩阵)，我们需要手动做归一化，或者简单实用平均聚合

        # 这里实现最稳健的 "Mean Pooling" 或 "Degree Normalized"
        # 简单实现：直接利用稀疏矩阵乘法聚合，然后除以度

        # h_u_direct = A * h_i
        # adj_matrix shape: [n_users, n_items]
        h_u_direct = torch.sparse.mm(adj_matrix, i_feat)

        # 3. 归一化 (Mean Pooling)
        # 计算每个用户的度 (交互数量)
        # adj_matrix 是稀疏的，to_dense() 太大，我们用 indices 计算度
        indices = adj_matrix._indices()
        row_indices = indices[0]

        # 计算度: degree[u]
        degree = torch.zeros(user_emb.size(0)).to(user_emb.device)
        ones = torch.ones(indices.size(1)).to(user_emb.device)
        degree.index_add_(0, row_indices, ones)

        # 加上 epsilon 避免除零
        degree = degree.unsqueeze(1) + 1e-9

        # 归一化
        h_u_direct = h_u_direct / degree

        # 4. 残差连接
        h_u_direct = h_u_direct + user_emb

        return h_u_direct


class GatedFusionLayer(nn.Module):
    def __init__(self, dim):
        super(GatedFusionLayer, self).__init__()
        self.W = nn.Linear(dim, dim)
        self.d = nn.Linear(2 * dim, 1)  # 对应论文中的 vector d
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, h_direct, h_common):
        # 投影
        h_direct_proj = self.W(h_direct)
        h_common_proj = self.W(h_common)

        # 计算 e' 和 e'' (Eq. 9, 10)
        # 注意：HAS-HGNN 原文是拼接后乘向量 d^T
        cat_1 = torch.cat((h_direct_proj, h_common_proj), dim=1)
        cat_2 = torch.cat((h_common_proj, h_direct_proj), dim=1)

        e_prime = self.leakyrelu(self.d(cat_1))
        e_double_prime = self.leakyrelu(self.d(cat_2))

        # 计算 softmax 权重 alpha (Eq. 11, 12)
        # 拼接以便在维度1上做softmax
        scores = torch.cat((e_prime, e_double_prime), dim=1)
        alphas = F.softmax(scores, dim=1)

        alpha_prime = alphas[:, 0].unsqueeze(1)
        alpha_double_prime = alphas[:, 1].unsqueeze(1)

        # 加权融合 (Eq. 13)
        h_final = alpha_prime * h_direct + alpha_double_prime * h_common
        return h_final