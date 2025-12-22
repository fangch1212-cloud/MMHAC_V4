import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import json
from codes.utility.parser import parse_args

args = parse_args()

import torch



class Data(object):
    def __init__(self, path, batch_size):
        self.path = path + '/%d-core' % args.core
        self.batch_size = batch_size

        train_file = path + '/%d-core/train.json' % (args.core)
        val_file = path + '/%d-core/val.json' % (args.core)
        test_file = path + '/%d-core/test.json' % (args.core)

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test, self.n_val = 0, 0, 0
        self.neg_pools = {}

        self.exist_users = []

        train = json.load(open(train_file))
        test = json.load(open(test_file))
        val = json.load(open(val_file))
        for uid, items in train.items():
            if len(items) == 0:
                continue
            uid = int(uid)
            self.exist_users.append(uid)
            self.n_items = max(self.n_items, max(items))
            self.n_users = max(self.n_users, uid)
            self.n_train += len(items)

        for uid, items in test.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_test += len(items)
            except:
                continue

        for uid, items in val.items():
            uid = int(uid)
            try:
                self.n_items = max(self.n_items, max(items))
                self.n_val += len(items)
            except:
                continue

        self.n_items += 1
        self.n_users += 1

        self.print_statistics()

        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R_Item_Interacts = sp.dok_matrix((self.n_items, self.n_items), dtype=np.float32)

        self.train_items, self.test_set, self.val_set = {}, {}, {}
        for uid, train_items in train.items():
            if len(train_items) == 0:
                continue
            uid = int(uid)
            for idx, i in enumerate(train_items):
                self.R[uid, i] = 1.

            self.train_items[uid] = train_items

        for uid, test_items in test.items():
            uid = int(uid)
            if len(test_items) == 0:
                continue
            try:
                self.test_set[uid] = test_items
            except:
                continue

        for uid, val_items in val.items():
            uid = int(uid)
            if len(val_items) == 0:
                continue
            try:
                self.val_set[uid] = val_items
            except:
                continue

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)

        # [修改] 使用 torch.sparse_coo_tensor 替代过时的 FloatTensor
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (
            self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        # users = self.exist_users[:]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)
            # neg_items += sample_neg_items_for_u(u, 3)
        return users, pos_items, neg_items

    # 原始的Latiice
    # general Model 返回的是numpy类型的，度归一化用-1
    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def get_D_inv(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            return d_mat_inv

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    # ---------------------------------------Own--------------------------------------------------

    def norm_dense(self, adj, normalization='origin'):
        if normalization == 'sym':
            rowsum = torch.sum(adj, -1)
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
            L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        elif normalization == "2sym":
            rowsum = torch.sum(adj, -1)
            d_row_inv_sqrt = torch.pow(rowsum, -0.5)
            d_row_inv_sqrt[torch.isinf(d_row_inv_sqrt)] = 0.
            d_row_mat_inv_sqrt = torch.diagflat(d_row_inv_sqrt)

            colsum = torch.sum(adj, -2)
            d_col_inv_sqrt = torch.pow(colsum, -0.5)
            d_col_inv_sqrt[torch.isinf(d_col_inv_sqrt)] = 0.
            d_col_mat_inv_sqrt = torch.diagflat(d_col_inv_sqrt)

            L_norm = torch.mm(torch.mm(d_row_mat_inv_sqrt, adj), d_col_mat_inv_sqrt)

        elif normalization == 'rw':
            rowsum = torch.sum(adj, -1)
            d_inv = torch.pow(rowsum, -1)
            d_inv[torch.isinf(d_inv)] = 0.
            d_mat_inv = torch.diagflat(d_inv)
            L_norm = torch.mm(d_mat_inv, adj)
        elif normalization == 'origin':
            L_norm = adj
        return L_norm

    # ============================================================================
    # [新增] 内存安全的核心工具函数 (请添加到 Data 类中)
    # ============================================================================
    def build_knn_sparse_batch(self, features, topk, batch_size=2048):
        """
        分块构建 KNN 图，避免 OOM。
        """
        n_nodes = features.shape[0]
        # 归一化特征
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        rows, cols, data = [], [], []

        # 如果显存不够，可以把 features 转到 cpu: features = features.cpu()
        start = 0
        while start < n_nodes:
            end = min(start + batch_size, n_nodes)
            # 1. 计算当前 Batch 的相似度 [Batch, N]
            batch_feats = features[start:end]
            sim_batch = torch.mm(batch_feats, features.t())

            # 2. 取 Top-K
            # vals: [Batch, K], inds: [Batch, K]
            knn_val, knn_ind = torch.topk(sim_batch, topk, dim=-1)

            # 3. 构造稀疏坐标
            row_idx = torch.arange(start, end).view(-1, 1).expand(-1, topk).flatten()
            col_idx = knn_ind.flatten()

            rows.append(row_idx.cpu().numpy())
            cols.append(col_idx.cpu().numpy())
            data.append(np.ones(len(row_idx)))  # Unweighted graph

            start += batch_size
            del sim_batch, knn_val, knn_ind  # 及时释放内存

        # 4. 合并所有 Batch
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)

        # 5. 构建 Scipy 稀疏矩阵
        adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
        return adj

    def norm_sparse(self, adj, norm_type='origin'):
        """
        稀疏矩阵归一化
        """
        if norm_type == 'origin':
            return adj

        adj = adj.tocsr()
        if norm_type == 'sym':
            rowsum = np.array(adj.sum(1))
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
            norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        elif norm_type == 'rw':
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
        else:
            raise NotImplementedError(f"Norm type {norm_type} not implemented for sparse.")
        return norm_adj.tocoo()

    # ============================================================================
    # [替换] 下面两个函数替换原来的同名函数
    # ============================================================================
    def get_I2I_Hypergrah_mat(self, norm_type="origin"):
        # [修改] 使用内存高效的方式构建多模态超图
        print(f"Loading I2I multi-media Hypergraph mat:({norm_type})_topk:{str(args.topk)}")
        t = time()
        try:
            Hypergraph = torch.load(f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")
        except Exception:
            print("Generating Hypergraph from scratch (Batch Mode)...")

            image_feats = np.load(f'../data/{args.dataset}/image_feat.npy')
            text_feats = np.load(f'../data/{args.dataset}/text_feat.npy')
            image_feats = torch.tensor(image_feats).float()
            text_feats = torch.tensor(text_feats).float()

            # [关键] 调用分块构建
            image_adj = self.build_knn_sparse_batch(image_feats, topk=args.topk)
            text_adj = self.build_knn_sparse_batch(text_feats, topk=args.topk)

            # 稀疏拼接
            Hypergraph = sp.hstack([image_adj, text_adj])
            Hypergraph = self.norm_sparse(Hypergraph, norm_type)

            # 转 Torch Sparse
            indices = torch.from_numpy(np.vstack((Hypergraph.row, Hypergraph.col)).astype(np.int64))
            values = torch.from_numpy(Hypergraph.data).float()
            shape = torch.Size(Hypergraph.shape)
            Hypergraph = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

            torch.save(Hypergraph, f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")

        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_I2I_Hypergraph_mul_mat(self, norm_type="sym"):
        # [修改] 全程稀疏计算 H * H^T
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_topk:{str(args.topk)}")
        t = time()
        try:
            Hypergraph_mul = torch.load(f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
        except Exception:
            print("Generating Hypergraph_mul from scratch (Sparse Mode)...")

            # 1. 获取 H (Torch Sparse)
            H_torch = self.get_I2I_Hypergrah_mat("origin")

            # 2. 转 Scipy Sparse
            H_indices = H_torch.coalesce().indices().cpu().numpy()
            H_values = H_torch.coalesce().values().cpu().numpy()
            H_shape = H_torch.size()
            H_scipy = sp.coo_matrix((H_values, (H_indices[0], H_indices[1])), shape=H_shape)

            # 3. 稀疏矩阵乘法 (内存极其高效)
            H_csr = H_scipy.tocsr()
            H_mul_scipy = H_csr.dot(H_csr.transpose())

            # 4. 归一化
            H_mul_scipy = self.norm_sparse(H_mul_scipy, norm_type)

            # 5. 转回 Torch Sparse
            H_mul_coo = H_mul_scipy.tocoo()
            indices = torch.from_numpy(np.vstack((H_mul_coo.row, H_mul_coo.col)).astype(np.int64))
            values = torch.from_numpy(H_mul_coo.data).float()
            shape = torch.Size(H_mul_coo.shape)

            Hypergraph_mul = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

            torch.save(Hypergraph_mul, f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")

        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul

    def get_UI_mat(self, norm_type='sym'):
        """
        [Fix] 针对大规模数据集（如 Sports/Clothing）优化内存。
        使用 scipy.sparse 进行稀疏矩阵归一化，避免 .todense() 导致的 11GB+ 内存溢出。
        """
        print("Loading UI_mat:(" + norm_type + ")")
        t = time()

        # 尝试加载缓存
        try:
            # 尝试加载已保存的稀疏张量
            UI_mat = torch.load(self.path + '/UI_mat_' + norm_type + ".pth")
        except Exception:
            print(f"Generating UI_mat from scratch (Sparse Mode)...")
            # 1. 构建基础邻接矩阵 (dok_matrix -> csr_matrix)
            # 矩阵大小: (users + items) x (users + items)
            n_nodes = self.n_users + self.n_items

            # 使用 scipy.sparse.bmat 快速构建大矩阵，或者手动构建 data/row/col
            # 这里沿用原逻辑但避免 todense
            R = self.R.tocsr()  # (n_users, n_items)

            # 构造大矩阵 A = [[0, R], [R.T, 0]]
            # 使用 scipy.sparse.vstack 和 hstack 组合
            # Top part: [Zero(n_u, n_u), R]
            top = sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), R])
            # Bottom part: [R.T, Zero(n_i, n_i)]
            bottom = sp.hstack([R.T, sp.csr_matrix((self.n_items, self.n_items))])
            adj_mat = sp.vstack([top, bottom]).tocsr()  # 此时是 CSR 格式的稀疏矩阵

            # [Fix] 关键修正：添加自环 (Self-Loop)
            # 对应 GCN 公式中的 (A + I)
            # 如果不加这一行，节点特征会丢失自身信息，导致欠拟合
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            # 2. 稀疏归一化 (Symmetric Normalization: D^-0.5 * A * D^-0.5)
            if norm_type == 'sym':
                # 计算度数 (行求和)
                rowsum = np.array(adj_mat.sum(1))

                # 计算 D^-0.5
                d_inv_sqrt = np.power(rowsum, -0.5).flatten()
                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

                # 稀疏矩阵乘法: (D^-0.5 * A) * D^-0.5
                norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)

                # 转换为 COO 格式以便创建 Torch Sparse Tensor
                norm_adj = norm_adj.tocoo()

            elif norm_type == 'rw':  # Random Walk Normalization
                rowsum = np.array(adj_mat.sum(1))
                d_inv = np.power(rowsum, -1).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat_inv = sp.diags(d_inv)
                norm_adj = d_mat_inv.dot(adj_mat).tocoo()
            else:
                norm_adj = adj_mat.tocoo()

            # 3. 转换为 PyTorch Sparse Tensor
            indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
            values = torch.from_numpy(norm_adj.data).float()
            shape = torch.Size(norm_adj.shape)
            UI_mat = torch.sparse.FloatTensor(indices, values, shape)

            # 4. 保存缓存
            print("Saving UI_mat to cache...")
            torch.save(UI_mat, self.path + '/UI_mat_' + norm_type + ".pth")

        print("End Load UI_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_UI_single_mat(self, norm_type='2sym'):
        print("Loading UI_single_mat:(" + norm_type + ")")
        t = time()
        try:
            UI_mat = torch.load(self.path + '/UI_single_mat_' + norm_type + ".pth")
        except Exception:
            adj_mat = self.R.todense()
            UI_mat = torch.from_numpy(adj_mat).float()
            UI_mat = self.norm_dense(UI_mat, norm_type)
            UI_mat = UI_mat.to_sparse()
            torch.save(UI_mat, self.path + '/UI_single_mat_' + norm_type + ".pth")
        print("End Load UI_single_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return UI_mat

    def get_U2U_mat(self, norm_type='rw'):
        # U2U_mat default use row normalization,and No-self-connection
        print("Loading User_mat:(" + norm_type + ")")
        t = time()
        try:
            User_mat = torch.load(self.path + '/User_mat_' + norm_type + ".pth")
        except Exception:
            # [优化] 始终使用稀疏矩阵计算
            R = self.R.tocsr()  # 转为 CSR 格式
            User_mat = R.dot(R.T)  # 稀疏矩阵乘法，结果仍为稀疏矩阵

            # 移除对角线 (自连接)
            User_mat.setdiag(0)
            User_mat.eliminate_zeros()  # 清理零元素

            # 归一化
            rowsum = np.array(User_mat.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            User_mat = d_mat_inv.dot(User_mat).tocoo()

            # 转 Tensor
            indices = torch.from_numpy(np.vstack((User_mat.row, User_mat.col)).astype(np.int64))
            values = torch.from_numpy(User_mat.data).float()
            shape = torch.Size(User_mat.shape)
            User_mat = torch.sparse.FloatTensor(indices, values, shape)

            torch.save(User_mat, self.path + '/User_mat_' + norm_type + ".pth")
        print("End Load User_mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return User_mat

    def get_I2I_single_mat(self, norm_type="sym"):
        # I2I_mat default use sym normalization,and must have self-connection because of similarity
        print("Loading I2I media-specific mat:(" + norm_type + ")")
        t = time()
        try:
            image_adj = torch.load(self.path + '/Image_mat_' + norm_type + ".pth")
            text_adj = torch.load(self.path + '/Text_mat_' + norm_type + ".pth")
            if args.dataset=="tiktok":
                audio_adj = torch.load(self.path + '/Audio_mat_' + norm_type + ".pth")

        except Exception:
            # image_feats = np.load('../data/old/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            # text_feats = np.load('../data/old/{}/text_feat.npy'.format(args.dataset))
            text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))
            if args.dataset == "tiktok":
                # audio_feats = np.load('../data/old/{}/audio_feat.npy'.format(args.dataset))
                audio_feats = np.load('../data/{}/audio_feat.npy'.format(args.dataset))

            image_feats = torch.tensor(image_feats).float()
            text_feats = torch.tensor(text_feats).float()
            if args.dataset == "tiktok":
                audio_feats = torch.tensor(audio_feats).float()

            image_adj = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)
            if args.dataset == "tiktok":
                audio_adj = self.build_sim(audio_feats)
                audio_adj = self.build_knn_normalized_graph(audio_adj, topk=args.topk)

            image_adj = self.norm_dense(image_adj, norm_type)
            text_adj = self.norm_dense(text_adj, norm_type)
            if args.dataset == "tiktok":
                audio_adj = self.norm_dense(audio_adj, norm_type)


            image_adj = image_adj.to_sparse()
            text_adj = text_adj.to_sparse()
            if args.dataset == "tiktok":
                audio_adj = audio_adj.to_sparse()


            torch.save(image_adj, self.path + '/Image_mat_' + norm_type + ".pth")
            torch.save(text_adj, self.path + '/Text_mat_' + norm_type + ".pth")
            if args.dataset == "tiktok":
                torch.save(audio_adj, self.path + '/Audio_mat_' + norm_type + ".pth")

        print("End Load I2I media-specific mat:[%.1fs](" % (time() - t) + norm_type + ")")
        if args.dataset == "tiktok":
            return image_adj, text_adj, audio_adj
        else:
            return image_adj, text_adj, ""

    # Order to speed up when Model forward this is be replaced
    # def get_I2I_Hypergrah_mat(self, norm_type="origin"):
    #     # I2I_Hypergraph_mat use origin normalization
    #     print(f"Loading I2I multi-media Hypergraph mat:({norm_type})_topk:{str(args.topk)}")
    #     t = time()
    #     try:
    #         Hypergraph = torch.load(f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")
    #     except Exception:
    #         # image_feats = np.load('../data/old/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
    #         # text_feats = np.load('../data/old/{}/text_feat.npy'.format(args.dataset))
    #         image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
    #         text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))
    #
    #         image_feats = torch.tensor(image_feats).float()
    #         text_feats = torch.tensor(text_feats).float()
    #
    #         image_adj = self.build_sim(image_feats)
    #         image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)
    #
    #         text_adj = self.build_sim(text_feats)
    #         text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)
    #
    #         Hypergraph = torch.cat((image_adj, text_adj), dim=1)
    #         Hypergraph = self.norm_dense(Hypergraph, norm_type)
    #         Hypergraph = Hypergraph.to_sparse()
    #         torch.save(Hypergraph, f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")
    #     print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
    #     return Hypergraph
    #
    # def get_I2I_Hypergraph_mul_mat(self, norm_type="sym"):
    #     # I2I_Hypergraph_mat*I2I_Hypergraph_mat.T use sys normalization
    #     print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_topk:{str(args.topk)}")
    #     t = time()
    #     try:
    #         Hypergraph_mul = torch.load(f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
    #     except Exception:
    #         Hypergraph = self.get_I2I_Hypergrah_mat("origin")
    #         Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
    #         Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
    #         Hypergraph_mul = Hypergraph_mul.to_sparse()
    #         torch.save(Hypergraph_mul, f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
    #     print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
    #     return Hypergraph_mul

    #pytorch---------------------------------------------------------------------------------------------------------------
    def get_I2I_Hypergrah_mat_pt(self, norm_type="origin"):
        # I2I_Hypergraph_mat use origin normalization
        print("Loading I2I multi-media Hypergraph mat:(" + norm_type + ")")
        t = time()
        try:
            Hypergraph = torch.load(self.path + '/hypergraph_mat_' + norm_type + ".pth")
        except Exception:
            image_feats = torch.load("../data/{}/img_feat.pt".format(args.dataset))
            text_feats=torch.load("../data/{}/text_feat.pt".format(args.dataset))

            # image_feats = torch.tensor(image_feats).float()
            # text_feats = torch.tensor(text_feats).float()

            # image_adj = self.build_sim_feature_nan(image_feats)
            image_adj = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            # text_adj = self.build_sim_feature_nan(text_feats)
            text_adj = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            Hypergraph = torch.cat((image_adj, text_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, self.path + '/hypergraph_mat_' + norm_type + ".pth")
        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph
    #pytorch
    def get_I2I_Hypergraph_mul_mat_pt(self,norm_type="sym"):
        # I2I_Hypergraph_mat*I2I_Hypergraph_mat.T use sys normalization
        print("Loading I2I multi-media Hypergraph mul mat*mat.T pytorch:(" + norm_type + ")")
        t = time()
        try:
            Hypergraph_mul = torch.load(self.path + '/hypergraph_mat_mul' + norm_type + ".pth")
        except Exception:
            Hypergraph = self.get_I2I_Hypergrah_mat_pt("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, self.path + '/hypergraph_mat_mul' + norm_type + ".pth")
        print("End Load I2I multi-media Hypergraph mul mat*mat.T pytorch:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul
    #---------------------------------------------------------------------------------------------------------------------

    # Order to speed up when Model forward this is be replaced
    def get_tiktok_I2I_Hypergrah_mat(self, norm_type="origin"):
        # I2I_Hypergraph_mat use origin normalization
        print(f"Loading I2I multi-media Hypergraph mat:({ norm_type })_topk:{str(args.topk)}")
        t = time()
        try:
            Hypergraph = torch.load(f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")
        except Exception:
            # image_feats = np.load('../data/old/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            # text_feats = np.load('../data/old/{}/text_feat.npy'.format(args.dataset))
            # audio_feats = np.load('../data/old/{}/audio_feat.npy'.format(args.dataset))
            image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))  # '../data/{}/image_feat.npy'
            text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))
            audio_feats = np.load('../data/{}/audio_feat.npy'.format(args.dataset))

            image_feats = torch.tensor(image_feats).float()
            text_feats = torch.tensor(text_feats).float()
            audio_feats = torch.tensor(audio_feats).float()

            image_adj = self.build_sim(image_feats)
            image_adj = self.build_knn_normalized_graph(image_adj, topk=args.topk)

            text_adj = self.build_sim(text_feats)
            text_adj = self.build_knn_normalized_graph(text_adj, topk=args.topk)

            audio_adj = self.build_sim(audio_feats)
            audio_adj = self.build_knn_normalized_graph(audio_adj, topk=args.topk)

            Hypergraph = torch.cat((torch.cat((image_adj, text_adj), dim=1), audio_adj), dim=1)
            Hypergraph = self.norm_dense(Hypergraph, norm_type)
            Hypergraph = Hypergraph.to_sparse()
            torch.save(Hypergraph, f"{self.path}/hypergraph_mat_{norm_type}_topk_{str(args.topk)}.pth")
        print("End Load I2I multi-media Hypergraph mat:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph

    def get_tiktok_I2I_Hypergraph_mul_mat(self, norm_type="sym"):
        # I2I_Hypergraph_mat*I2I_Hypergraph_mat.T use sys normalization
        print(f"Loading I2I multi-media Hypergraph mul mat*mat.T:({norm_type})_topk:{str(args.topk)}")
        t = time()
        try:
            Hypergraph_mul = torch.load(f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
        except Exception:
            Hypergraph = self.get_tiktok_I2I_Hypergrah_mat("origin")
            Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)
            Hypergraph_mul = self.norm_dense(Hypergraph_mul, norm_type)
            Hypergraph_mul = Hypergraph_mul.to_sparse()
            torch.save(Hypergraph_mul, f"{self.path}/hypergraph_mat_mul_{norm_type}_topk_{str(args.topk)}.pth")
        print("End Load I2I multi-media Hypergraph mul mat*mat.T:[%.1fs](" % (time() - t) + norm_type + ")")
        return Hypergraph_mul


    def build_sim(self, context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    def build_sim_feature_nan(self, context):
        #image feature extract when url is unvalid or image is destroy features=0,if use norm will nan
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        context_norm[context_norm.isnan()] = 0
        sim = torch.mm(context, context.transpose(1, 0))
        return sim

    def build_knn_normalized_graph(self, adj, topk):
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        adj = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        adj[adj > 0] = 1.
        return adj
