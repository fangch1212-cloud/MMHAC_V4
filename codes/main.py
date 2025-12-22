from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import json

import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.sparse as sparse

from utility.load_data import Data
from utility.parser import parse_args
from Models import MM_HAC

from utility.batch_test import *
from utility.logging import Logger
import pathlib

args = parse_args()

# print(torch.cuda.current_device())

path_name = f"uu_ii={args.User_layers}_{args.Item_layers}_{args.user_loss_ratio}_{args.item_loss_ratio}" \
            f"_topk={args.topk}_t={args.temperature}_regs={args.regs}_dim={args.embed_size}_{args.ablation_target}"
path = f"../{args.dataset}/{path_name}/"  # Separate folders for records and weights
record_path = f"../{args.dataset}/MM/"  # Folders summarizing ablation experiments
pathlib.Path(f"{path}").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{record_path}").mkdir(parents=True, exist_ok=True)


class Trainer(object):
    def __init__(self, data_config):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.logger = Logger(path, is_debug=args.debug, target=path_name, path2=record_path,
                             ablation_target=args.ablation_target)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = args.regs
        self.decay = self.regs

        # 这里的矩阵对于 MM_HAC 不是必须手动传入的，但保留也无妨
        self.UI_mat = data_config['UI_mat'].cuda()
        self.User_mat = data_config['User_mat'].cuda()
        self.Item_mat = data_config['Item_mat'].cuda()

        # [修改] 构建配置字典并初始化 MM_HAC
        model_config = {
            'recdim': self.emb_dim,
            'layer': args.User_layers,
            'ssl_reg': args.ssl_reg,
            'temperature': args.temperature,
            'reg': args.regs,
            'dataset_name': args.dataset,

            # [新增] 必须把新参数传进去！
            'proto_reg': args.proto_reg,
            'n_clusters': args.n_clusters,
            'modal_align_reg': args.modal_align_reg,
            'eps': args.eps  # 如果用了 SimGCL
        }
        # data_generator 是全局变量
        self.model = MM_HAC(model_config, data_generator)

        self.model = self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            # [修改] 增加一个 _ 来接收 modal_embeddings_list (共5个返回值)
            ua_embeddings, ia_embeddings, _, _, _ = self.model()
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        best_recall = 0
        best_ndcg = 0
        test_ret = ""

        for epoch in (range(args.epoch)):
            t1 = time()

            # [Add] MM-HAC v3.0: Update clusters periodically
            if epoch % 5 == 0:
                self.model.update_clusters()

            loss, mf_loss, emb_loss, reg_loss, modal_loss = 0., 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1

            for idx in (range(n_batch)):
                self.model.train()
                self.optimizer.zero_grad()

                users, pos_items, neg_items = data_generator.sample()

                # [Mod] Unpack 5 values
                batch_loss, batch_mf, batch_cl, batch_modal, batch_reg = self.model.calc_loss(users, pos_items,
                                                                                              neg_items)

                batch_loss.backward(retain_graph=False)
                self.optimizer.step()

                # loss += float(batch_loss)
                # mf_loss += float(batch_mf)
                # emb_loss += float(batch_cl)    # Structural CL
                # modal_loss += float(batch_modal) # Modal/Prototype CL
                # reg_loss += float(batch_reg)

                loss += batch_loss.item()
                mf_loss += batch_mf.item()
                emb_loss += batch_cl.item()
                modal_loss += batch_modal.item()
                reg_loss += batch_reg.item()

            self.lr_scheduler.step()

            if math.isnan(loss):
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                # [Mod] Log all loss components
                perf_str = 'Epoch %d [%.1fs]: train==[%.4f=%.4f + %.4f(SCL) + %.4f(Proto) + %.4f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, modal_loss, reg_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_val, is_val=True)
            training_time_list.append(t2 - t1)
            t3 = time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])
            hit_loger.append(ret['hit_ratio'])

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.4f=%.4f + %.4f + %.4f + %.4f], recall=[%.5f, %.5f], ' \
                           'ndcg=[%.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, modal_loss, reg_loss,
                            ret['recall'][0], ret['recall'][-1],
                            ret['ndcg'][0], ret['ndcg'][-1])
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall or ret['ndcg'][1] > best_ndcg:
                # 简单的保存最佳逻辑
                if ret['recall'][1] > best_recall:
                    best_recall = ret['recall'][1]
                    test_ret = self.test(users_to_test, is_val=False)
                    self.logger.logging("Test_Recall@%d: %.8f  Test_NDCG@%d: %.8f" % (
                        eval(args.Ks)[1], test_ret['recall'][1], eval(args.Ks)[1], test_ret['ndcg'][1]))

                if ret['ndcg'][1] > best_ndcg: best_ndcg = ret['ndcg'][1]
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break

        self.logger.logging(str(test_ret))

    # def train(self):
    #     training_time_list = []
    #     loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    #     stopping_step = 0
    #     best_recall = 0
    #     best_ndcg = 0
    #     test_ret = ""

    #     # 初始化 CSV 日志
    #     csv_log_path = os.path.join(path, 'training_curves.csv')
    #     with open(csv_log_path, 'w') as f:
    #         # [修改] 表头增加 Modal_Loss
    #         f.write("Epoch,Total_Loss,BPR_Loss,SCL_Loss,Modal_Loss,Reg_Loss,Val_Recall,Val_NDCG\n")

    #     # Warm-up 阶段的参数
    #     WARMUP_EPOCHS = 10  # 预热 10 个 Epoch
    #     initial_ssl_reg = args.ssl_reg  # 结构对比的原始权重 (例如 0.05)
    #     initial_modal_reg = args.modal_align_reg  # 模态对比的原始权重 (例如 0.01)

    #     for epoch in (range(args.epoch)):
    #         # 动态调整对比学习权重
    #         if epoch < WARMUP_EPOCHS:
    #             current_ssl_reg = 0.0
    #             current_modal_reg = 0.0
    #         else:
    #             current_ssl_reg = initial_ssl_reg
    #             current_modal_reg = initial_modal_reg

    #         # 在计算 calc_loss 之前，将当前权重赋给 model.config
    #         self.model.config['ssl_reg'] = current_ssl_reg
    #         self.model.modal_align_reg = current_modal_reg  # 假设您已将 modal_align_reg 存入 model 实例

    #         t1 = time()
    #         # [修改] 增加 modal_loss 累加器
    #         loss, mf_loss, emb_loss, reg_loss, modal_loss = 0., 0., 0., 0., 0.
    #         contrastive_loss = 0.
    #         n_batch = data_generator.n_train // args.batch_size + 1

    #         for idx in (range(n_batch)):
    #             self.model.train()
    #             self.optimizer.zero_grad()

    #             users, pos_items, neg_items = data_generator.sample()

    #             # [修改] 接收 5 个返回值 (Total, BPR, Total_SCL, Modal_Align, Reg)
    #             batch_loss, batch_mf, batch_cl, batch_modal, batch_reg = self.model.calc_loss(users, pos_items,
    #                                                                                           neg_items)

    #             batch_loss.backward(retain_graph=False)
    #             self.optimizer.step()

    #             loss += float(batch_loss)
    #             mf_loss += float(batch_mf)
    #             contrastive_loss += float(batch_cl)  # 这里包含了 结构SCL + 模态SCL
    #             modal_loss += float(batch_modal)  # 单独记录一下模态对齐损失，便于观察
    #             reg_loss += float(batch_reg)

    #         self.lr_scheduler.step()

    #         if math.isnan(loss):
    #             self.logger.logging('ERROR: loss is nan.')
    #             sys.exit()

    #         # 计算平均 Loss
    #         avg_loss = loss / n_batch
    #         avg_mf = mf_loss / n_batch
    #         avg_cl = contrastive_loss / n_batch
    #         avg_modal = modal_loss / n_batch
    #         avg_reg = reg_loss / n_batch

    #         # 验证集测试
    #         t2 = time()
    #         users_to_val = list(data_generator.val_set.keys())
    #         ret = self.test(users_to_val, is_val=True)
    #         training_time_list.append(t2 - t1)
    #         t3 = time()

    #         loss_loger.append(loss)
    #         rec_loger.append(ret['recall'])
    #         pre_loger.append(ret['precision'])
    #         ndcg_loger.append(ret['ndcg'])
    #         hit_loger.append(ret['hit_ratio'])

    #         # [修改] 写入 CSV (增加 avg_modal)
    #         with open(csv_log_path, 'a') as f:
    #             f.write(f"{epoch},{avg_loss:.5f},{avg_mf:.5f},{avg_cl:.5f},{avg_modal:.5f},{avg_reg:.5f},"
    #                     f"{ret['recall'][1]:.5f},{ret['ndcg'][1]:.5f}\n")

    #         # if args.verbose > 0:
    #         #     # [修改] 日志打印增加 Modal Loss 信息
    #         #     perf_str = 'Epoch %d [%.1fs]: train==[%.5f = %.5f + %.5f(SCL_All) + %.5f(Modal) + %.5f]' % \
    #         #                (epoch, t2 - t1, avg_loss, avg_mf, avg_cl, avg_modal, avg_reg)
    #         #     self.logger.logging(perf_str)

    #         # 在 train 方法的 epoch 循环内部
    #         # 如果当前 epoch 不是 verbose 的倍数，也不是最后一个 epoch，则跳过测试
    #         if (epoch + 1) % args.verbose != 0 and epoch != args.epoch - 1:
    #             # 只打印训练日志
    #             if args.verbose > 0:
    #                 perf_str = 'Epoch %d [%.1fs]: train==[%.5f = %.5f + %.5f(SCL_All) + %.5f(Modal) + %.5f]' % \
    #                            (epoch, t2 - t1, avg_loss, avg_mf, avg_cl, avg_modal, avg_reg)
    #                 self.logger.logging(perf_str)
    #             continue  # <--- 关键：跳过后面的 test() 调用

    #         # 检查是否打破记录
    #         if ret['recall'][1] > best_recall or ret['ndcg'][1] > best_ndcg:
    #             if ret['recall'][1] > best_recall: best_recall = ret['recall'][1]
    #             if ret['ndcg'][1] > best_ndcg: best_ndcg = ret['ndcg'][1]

    #             users_to_test = list(data_generator.test_set.keys())
    #             test_ret = self.test(users_to_test, is_val=False)

    #             self.logger.logging(">>> BEST! Test_Recall@%d: %.8f  Test_NDCG@%d: %.8f" % (
    #                 eval(args.Ks)[1], test_ret['recall'][1], eval(args.Ks)[1], test_ret['ndcg'][1]))

    #             save_path = os.path.join(path, 'best_model.pth')
    #             torch.save(self.model.state_dict(), save_path)

    #             stopping_step = 0
    #         elif stopping_step < args.early_stopping_patience:
    #             stopping_step += 1
    #             self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
    #         else:
    #             self.logger.logging('#####Early stop! #####')
    #             break

    #     self.logger.logging(str(test_ret))
    #     self.logger.logging_sum(f"{path_name}:{str(test_ret)}")


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    # 下面这几行获取代码其实对于 main 来说不是必须的，因为我们传了 data_generator
    # 但保留它们以维持 load_data 中的单例状态初始化是好的
    UI_mat = data_generator.get_UI_mat()
    User_mat = data_generator.get_U2U_mat()

    if args.dataset == "Tiktok":
        Item_mat = data_generator.get_tiktok_I2I_Hypergraph_mul_mat()
    elif args.dataset in ["Clothing", "Sports"]:
        Item_mat = data_generator.get_I2I_Hypergraph_mul_mat()

    config['UI_mat'] = UI_mat
    config['User_mat'] = User_mat
    config['Item_mat'] = Item_mat

    trainer = Trainer(data_config=config)
    trainer.train()
