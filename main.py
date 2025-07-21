import datetime
import os
import argparse
import time

from scipy.stats import entropy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models import SessionGraph
from dataset import SessionData
from datasets.data_processor.data_util import Process

import warnings

# CUDA_LAUNCH_BLOCKING = 1.
# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning,
                        message="Creating a tensor from a list of numpy.ndarrays is extremely slow.*")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='jdata_cd', help='dataset name: yc_BT_4/jdata_cd/diginetica_x')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--epoch', type=int, default=40, help='the number of epochs to train for')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]0.001
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--lr_dc', type=float, default=0.1)

parser.add_argument('--patience', type=int, default=6, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', type=bool, default=False, help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--alpha', type=float, default=0.75, help='parameter for beta distribution')
parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
parser.add_argument('--scale', default=True, help='scaling factor sigma')
parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--use_lp_pool', type=str, default="True")
parser.add_argument('--train_flag', type=str, default="True")
parser.add_argument('--PATH', default='../checkpoint/Atten-Mixer_gowalla.pt', help='checkpoint path')

parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--softmax', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dot', default=True, action='store_true')
parser.add_argument('--last_k', type=int, default=7)
parser.add_argument('--l_p', type=int, default=4)
parser.add_argument("--topk", type=int, default=5, help="compute metrics@top_k")
parser.add_argument('--window_size', type=int, default=2)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--miu', default='fusion', help='Dirichlet/fusion')
parser.add_argument('--model', default='SessionGraphAttn3', help='SessionGraphAttn1/SessionGraphAttn')

parser.add_argument('--use_capsule_net', type=str, default="False")
parser.add_argument('--num_capsules', type=int, default=3)
parser.add_argument('--routing_iters', type=int, default=3)

parser.add_argument('--use_attention_net', type=str, default="True")
parser.add_argument('--use_se_net', type=str, default="False")

parser.add_argument('--sequence', type=str, default="sequence", help="last/sequence")

parser.add_argument('--intent_sequence', type=str, default="last", help="last/sequence")
parser.add_argument('--fusion', default='cat', help='cat/jia')   # 1.拼接的方式
parser.add_argument('--fusion_last', default='jia', help='cat/jia')   # 1.拼接的方式

parser.add_argument('--seed', type=int, default=123)

parser.add_argument_group()

opt = parser.parse_args()
print(opt)

hyperparameter_defaults = vars(opt)
config = hyperparameter_defaults


class AreaAttnModel(pl.LightningModule):

    def __init__(self, opt, n_node, cat_num, item_to_cate):
        super().__init__()
        self.opt = opt
        self.best_res = [0, 0, 0, 0]  # Adding diversity to the tracked metrics

        self.model = SessionGraph(opt, n_node, cat_num, item_to_cate)

        self.item_to_cate = item_to_cate

    def forward(self, *args):

        return self.model(*args)

    def compute_diversity(self, recommended_items, item_to_cate):
        """
        Compute diversity of recommendations using entropy of the category distribution.
        """
        categories = [item_to_cate[item] for item in recommended_items if item in item_to_cate]
        category_counts = np.bincount(categories, minlength=self.opt.hiddenSize)
        category_distribution = category_counts / np.sum(category_counts)
        diversity = entropy(category_distribution)
        return diversity

    def training_step(self, batch, batch_idx):

        alias_inputs, A, items, mask, mask1, targets, n_node = batch

        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        mask1.squeeze_()
        targets.squeeze_()
        n_node.squeeze_()

        hidden = self(items)

        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask, items)
        loss = self.model.loss_function(scores, targets - 1)

        return loss

    def validation_step(self, batch, batch_idx):
        alias_inputs, A, items, mask, mask1, targets, n_node = batch
        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        mask1.squeeze_()
        targets.squeeze_()
        n_node.squeeze_()
        hidden = self(items)
        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask, items)
        targets = targets.cpu().detach().numpy()
        sub_scores = scores.topk(self.opt.topk)[1]
        sub_scores = sub_scores.cpu().detach().numpy()

        res = []
        diversity_scores = []
        for score, target in zip(sub_scores, targets):
            hit = float(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr = 0
                ndcg = 0
            else:
                mrr = 1 / (np.where(score == target - 1)[0][0] + 1)
                ndcg = 1 / np.log2(np.where(score == target - 1)[0][0] + 2)
            diversity = self.compute_diversity(score, self.item_to_cate)
            diversity_scores.append(diversity)
            res.append([hit, mrr, ndcg, diversity])

        return torch.tensor(res)

    def validation_epoch_end(self, validation_step_outputs):
        output = torch.cat(validation_step_outputs, dim=0)
        hit = torch.mean(output[:, 0]) * 100
        mrr = torch.mean(output[:, 1]) * 100
        ndcg = torch.mean(output[:, 2]) * 100
        diversity = torch.mean(output[:, 3])
        if hit > self.best_res[0]:
            self.best_res[0] = hit
        if mrr > self.best_res[1]:
            self.best_res[1] = mrr
        if ndcg > self.best_res[2]:
            self.best_res[2] = ndcg
        if diversity > self.best_res[3]:
            self.best_res[3] = diversity
        self.log('hit@20', self.best_res[0])
        self.log('mrr@20', self.best_res[1])
        self.log('ndcg@20', self.best_res[2])
        self.log('diversity', self.best_res[3])
        print(f"Validation Results - MRR: {mrr:.4f}, Hit Rate: {hit:.4f}, NDCG: {ndcg:.4f}, Diversity: {diversity:.4f}")

    def test_step(self, batch, idx):

        alias_inputs, A, items, mask, mask1, targets, n_node = batch
        alias_inputs.squeeze_()
        A.squeeze_()
        items.squeeze_()
        mask.squeeze_()
        mask1.squeeze_()
        targets.squeeze_()
        n_node.squeeze_()
        hidden = self(items)
        seq_hidden = torch.stack([self.model.get(i, hidden, alias_inputs) for i in range(len(alias_inputs))])
        seq_hidden = torch.cat((seq_hidden, hidden[:, max(n_node):]), dim=1)
        seq_hidden = seq_hidden * mask.unsqueeze(-1)
        if self.opt.norm:
            seq_shape = list(seq_hidden.size())
            seq_hidden = seq_hidden.view(-1, self.opt.hiddenSize)
            norms = torch.norm(seq_hidden, p=2, dim=-1) + 1e-12  # l2 norm over session embedding
            seq_hidden = seq_hidden.div(norms.unsqueeze(-1))
            seq_hidden = seq_hidden.view(seq_shape)
        scores = self.model.compute_scores(seq_hidden, mask)
        targets = targets.cpu().detach().numpy()
        sub_scores = scores.topk(self.opt.topk)[1]
        sub_scores = sub_scores.cpu().detach().numpy()

        # 计算推荐结果的流行度
        recommended_items = sub_scores.flatten()
        self.recommendation_popularity = torch.bincount(torch.tensor(recommended_items)).float() / len(
            recommended_items)
        print("popularity:", self.recommendation_popularity)

        res = []
        for score, target in zip(sub_scores, targets):
            hit = float(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr = 0
            else:
                mrr = 1 / (np.where(score == target - 1)[0][0] + 1)
            res.append([hit, mrr])

        return torch.tensor(res)

    def test_epoch_end(self, test_step_outputs):

        output = torch.cat(test_step_outputs, dim=0)
        hit = torch.mean(output[:, 0]) * 100
        mrr = torch.mean(output[:, 1]) * 100
        if hit > self.best_res[0]:
            self.best_res[0] = hit
        if mrr > self.best_res[1]:
            self.best_res[1] = mrr
        self.log('hit@20', self.best_res[0])
        self.log('mrr@20', self.best_res[1])
        print(mrr, hit)

    def configure_optimizers(self):

        optimizer = optim.Adam(self.parameters(), lr=self.opt.lr, weight_decay=self.opt.l2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_dc_step, gamma=opt.lr_dc)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def on_epoch_start(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Epoch {self.current_epoch} started at {current_time}")


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed = opt.seed
    pl.seed_everything(seed)
    # init_seed(seed)

    path = './datasets/data/'
    data_path = path + opt.dataset
    cur_dataset = opt.dataset

    if opt.dataset.startswith('yc'):
        p = Process(cur_dataset, data_path, 'train')
    elif opt.dataset.startswith('jd'):
        p = Process(cur_dataset, data_path, 'test')
    elif opt.dataset.startswith('digi'):
        p = Process(cur_dataset, data_path, 'test')

    item_to_category = p.item_to_category
    cat_num = len(set(item_to_category.values()))
    print("种类数", cat_num)
    # print(item_to_category)

    if opt.dataset.startswith('yc'):
        n_node = 12577
        cat_num = 12
    elif opt.dataset.startswith('jd'):
        n_node = 31401

    elif opt.dataset.startswith('digi'):
        n_node = 15661
        cat_num = 846

    session_data = SessionData(name=opt.dataset, batch_size=opt.batchSize)
    early_stop_callback = EarlyStopping(
        monitor='mrr@20',
        min_delta=0.00,
        patience=opt.patience,
        verbose=False,
        mode='max'
    )
    # trainer = pl.Trainer(gpus=[get_freer_gpu()], deterministic=True, max_epochs=10, num_sanity_val_steps=2,
    #                      callbacks=[early_stop_callback])

    # 获取可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    gpus = [0] if num_gpus > 0 else None  # 如果有可用的 GPU，则使用第一个 GPU，否则不使用 GPU

    trainer = pl.Trainer(gpus=gpus, deterministic=True, max_epochs=opt.epoch, num_sanity_val_steps=2,
                         callbacks=[early_stop_callback], progress_bar_refresh_rate=0)
    if opt.train_flag == "True":
        model = AreaAttnModel(opt=opt, n_node=n_node, cat_num=cat_num, item_to_cate=item_to_category)
        trainer.fit(model, session_data)
    else:
        model = AreaAttnModel(opt=opt, n_node=n_node, cat_num=cat_num, item_to_cate=item_to_category)
        model.load_state_dict(torch.load(opt.PATH))
        model.eval()
        trainer.test(model, session_data.test_dataloader())


if __name__ == "__main__":
    main()
