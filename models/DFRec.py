# /usr/bin/env python36
# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from torch.distributions import Dirichlet, Normal


# CUDA_LAUNCH_BLOCKING = 1.

class SELayer(Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1)
        avg_pool = self.fc1(avg_pool)  # (batch_size, channels // reduction)
        avg_pool = F.relu(avg_pool)
        avg_pool = self.fc2(avg_pool)
        attention = self.sigmoid(avg_pool).unsqueeze(1)  # (batch_size, 1, channels)
        return x * attention


class Transformer(nn.Module):
    def __init__(self, input_dim, num_capsules, output_dim):
        super(Transformer, self).__init__()
        self.fc = nn.Linear(input_dim, num_capsules * output_dim)
        self.num_capsules = num_capsules
        self.output_dim = output_dim

    def forward(self, inputs):
        batch_size, L = inputs.shape[:2]
        flattened = inputs.view(batch_size * L, -1)  # (batch_size * L, input_dim)
        transformed = self.fc(flattened)  # (batch_size * L, num_capsules * output_dim)
        u_hat = transformed.view(batch_size, L, self.num_capsules,
                                 self.output_dim)  # (batch_size, L, num_capsules, output_dim)
        return u_hat


class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_capsules, routing_iters=3):
        super(CapsuleLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters

        # 定义权重矩阵，维度为 (input_dim, num_capsules, output_dim)
        self.weights = nn.Parameter(torch.randn(input_dim, num_capsules, output_dim))
        self.tran = Transformer(input_dim, num_capsules, output_dim)

    def squash(self, x):
        """ Squashing function for capsule vectors. """
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)

    def forward(self, inputs):
        """
        inputs: shape (batch_size, L, input_dim)
        """
        batch_size, L, _ = inputs.size()

        # 进行矩阵乘法
        # print('inputs.shape', inputs.shape)
        # print('self.weights.shape', self.weights.shape)
        # u_hat = torch.matmul(inputs, self.weights)
        u_hat = self.tran(inputs)
        u_hat = u_hat.view(batch_size, L, self.num_capsules,
                           self.output_dim)  # (batch_size, L, num_capsules, output_dim)

        # 初始化路由 logits
        b_ij = torch.zeros(batch_size, L, self.num_capsules).to(inputs.device)

        for i in range(self.routing_iters):
            # 计算 coupling coefficients (routing by agreement)
            c_ij = F.softmax(b_ij, dim=2)  # (batch_size, L, num_capsules)

            # 计算加权和
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)  # (batch_size, num_capsules, output_dim)

            # Squash
            v_j = self.squash(s_j)  # (batch_size, num_capsules, output_dim)

            # 更新路由 logits
            b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)

        return v_j  # (batch_size, num_capsules, output_dim)


class LastAttention1(nn.Module):
    def __init__(self, hidden_size, num_heads, num_capsules, routing_iters=3, dropout=0.1, opt=None):
        super(LastAttention1, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters
        self.dropout = dropout
        self.opt = opt

        # 初始化用于线性变换的层
        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # 胶囊层
        self.capsule_layer = CapsuleLayer(input_dim=hidden_size,
                                          output_dim=hidden_size,
                                          num_capsules=num_capsules,
                                          routing_iters=routing_iters)

        self.attention_layer = nn.Linear(self.hidden_size, 1)  # output_dim 是胶囊输出的维度

    def forward(self, hts, hidden, mask):
        """
        hts: (batch_size, L, hidden_dim)
        hidden: (batch_size, hidden_dim)
        mask: (batch_size, seq_len)
        """
        # Step 1: 通过线性层处理查询
        q = self.linear(hts)

        # Step 2: 胶囊网络处理查询
        capsules = self.capsule_layer(q)  # (batch_size, num_capsules, output_dim)

        # Step 3: 聚合胶囊向量（加权求和）
        # 计算注意力权重
        attention_weights = F.softmax(self.attention_layer(capsules), dim=1)  # (batch_size, num_capsules)
        # 使用注意力权重加权求和
        attention_weights = attention_weights.view(-1, 1, capsules.size(1))  # (batch_size, 1, num_capsules)
        final_representation = torch.bmm(attention_weights, capsules)  # (batch_size, 1, output_dim)
        final_representation = final_representation.squeeze(1)  # (batch_size, output_dim)

        final_representation = F.dropout(final_representation, p=self.dropout, training=self.training)
        return final_representation


class LastAttenion(Module):

    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_lp_pool=False, use_capsule_net=False, opt=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_lp_pool = use_lp_pool
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

        self.use_capsule_net = use_capsule_net

        self.opt = opt
        # 新增的SE层
        self.se_layer = SELayer(self.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):

        q0 = self.linear_zero(ht1).view(-1, ht1.size(1), self.hidden_size // self.heads)
        q1 = self.linear_one(hidden).view(-1, hidden.size(1),
                                          self.hidden_size // self.heads)  # batch_size x seq_length x latent_size
        q2 = self.linear_two(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        assert not torch.isnan(q0).any()
        assert not torch.isnan(q1).any()
        alpha = torch.sigmoid(torch.matmul(q0, q1.permute(0, 2, 1)))
        assert not torch.isnan(alpha).any()
        alpha = alpha.view(-1, q0.size(1) * self.heads, hidden.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(2 * alpha, dim=1)
        assert not torch.isnan(alpha).any()
        # print("alpha.shape", alpha.shape)  # alpha.shape torch.Size([100, 85, 56])

        if self.use_lp_pool == "True":
            m = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha = m(alpha)
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
            alpha = torch.softmax(2 * alpha, dim=1)
            # print("alpha.shape", alpha.shape)  # alpha.shape torch.Size([100, 85, 8])

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        a = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        a = self.last_layernorm(a)

        if self.opt.use_se_net == 'True':
            # 在最后添加SE层
            a = self.se_layer(a)  # 应用SE层
        return a, alpha


class SessionGraph(Module):
    def __init__(self, opt, n_node, cat_num, item_to_cate):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.norm = opt.norm
        self.scale = opt.scale
        self.batch_size = opt.batchSize
        self.heads = opt.heads
        self.use_lp_pool = opt.use_lp_pool
        self.use_capsule_net = opt.use_capsule_net
        self.softmax = opt.softmax
        self.dropout = opt.dropout
        self.last_k = opt.last_k
        self.dot = opt.dot
        self.embedding = nn.Embedding(self.n_node + 1, self.hidden_size)
        self.l_p = opt.l_p
        self.args = opt
        self.mattn = LastAttenion(self.hidden_size, self.heads, self.dot, self.l_p, last_k=self.last_k,
                                  use_lp_pool=self.use_lp_pool, use_capsule_net=self.use_capsule_net, opt=self.args)

        self.num_capsules = opt.num_capsules
        self.routing_iters = opt.routing_iters
        self.mattn_Capsule = LastAttention1(self.hidden_size, self.heads, num_capsules=self.num_capsules,
                                            routing_iters=self.routing_iters, dropout=self.dropout, opt=self.args)

        self.linear_q = nn.ModuleList()

        for i in range(1, self.last_k + 1):
            self.linear_q.append(nn.Linear(i * self.hidden_size, self.hidden_size))

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_transform1 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

        # ##################################### 新加的 #############################################
        self.cat_num = cat_num  # 类别数
        self.item_category_map = item_to_cate  # 物品-种类映射字典
        self.latent_dim = self.hidden_size  # 隐层维度
        self.cat_dim = self.hidden_size  # 种类维度

        self.cat_emb = torch.nn.Embedding(self.cat_num + 1, self.hidden_size, padding_idx=0)  # 种类表征

        self.combined_linear = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 激发意图网络的相关参数
        self.fc_mu = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_sigma = torch.nn.Linear(self.latent_dim, self.latent_dim)

        # 兴趣表征的融合层
        self.fc_intent = nn.Linear(self.latent_dim, self.latent_dim)

        # GARCH模型的可学习参数
        self.alpha = nn.Parameter(torch.tensor(0.1))  # α 参数，初始值为 0.1
        self.beta = nn.Parameter(torch.tensor(0.8))  # β 参数，初始值为 0.8
        self.gamma = nn.Parameter(torch.tensor(0.1))  # 𝛾 参数，初始值为 0.1
        self.delta = nn.Parameter(torch.tensor(0.1))  # δ 参数，初始值为 0.1

        # 新增：可学习的平衡因子
        self.balance_factor = nn.Parameter(torch.tensor(0.5))

        # GRU用于计算兴趣波动的隐状态
        self.variance_rnn = nn.GRU(input_size=self.cat_dim, hidden_size=self.latent_dim, batch_first=True)
        self.window_size = opt.window_size
        self.linear_mapping = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_mapping1 = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=True)

        # 新增的SE层
        self.se_layer = SELayer(self.hidden_size)

        # 新增的线性层，用于将拼接后的hts映射回原始维度
        self.linear_mapping_back = nn.Linear(self.hidden_size + self.cat_dim, self.hidden_size)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get(self, i, hidden, alias_inputs):
        return hidden[i][alias_inputs[i]]

    def user_intent(self, log_seqs):
        batch_size = log_seqs.shape[0]
        seq_len = log_seqs.shape[1]

        if self.args.miu == 'Dirichlet':
            current_session_categories = torch.stack(
                [torch.tensor([self.item_category_map.get(item_id.item(), 0) for item_id in session], dtype=torch.long)
                 for session in log_seqs])

            # 统计每个种类的出现次数
            category_counts = torch.zeros((batch_size, self.cat_num + 1),
                                          dtype=torch.float)  # (batch_size, num_categories)
            # 遍历 batch
            for i in range(batch_size):
                category_counts[i].scatter_add_(0, current_session_categories[i],
                                                torch.ones_like(current_session_categories[i], dtype=torch.float))

            alpha_prior = category_counts + 1e-3
            dirichlet_dist = Dirichlet(alpha_prior)
            category_weights = dirichlet_dist.sample()

            # 计算用户的通用意图表征
            category_embeddings = self.cat_emb.weight
            category_weights = category_weights.to(self.args.device)
            category_embeddings = category_embeddings.to(self.args.device)
            general_intent = torch.matmul(category_weights, category_embeddings)
            general_intent = torch.clamp(general_intent, min=1e-3)

            current_session_categories = current_session_categories.to(self.args.device)

            item_embeddings = self.embedding(log_seqs)
            category_embeddings = self.cat_emb(current_session_categories)

            combined_embeddings = torch.cat((item_embeddings, category_embeddings),
                                            dim=-1)

            combined_embeddings = self.combined_linear(combined_embeddings)

        elif self.args.miu == 'fusion':
            current_session_categories = torch.stack(
                [torch.tensor([self.item_category_map.get(item_id.item(), 0) for item_id in session], dtype=torch.long)
                 for session in log_seqs])
            current_session_categories = current_session_categories.to(self.args.device)

            item_embeddings = self.embedding(log_seqs)
            category_embeddings = self.cat_emb(current_session_categories)

            combined_embeddings = torch.cat((item_embeddings, category_embeddings),
                                            dim=-1)

            combined_embeddings = self.combined_linear(combined_embeddings)

            # 通用意图
            general_intent = torch.mean(combined_embeddings, dim=1)

        # 计算用户兴趣波动
        gru_output, _ = self.variance_rnn(combined_embeddings)

        # 初始化短期方差和长期方差
        short_term_variance = torch.zeros(batch_size, self.latent_dim).to(
            combined_embeddings.device)
        long_term_variance = torch.zeros(batch_size, self.latent_dim).to(
            combined_embeddings.device)

        # 计算残差 epsilon_t = e_t^c - e_{t-1}^c
        residuals = combined_embeddings[:, 1:, :] - combined_embeddings[:, :-1,
                                                    :]
        epsilon = 1e-3

        emerging_intents = []
        # 短期方差和长期方差更新
        for t in range(self.args.last_k):
            t_reverse = seq_len - 1 - t
            if t_reverse > 0:
                hidden_diff = gru_output[:, t_reverse, :] - gru_output[:, t_reverse - 1,
                                                            :]

                residual_square = residuals[:, t_reverse - 1, :] ** 2
                short_term_variance = (self.alpha * residual_square +
                                       self.beta * short_term_variance ** 2 +
                                       self.gamma * (hidden_diff ** 2))
                short_term_variance = torch.clamp(short_term_variance, min=epsilon)  # 避免出现负值
                short_term_variance = F.softplus(short_term_variance)  # 使用 softplus 保证正值
                short_term_variance = torch.nan_to_num(short_term_variance, nan=epsilon, posinf=epsilon,
                                                       neginf=epsilon)

            if t_reverse >= self.window_size:
                window_residuals = residuals[:, t_reverse - self.window_size:t_reverse,
                                   :]
                window_residuals_mean = window_residuals.mean(dim=1)
                long_term_variance = (
                        (window_residuals_mean ** 2).mean(dim=1, keepdim=True) +
                        self.delta * (
                                gru_output[:, t_reverse, :] - general_intent) ** 2)

            long_term_variance = torch.clamp(long_term_variance, min=epsilon)
            long_term_variance = F.softplus(long_term_variance)
            long_term_variance = torch.nan_to_num(long_term_variance, nan=epsilon, posinf=epsilon, neginf=epsilon)

            combined_variance = short_term_variance + long_term_variance
            combined_variance = torch.clamp(combined_variance, min=epsilon)

            # 从正态分布中采样得到激发意图
            normal_dist = Normal(general_intent.unsqueeze(1),
                                 torch.sqrt(combined_variance).unsqueeze(1))
            emerging_intent = normal_dist.rsample()
            emerging_intents.append(emerging_intent.squeeze(1))

        emerging_intents = torch.stack(emerging_intents, dim=1)

        return emerging_intents

    def compute_scores(self, hidden, mask, items):

        hts = []

        lengths = torch.sum(mask, dim=1)

        for i in range(1, self.last_k + 1):
            hts.append(self.linear_q[i - 1](torch.cat(
                [hidden[torch.arange(mask.size(0)).long(), torch.clamp(lengths - j, -1, 1000)] for j in range(i)],
                dim=-1)).unsqueeze(1))

        ht0 = hidden[torch.arange(mask.size(0)).long(), torch.sum(mask, 1) - 1]

        hts = torch.cat(hts, dim=1)
        hts = hts.div(torch.norm(hts, p=2, dim=1, keepdim=True) + 1e-12)  # 所有的查询向量

        emerging_intents = self.user_intent(items)
        emerging_intents = emerging_intents[:, :hts.size(1), :]

        hts = torch.cat([hts, emerging_intents], dim=-1)
        hts = self.linear_mapping(hts)

        hidden1 = hidden
        hidden = hidden1[:, :mask.size(1)]

        if self.args.use_capsule_net == 'True' and self.args.use_attention_net == 'False':
            ais = self.mattn_Capsule(hts, hidden, mask)
        elif self.args.use_capsule_net == 'False' and self.args.use_attention_net == 'True':
            ais, weights = self.mattn(hts, hidden, mask)
        elif self.args.use_capsule_net == 'True' and self.args.use_attention_net == 'True':
            ais1, weights = self.mattn(hts, hidden, mask)
            ais2 = self.mattn_Capsule(hts, hidden, mask)
            ais = self.linear_transform1(torch.cat((ais1, ais2), dim=-1))

        a = self.linear_transform(torch.cat((ais.squeeze(), ht0), 1))
        b = self.embedding.weight[1:]

        if self.norm:
            a = a.div(torch.norm(a, p=2, dim=1, keepdim=True) + 1e-12)
            b = b.div(torch.norm(b, p=2, dim=1, keepdim=True) + 1e-12)
        b = F.dropout(b, self.dropout, training=self.training)
        scores = torch.matmul(a, b.transpose(1, 0))
        if self.scale:
            scores = 16 * scores
        return scores

    def forward(self, inputs):

        hidden = self.embedding(inputs)

        if self.norm:
            hidden = hidden.div(torch.norm(hidden, p=2, dim=-1, keepdim=True) + 1e-12)

        hidden = F.dropout(hidden, self.dropout, training=self.training)

        if self.args.use_se_net == 'True':
            hidden = self.se_layer(hidden)

        return hidden

