import pickle

import numpy as np
import csv
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from datasets.data_processor import yoochoose_dataset
from datasets.data_processor import jdata_dataset
from datasets.data_processor import diginetica_dataset


class Process:
    def __init__(self, dataset, data_path, select='test'):
        self.dataset = dataset
        self.data_path = data_path

        self.item2index = {}
        self.item2count = {}
        self.index2item = {}
        self.item_to_category = {}

        self.num_histories = 0
        self.num_words = 0

        self.all_sessions_index = []  # 根据映射字典创建数字编号格式的交互数据

        if self.dataset.startswith('yc'):
            self.all_data, self.max_vid, self.item_cates = yoochoose_dataset.load_cd_data(data_path, type='aug',
                                                                                          test_length=True,
                                                                                          highfreq_only=True)
        elif self.dataset.startswith('jd'):
            self.all_data, self.max_vid, self.item_cates = jdata_dataset.load_cd_data(data_path, type='aug',
                                                                                      test_length=True,
                                                                                      highfreq_only=True)
        elif self.dataset.startswith('digi'):
            self.all_data, self.max_vid, self.item_cates = diginetica_dataset.load_cd_data(data_path, type='aug',
                                                                                           test_length=True,
                                                                                           highfreq_only=True)
        self.cur_all_data = self.all_data[select]
        # self.cur_all_data = self.cur_all_data[:10000]

        self.remove_duplicates()
        self.process_data()
        self.split_data()
        self.print_info()
        self.save()

    def remove_duplicates(self):
        def to_hashable(item):
            if isinstance(item, list):
                return tuple(to_hashable(i) for i in item)
            return item

        # 将嵌套列表转换为可哈希的形式
        hashable_data = [to_hashable(x) for x in self.cur_all_data]
        # 使用集合去重
        unique_data = set(hashable_data)
        # 保持数据为元组形式
        self.cur_all_data = list(unique_data)
        print(f"去重后的数据条数: {len(self.cur_all_data)}")

        if self.dataset.startswith('jd'):
            self.cur_all_data = self.cur_all_data[:int(0.2 * len(self.cur_all_data))]
        # elif self.dataset.startswith('di'):
        #     self.cur_all_data = self.cur_all_data[:int(0.5 * len(self.cur_all_data))]

    def process_data(self):
        # 新增字典用于连续化类别ID
        self.category_mapping = {}
        self.next_category_id = 0

        for session in self.cur_all_data:
            item_ids = list(session[1]) + [session[2][0]]  # 将元组转换为列表，然后合并
            category_ids = list(session[3]) + [session[4][0]]  # 将元组转换为列表，然后合并

            session_index = []
            for item_id, category_id in zip(item_ids, category_ids):
                # 处理类别ID的连续化
                if category_id not in self.category_mapping:
                    self.category_mapping[category_id] = self.next_category_id
                    self.next_category_id += 1
                continuous_category_id = self.category_mapping[category_id]

                if item_id not in self.item2index:
                    self.num_words += 1
                    new_id = self.num_words
                    self.item2index[item_id] = new_id
                    self.index2item[new_id] = item_id
                    self.item2count[new_id] = 1
                else:
                    new_id = self.item2index[item_id]
                    self.item2count[new_id] += 1

                self.item_to_category[new_id] = continuous_category_id  # 使用连续化的类别ID
                session_index.append(new_id)
                self.num_histories += 1

            self.all_sessions_index.append(session_index)
        print("物品映射字典建立完毕")
        print("重新编号后的交互数据列表创建完毕")

    # 划分数据集
    def split_data(self):
        # ###########得到验证集############
        # 删掉每个会话里最后一个物品
        self.valid_data = []
        for session in self.all_sessions_index:
            self.valid_data.append(session[:-1])

        # #########得到训练集##########
        self.train_data = []
        # 数据增强：从第一个元素开始，每次取多个元素并将它们存储为一个列表添加到训练集中
        for session in self.valid_data:
            for i in range(len(session)):
                # 从第一个元素开始，每次取多个元素并添加到result_list中
                self.train_data.append(session[:i + 3])

        # ###########得到测试集############
        self.test_data = self.all_sessions_index
        print("数据集划分结束")

    # 输出数据集的统计信息
    def print_info(self):
        print('-' * 50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(self.all_sessions_index)))
        print('Number of train sessions: {}'.format(len(self.train_data)))
        print('Number of valid sessions: {}'.format(len(self.valid_data)))
        print('Number of test sessions: {}'.format(len(self.test_data)))
        print('Number of categories: {}'.format(self.next_category_id))
        # print('Number of valid sessions: {}'.format(len(valid_data)))
        print('Number of items: {}'.format(self.num_words))
        print('The Avg. length of sessions: {}'.format(self.num_histories / len(self.all_sessions_index)))
        sparsity = (self.num_histories / (len(self.all_sessions_index) * self.num_words))
        sparsity = 1 - sparsity
        sparsity_percent = sparsity * 100
        print('Sparsity: {:.2f}%'.format(sparsity_percent))
        # print('稀疏度: {}'.format(sparsity))
        print('-' * 50)

    # 保存数据
    def save(self):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        valid_x = []
        valid_y = []
        all_train_seq = []

        for session in self.train_data:
            train_x.append(session[:-1])
            train_y.append(session[-1])
            all_train_seq.append(session[:])

        self.train_set = (train_x, train_y)

        for session in self.test_data:
            test_x.append(session[:-1])
            test_y.append(session[-1])
        self.test_set = (test_x, test_y)

        for session in self.valid_data:
            valid_x.append(session[:-1])
            valid_y.append(session[-1])
        self.valid_set = (valid_x, valid_y)

        # 打开文件以写入pickle数据
        with open(self.data_path + '/train1.pkl', 'wb') as file:
            pickle.dump(self.train_set, file)

        # 打开文件以写入pickle数据
        with open(self.data_path + '/test1.pkl', 'wb') as file:
            pickle.dump(self.test_set, file)

        # 打开文件以写入pickle数据
        with open(self.data_path + '/valid1.pkl', 'wb') as file:
            pickle.dump(self.valid_set, file)

        # 打开文件以写入pickle数据
        with open(self.data_path + '/all_train_seq1.pkl', 'wb') as file:
            pickle.dump(all_train_seq, file)

    def get_data(self):
        return self.train_data, self.test_data

    def computer_item_counts(self):
        self.item_counts = {}
        for idx, session in enumerate(self.all_sessions_index):
            for item in session:
                if item in self.item_counts:
                    self.item_counts[item] += 1
                else:
                    self.item_counts[item] = 1
        print("计算每个物品被交互的次数完毕")
        return self.item_counts

    def get_extra_data(self):
        return self.item_counts


if __name__ == '__main__':
    path = '../data/'
    # cur_dataset = 'yc_BT_4'
    # data_path = path + cur_dataset
    # p = Process(cur_dataset, data_path, 'train')
    # cur_dataset = 'jdata_cd'
    # data_path = path + cur_dataset
    # p = Process(cur_dataset, data_path, 'test')
    cur_dataset = 'diginetica_x'  # 846
    data_path = path + cur_dataset
    p = Process(cur_dataset, data_path, 'test')
    print(p.category_mapping)  # 846
    # if cur_dataset.startswith('yc'):
    #     all_data, max_vid, item_cates = yoochoose_dataset.load_cd_data(data_path, type='aug', test_length=True,
    #                                                                    highfreq_only=True)
    # elif cur_dataset.startswith('jd'):
    #     all_data, max_vid, item_cates = jdata_dataset.load_cd_data(data_path, type='aug', test_length=True,
    #                                                                highfreq_only=True)
    # elif cur_dataset.startswith('digi'):
    #     all_data, max_vid, item_cates = diginetica_dataset.load_cd_data(data_path, type='aug',
    #                                                                     test_length=True,
    #                                                                     highfreq_only=True)
    # cur_all_data = all_data['test']
    # # csv_file_path = data_path + '/all_data.csv'
    # # item_to_category = save_train_data_to_csv(cur_all_data, csv_file_path)
    #
    # dataset = data_partition(cur_all_data)
    # [user_train, user_valid, user_test, usernum, itemnum] = dataset

    # train_data = path + cur_dataset + '/train_data.csv'
    # test_data = path + cur_dataset + '/test_data.csv'
    #
    # train_user_item_dict = read_csv(train_data)
    # test_user_item_dict = read_csv(test_data)
    #
    # train_user_data_list = dict_to_list(train_user_item_dict)
    # test_user_data_list = dict_to_list(test_user_item_dict)
