from typing import Dict, List
import numpy as np
import csv


def data_statistics(all_data: Dict[str, List[List[List]]]):
    '''
    get statistics from data

    Args:
        data (List[List[List]]): list of [0, [item], [next_item](1), [category], [next_category](1)]
    '''
    data = all_data['train'] + all_data['test']
    items = set()
    total_session_length = 0
    cats = set()
    total_cat_per_session = 0

    for x in data:
        total_session_length += len(x[1])
        for i in x[1]:
            items.add(i)
        items.add(x[2][0])
        for c in x[3]:
            cats.add(c)
        cats.add(x[4][0])

        total_cat_per_session += len(np.unique(x[3]))

    print('')
    print('* dataset statistics:')
    print('=====================')
    print('No. of items: {}'.format(len(items)))
    print('No. of sessions: {}'.format(len(data)))
    print('Avg. of session length: {}'.format(total_session_length / len(data)))
    print('No. of categories: {}'.format(len(cats)))
    print('No. of cats/session: {}'.format(total_cat_per_session / len(data)))
    print('')


def save_train_data_to_csv(train_data, csv_file_path):
    item_to_category = {}  # 用于存储物品ID到类别ID的映射
    item_to_new_id = {}  # 用于存储物品ID到新ID的映射
    next_new_id = 1  # 新ID从1开始
    all_rows = []  # 用于存储所有行

    # train_data = train_data[:10000]

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['session_id', 'item_id', 'category_id'])  # 写入表头
        for index, session in enumerate(train_data):
            item_ids = session[1]  # out_seqs
            item_ids.append(session[2][0])
            category_ids = session[3]  # out_behavs
            category_ids.append(session[4][0])

            for item_id, category_id in zip(item_ids, category_ids):
                # writer.writerow([index + 1, item_id, category_id])
                if item_id not in item_to_new_id:
                    item_to_new_id[item_id] = next_new_id
                    next_new_id += 1
                new_item_id = item_to_new_id[item_id]
                item_to_category[new_item_id] = category_id  # 更新映射字典
                # all_rows.append([index + 1, new_item_id, category_id])  # 添加到列表
                all_rows.append([index + 1, new_item_id])  # 添加到列表

        # 写入所有行
        writer.writerows(all_rows)
