import os
import pickle
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, BatchSampler, SequentialSampler
from utils import split_validation, data_masks, Data
import pytorch_lightning as pl


# 从文件中读取对象
def pick_load(dir):
    with open(dir, 'rb') as f:
        obj = pickle.load(f)
    return obj


class SessionData(pl.LightningDataModule):

    def __init__(self, data_dir='../../datasets/data/', name='gowalla', validation=False, batch_size=100):
        super(SessionData, self).__init__()
        self.validation = validation
        self.batch_size = batch_size
        # self.data_path = os.path.join(data_dir, name)
        self.data_name = name

        self.path = '../datasets/data/'
        self.data_path = self.path + self.data_name

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            # self.train_data = pickle.load(open('../data/gowalla/train.txt', 'rb'))
            self.train_data = pick_load(self.data_path + '/train1.pkl')
            if self.validation:
                self.train_data, self.valid_data = split_validation(self.train_data, 0.1)
            else:
                self.valid_data = pick_load(self.data_path + '/test1.pkl')
                # self.valid_data = pickle.load(open('../data/gowalla/test.txt', 'rb'))

            # (valid_x, valid_y) = self.valid_data
            # print(valid_x[:3], valid_y[:3])
            self.train_data = Data(self.train_data, shuffle=True)
            self.valid_data = Data(self.valid_data, shuffle=True)

        if stage == 'test' or stage is None:
            # self.test_data = pickle.load(open('../data/gowalla/test.txt', 'rb'))
            self.test_data = pick_load(self.data_path + '/test1.pkl')
            self.test_data = Data(self.test_data, shuffle=True)

    def train_dataloader(self):
        sampler = BatchSampler(SequentialSampler(self.train_data), batch_size=self.batch_size, drop_last=False)

        return DataLoader(self.train_data, sampler=sampler, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        sampler = BatchSampler(SubsetRandomSampler(range(len(self.valid_data))), batch_size=self.batch_size,
                               drop_last=False)

        return DataLoader(self.valid_data, sampler=sampler, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        self.test_data = pick_load(self.data_path + '/test1.pkl')
        # self.test_data = pickle.load(open('../data/gowalla/test.txt', 'rb'))
        self.test_data = Data(self.test_data, shuffle=True)
        sampler = BatchSampler(SubsetRandomSampler(range(len(self.test_data))), batch_size=self.batch_size,
                               drop_last=False)

        return DataLoader(self.test_data, sampler=sampler, num_workers=4, pin_memory=True)


if __name__ == "__main__":

    # data = SessionData()
    # data.setup()
    # val_loader = data.val_dataloader()
    # train_loader = data.train_dataloader()

    session_data = SessionData(name='yc_BT_4', batch_size=100)
    session_data.setup()

