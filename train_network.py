
from utils import load_model, make_input, discern_model, load_data, write_data, make_path
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
# from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
import time
import os

import torch_optimizer as optim

# パラメータの準備
EPOCHS = 100 # 学習回数 (元は100)
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

# Datasetの作成
class ClassifierDataset(Dataset): 
    def __init__(self, boards, policies, values, attention=None):
        self.boards = boards
        self.policies = policies
        self.values = values
        self.attention = attention
        
    def __getitem__(self, index):
        if self.attention is None:
            return self.boards[index], self.policies[index], self.values[index]
        else: 
            return self.boards[index], self.policies[index], self.values[index], self.attention[index]

    def __len__ (self):
        return len(self.boards)
    
#DataLoaderの作成
def make_dataloader(boards, policies, values, shuffle=True):
    # 入力を変換
    new_boards = []
    for board in boards:
        new_boards.append(make_input(board))

    policies = torch.tensor(policies).float()
    values = torch.tensor(values).float()

    new_boards = torch.tensor(new_boards).long()
    dataset = ClassifierDataset(new_boards, policies, values)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
    return dataloader

def make_dataloader_dict(dataset):
    boards, policies, values = zip(*dataset)

    boards_train, boards_val, policies_train, policies_val, values_train, values_val \
        = train_test_split(boards, policies, values, test_size=0.1, random_state=0)

    train_dataloader = make_dataloader(boards_train, policies_train, values_train, shuffle=True)
    val_dataloader = make_dataloader(boards_val, policies_val, values_val, shuffle=False)

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}
    return dataloader_dict

# デュアルネットワークの学習
def train_network(model, history):

    #要確認
    # history_path = sorted(Path('./data/' + model_type + '_history').glob('*.history'))[-1]
    dataloader_dict = make_dataloader_dict(history)
    # model = load_model(model_path, model_type)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # モデルのコンパイル
    pol_criterion = nn.CrossEntropyLoss()
    val_criterion = nn.MSELoss()
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-5)
    # sheduler = MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

    record_dict = {}
    # record_dict['model'] = model_type
    record_dict['epochs'] = EPOCHS
    record_dict['batch_size'] = BATCH_SIZE
    record_dict["learning_rate"] = LEARNING_RATE
    record_dict['time'] = []
    record_dict['train_loss'], record_dict['val_loss']  = [], []
    record_dict['train_pol_loss'], record_dict['train_val_loss'] = [], []
    record_dict['val_pol_loss'], record_dict['val_val_loss'] = [], []

    # lrを取得するための関数
    # def get_lr(optimizer):
    #     for param_group in optimizer.param_groups:
    #         return param_group['lr']

    for epoch in range(EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                train_epoch_loss, train_epoch_pol_loss, train_epoch_val_loss = 0, 0, 0
                start = time.time()
            else:
                model.eval()
                val_eopch_loss, val_epoch_pol_loss, val_epoch_val_loss = 0, 0, 0
            
            for boards, policies, values in dataloader_dict[phase]:
                boards = boards.to(device)
                policies = policies.to(device)
                values = values.to(device)

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    pol_out, val_out = model(boards, mask=None, attention_show_flg=False)

                    pol_loss = pol_criterion(pol_out, policies)
                    val_loss = val_criterion(val_out, values)

                    loss = pol_loss + val_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        train_epoch_loss += loss.item() * boards.size(0)
                        train_epoch_pol_loss += pol_loss.item() * boards.size(0)
                        train_epoch_val_loss += val_loss.item() * boards.size(0)
                    else:
                        val_eopch_loss += loss.item() * boards.size(0)
                        val_epoch_pol_loss += pol_loss.item() * boards.size(0)
                        val_epoch_val_loss += val_loss.item() * boards.size(0)
    
            if phase == 'train':
                train_epoch_loss /= len(dataloader_dict[phase].dataset)
                train_epoch_pol_loss /= len(dataloader_dict[phase].dataset)
                train_epoch_val_loss /= len(dataloader_dict[phase].dataset)
            else:
                val_eopch_loss /= len(dataloader_dict[phase].dataset)
                val_epoch_pol_loss /= len(dataloader_dict[phase].dataset)
                val_epoch_val_loss /= len(dataloader_dict[phase].dataset)
                
        print('Epoch {}/{} | Time: {}s | train_loss: {:.4f}, val_loss: {:.4f}'.format( \
            epoch+1, EPOCHS, int(time.time()-start), train_epoch_loss, val_eopch_loss))

        # sheduler.step()

        record_dict['time'].append(int(time.time()-start))
        record_dict['train_loss'].append(train_epoch_loss)
        record_dict['train_pol_loss'].append(train_epoch_pol_loss)
        record_dict['train_val_loss'].append(train_epoch_val_loss)
        record_dict['val_loss'].append(val_eopch_loss)
        record_dict['val_pol_loss'].append(val_epoch_pol_loss)
        record_dict['val_val_loss'].append(val_epoch_val_loss)

    # model_save_path = './model/latest_' + model_type + '.pth'
    # torch.save(model.state_dict(), model_save_path)
    
    # if model_path[-5].isdecimal():
    #     next = int(model_path[-5]) + 1
    #     save_path = model_path[:-5] + str(next) + model_path[-4:]
    # else:
    #     save_path = model_path
    # torch.save(model.state_dict(), save_path)


    # directory = './record/sp_{}/'.format(model_type)
    # os.makedirs(directory, exist_ok=True)
    # record_save_path = make_path(directory, '.record')
    # write_data(record_dict, record_save_path)

    # print('')

    # モデルの破棄
    # del model
    # torch.cuda.empty_cache()

    return model, record_dict

# 動作確認
# if __name__ == '__main__':
#     model_path = './model/best_transformer.pth'
#     train_network(model_path)
