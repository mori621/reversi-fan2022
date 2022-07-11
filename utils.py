import pickle
import sys
import torch
# from network_resnet import ResNet
from network_transformer import PolicyValueTransfomer
# from network_bert import config, BertNextAction
from datetime import datetime

def load_data(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)

def write_data(dataset, path):
    with open(path, mode='wb') as f:
        pickle.dump(dataset, f)

# 指手をstate用のインデックスに変換('a1'→0)
def move_to_idx(move):
    i = ord(move[0]) - ord('a')
    j = int(move[1])
    return i + (j - 1) * 8

def idx_to_move(idx):
    i = str(idx // 8 + 1)
    j = chr(idx % 8 + 97)
    return j + i

#pathの種類の半別
def discern_model(path):
    if 'resnet' in path:
        return 'resnet'
    elif 'transformer' in path:
        return 'transformer'
    # elif 'bert' in path:
    #     return 'bert'
    else:
        print('Model Path is incorrect.')
        sys.exit()

#モデルの読込
def load_model(path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PolicyValueTransfomer().to(device)

    if device == 'cuda':
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

# def enemy_to_2(board):
#     return [2 if i == -1 else i for i in board]

def to_2ch(board):
    my_pieces, enemy_pieces = [0]*64, [0]*64
    for i, piece in enumerate(board):
        if piece == 1:
            my_pieces[i] = 1
        elif piece == -1:
            enemy_pieces[i] = 1
    return my_pieces, enemy_pieces

def to_2d(board, col=8):
    return [board[i: i+col] for i in range(0, len(board), col)]

def make_input(board):
    #clsとeosを消去していることに注意
    # my: 1, enemy: 2, space: 0に変換
    return [2 if i == -1 else i for i in board]

def make_path(directory, extension):
    now = datetime.datetime.now()
    time = '{:04}{:02}{:02}{:02}{:02}{:02}'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    return directory + time + extension