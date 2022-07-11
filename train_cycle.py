from self_play import self_play
from train_network import train_network
# from evaluate_network import evaluate_network
from utils import discern_model, load_model, write_data
import argparse
import re
import os
import torch
import ray

#実際には分割して合計16回行う
# NUM = 16

# def parse_arg():
#     parser = argparse.ArgumentParser(description='Train model by self play.')
#     parser.add_argument('MODEL', type=str, help='model.')
#     return parser.parse_args()

#1回のサイクル
def train_cycle(model_path, num_cycle):
    model = load_model(model_path)
    # これまでの学習回数をpathから判別
    l = re.findall(r"\d+", model_path)
    cnt = int(l[-1])

    directory = ['./data/history/', './data/record/', './model/']
    for i in directory:
        os.makedirs(i, exist_ok=True)

    ray.init(num_gpus=1)
    for i in range(num_cycle):
        print('Train{}/{}===================='.format(i+1, num_cycle))

        # セルフプレイ部
        history = self_play(model)

        path = directory[0] + 'transformer' + str(cnt) + '.his'
        write_data(history, path)

        # パラメータ更新部
        model, record = train_network(model, history)

        path = directory[1] + 'transformer' + str(cnt) + '.rec'
        write_data(record, path)

        cnt += 1

        path = directory[2] + 'transformer' + str(cnt) + '.pth'
        torch.save(model.state_dict(), path)

# if __name__ == '__main__':

#     args = parse_arg()
#     model_path = args.MODEL

#     model_type = discern_model(model_path)
#     # new_model_path = './model/latest_' + model_type + '.pth'

#     for i in range(NUM):
#         print('Train{}/{}===================='.format(i+1, NUM))

#         # セルフプレイ部
#         self_play(model_path)

#         # パラメータ更新部
#         train_network(model_path)

#         # 新パラメータ評価部
#         # evaluate_network(new_model_path, best_model_path)

#         if model_path[-5].isdecimal():
#             next = int(model_path[-5]) + 1
#             model_path = model_path[:-5] + str(next) + model_path[-4:]
#         else:
#             save_path = model_path