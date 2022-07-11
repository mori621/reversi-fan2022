from game import State
from pv_mcts import pv_mcts_scores
# from utils import write_data, make_path
# import torch
import numpy as np
import os
import random
import ray

# パラメータの準備
GAME_COUNT = 500 # セルフプレイを行うゲーム数（本家は25000）
# TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ
NUM_PARALLEL = 20

#反転用のindexのlist
REVERSE = [8 * j + i for i in range(8) for j in range(8)]
#180回転用
ROTATE = [64 - i - 1  for i in range(64)]
#反転+回転用
REVERSE_ROTATE = [8 * (8 - j) - i - 1 for i in range(8) for j in range(8)]

# 先手プレイヤーの価値
def first_player_value(ended_state):
    # 1:先手勝利, -1:先手敗北, 0:引き分け
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# 1ゲームの実行
@ray.remote(num_gpus=0.05)
def play(model):

    # 学習データ
    history, reverse_history = [], []
    rotate_history, reverse_rotate_history = [], []

    # 状態の生成
    state = State()
    root_node = None
    cnt = 0
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 合法手の確率分布の取得
        scores, child_nodes = pv_mcts_scores(model, state, root_node)

        # 学習データに状態と方策を追加
        policies = [0] * 65
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        board = state.return_board()
        history.append([board, policies, None])
        reverse_history.append([[board[i] for i in REVERSE], [policies[i] for i in REVERSE] + [policies[64]], None])
        rotate_history.append([[board[i] for i in ROTATE], [policies[i] for i in ROTATE] + [policies[64]], None])
        reverse_rotate_history.append([[board[i] for i in REVERSE_ROTATE], [policies[i] for i in REVERSE_ROTATE] + [policies[64]], None])

        # 行動の取得
        if cnt < 20:
            #temperature=1
            legal_actions = state.legal_actions()
            idx = np.random.choice(np.arange(len(legal_actions)), p=scores)
            action = legal_actions[idx]
        else:
            #実質的にtemperature=0,確率が最大の手を選択
            idx = random.choice(np.where(scores==np.max(scores))[0])
            action = state.legal_actions()[idx]
        
        root_node = child_nodes[idx]

        # 次の状態の取得
        state = state.next(action)

    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        reverse_history[i][2] = value
        rotate_history[i][2] = value
        reverse_rotate_history[i][2] = value
        value = -value
    return history + reverse_history + rotate_history + reverse_rotate_history

# セルフプレイ
def self_play(model):

    model_id = ray.put(model)
    work_in_progresses = [play.remote(model_id) for _ in range(NUM_PARALLEL)]

    # 学習データ
    history = []

    # 複数回のゲームの実行
    for i in range(GAME_COUNT):

        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        history.extend(ray.get(finished[0]))
        work_in_progresses.extend([play.remote(model_id)])
        # 1ゲームの実行
        # h = play(model)
        # history.extend(h)

        # 出力
        print('\rSelfPlay {}/{}'.format(i+1, GAME_COUNT), end='')
    print('')

    # 学習データの保存
    # directory = './data/history/'
    # os.makedirs(directory, exist_ok=True) # フォルダがない時は生成
    # path = make_path(directory, '.his')
    # write_data(history, path)

    # モデルの破棄
    # del model
    # torch.cuda.empty_cache()

    return history

# 動作確認
# if __name__ == '__main__':
#     # trainedでも
#     model_path = './model/best_transformer.pth'
#     self_play(model_path)
