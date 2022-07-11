from utils import make_input
from math import sqrt
import random
import torch
import numpy as np

# パラメータの準備
EVALUATE_COUNT = 160 # 1推論あたりのシミュレーション回数（本家は1600）

# 推論
def predict(model, state):
    board = make_input(state.return_board())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    #バッチ形式用に次元を調節して入力する
    board = torch.tensor([board]).long().to(device)
    y = model(board, mask=None, attention_show_flg=False, output_softmax=True)

    # 方策の取得
    # 合法手のみ
    policies = y[0][0].cpu().detach().numpy()[state.legal_actions()]
    # 合計1の確率分布に変換
    policies /= sum(policies) if sum(policies) else 1 

    # 価値の取得
    value = y[1][0].cpu().detach().numpy()
    return policies, value

# ノードのリストを試行回数のリストに変換
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

# モンテカルロ木探索のスコアの取得
def pv_mcts_scores(model, state, root_node=None):
    # モンテカルロ木探索のノードの定義
    class Node:
        # ノードの初期化
        def __init__(self, state, p):
            self.state = state # 状態
            self.p = p # 方策
            self.w = 0 # 累計価値
            self.n = 0 # 試行回数
            self.child_nodes = None  # 子ノード群

        # 局面の価値の計算
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # ニューラルネットワークの推論で方策と価値を取得
                policies, value = predict(model, self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(Node(self.state.next(action), policy))
                return value

            # 子ノードが存在する時
            else:
                # アーク評価値が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # アーク評価値が最大の子ノードを取得
        def next_child_node(self):
            # アーク評価値の計算
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                    C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

            # アーク評価値が最大の子ノードを返す
            index = random.choice(np.where(pucb_values==np.max(pucb_values))[0])
            return self.child_nodes[index]

    # 現在の局面のノードの作成
    if root_node is None or root_node.child_nodes == []:
        root_node = Node(state, 0)
        num_simulations = EVALUATE_COUNT
    else:
        num_simulations = EVALUATE_COUNT - sum(nodes_to_scores(root_node.child_nodes))

    # 複数回の評価の実行
    for _ in range(num_simulations):
        root_node.evaluate()

    # 合法手の確率分布
    scores = nodes_to_scores(root_node.child_nodes)
    #合計を1に
    scores = [n / sum(scores) for n in scores]

    return scores, root_node.child_nodes

#self-playのみ
# def pv_mcts_action_v(model, temperature=0):
#     def pv_mcts_action(state, root_node=None):
#         scores, child_nodes = pv_mcts_scores(model, state, temperature, root_node)
#         legal_actions = state.legal_actions()
#         idx= np.random.choice(np.arange(len(legal_actions)), p=scores)
#         return legal_actions[idx], child_nodes[idx]
#     return pv_mcts_action

#従来のものと同様に動作する
#selfーplayでは他のものを使用する
def pv_mcts_action(model):
    def pv_mcts_action(state):
        scores, _ = pv_mcts_scores(model, state)
        #実質的にtemperature=0,確率が最大の手を選択
        action = random.choice(np.where(scores==np.max(scores))[0])
        scores = np.zeros(len(scores))
        scores[action] = 1
        return np.random.choice(state.legal_actions(), p=scores)
    return pv_mcts_action

# ボルツマン分布
# def boltzman(xs, temperature):
#     xs = [x ** (1 / temperature) for x in xs]
#     return [x / sum(xs) for x in xs]

# 動作確認
# if __name__ == '__main__':
#     # モデルの読み込み
#     path = './model/best_transformer.pth'
#     model_type = discern_model(path)
#     model = load_model(path, model_type)

#     # 状態の生成
#     state = State()

#     # モンテカルロ木探索で行動取得を行う関数の生成
#     next_action = pv_mcts_action(model, model_type , 1.0)

#     # ゲーム終了までループ
#     while True:
#         # ゲーム終了時
#         if state.is_done():
#             break

#         # 行動の取得
#         action = next_action(state)

#         # 次の状態の取得
#         state = state.next(action)

#         # 文字列表示
#         print(state)