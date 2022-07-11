from game import State
from pv_mcts import pv_mcts_action
from utils import load_model
import tkinter as tk
import random
import argparse

def parse_arg():
    parser = argparse.ArgumentParser(description='You play with Model.')
    parser.add_argument('MODEL', type=str, help='Model.')
    return parser.parse_args()

# ゲームUIの定義
class GameUI(tk.Frame):
    # 初期化
    def __init__(self, master=None, model=None, human_first=None):
        tk.Frame.__init__(self, master)
        self.master.title('リバーシ')

        # ゲーム状態の生成
        self.state = State()
        
        # 行動選択を行う関数の生成
        self.next_action = pv_mcts_action(model)

        #先行後攻を決める
        #指定がない場合はランダムに、ある場合は従う
        self.is_random = False
        if human_first is None:
            self.human_first = random.randint(0, 1)
            self.is_random = True
        elif human_first:
            self.human_first = 1
        else:
            self.human_first = 0

        # キャンバスの生成
        self.c = tk.Canvas(self, width = 320, height = 360, highlightthickness = 0)
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.focus_set()
        self.c.bind('<Left>', self.previous_board)
        self.c.bind('<Return>', self.restart_game)
        self.c.pack()

        # 描画の更新
        self.on_draw()

    # 人間のターン
    def turn_of_human(self, event):
        # ゲーム終了時
        if self.state.is_done():
            if self.is_random:
                self.human_first = random.randint(0, 1)
            self.state = State()
            self.on_draw()
            return

        #先行後攻は全体を通してもの、先手後手はある盤面状態のおけるターンのこと（次の手を打つのはどっちか）
        #先行の時
        if self.human_first:
            #先手ではないとき（　AIの思考中）はクリックしても無反応に
            if not self.state.is_first_player():
                return 
        #後攻の時
        else:
            #後攻で初めてクリックした時はAIにターンを渡す
            if self.state.pieces.count(1) == 2 & self.state.enemy_pieces.count(1) == 2:
                self.master.after(1, self.turn_of_ai)
            #後手ではないとき()はクリックしでも無反応
            if self.state.is_first_player():
                return

        #盤面外のクリック時
        if event.y > 320:
            return

         # クリック位置を行動に変換
        x = int(event.x/40)
        y = int(event.y/40)
        if x < 0 or 7 < x or y < 0 or 7 < y: # 範囲外
            return
        action = x + y * 8

        # 合法手でない時
        legal_actions = self.state.legal_actions()
        if legal_actions == [64]:
            action = 64 # パス
        if action != 64 and not (action in legal_actions):
            return

        # 次の状態の取得
        self.state = self.state.next(action)
        self.on_draw()

        # AIのターン
        self.master.after(1, self.turn_of_ai)

    # AIのターン
    def turn_of_ai(self):
        # ゲーム終了時
        if self.state.is_done():
            return

        # 行動の取得
        action = self.next_action(self.state)

        # 次の状態の取得
        self.state = self.state.next(action)
        self.on_draw()

    def previous_board(self, _):
        if self.state.piece_count(self.state.pieces) + self.state.piece_count(self.state.enemy_pieces) <= 5:
            self.on_draw()
            return
        
        if self.human_first:
            if not self.state.is_first_player():
                return 
        else:
            if self.state.is_first_player():
                return

        self.state.pieces_history.pop()
        self.state.enemy_pieces_history.pop()
        self.state.pieces  = self.state.pieces_history.pop()
        self.state.enemy_pieces = self.state.enemy_pieces_history.pop()
        self.state.depth += 2

        self.on_draw()

    def restart_game(self, _):
        if self.is_random:
            self.human_first = random.randint(0, 1)
        self.state = State()
        self.on_draw()

    # 石の描画
    def draw_piece(self, index, first_player):
        x = (index%8)*40+5
        y = int(index/8)*40+5
        if first_player:
            self.c.create_oval(x, y, x+30, y+30, width = 1.0, outline = '#000000', fill = '#000000')
        else:
            self.c.create_oval(x, y, x+30, y+30, width = 1.0, outline = '#000000', fill = '#FFFFFF')

    #合法手の描画
    def draw_legal_action(self, index):
        x = (index%8)*40+(20-2)
        y = int(index/8)*40+(20-2)
        self.c.create_oval(x, y, x+4, y+4, width= 1.0, outline = '#C0C0C0', fill = '#C0C0C0')

    # 描画の更新
    def on_draw(self):
        self.c.delete('all')
        self.c.create_rectangle(0, 0, 320, 320, width = 0.0, fill = '#105511')
        for i in range(1, 10):
            self.c.create_line(0, i*40, 320, i*40, width = 1.0, fill = '#000000')
            self.c.create_line(i*40, 0, i*40, 320, width = 1.0, fill = '#000000')
        for i in range(64):
            if self.state.pieces[i] == 1:
                self.draw_piece(i, self.state.is_first_player())
            if self.state.enemy_pieces[i] == 1:
                self.draw_piece(i, not self.state.is_first_player())
        
        # 合法手の表示
        if self.state.legal_actions() != [64]:
            for index in self.state.legal_actions():
                self.draw_legal_action(index)
        #下部の表示
        self.c.create_line(160, 320, 160, 360, width = 1.0, fill = '#000000')
        self.c.create_oval(5, 325, 5+30, 325+30, width = 1.0, outline = '#000000', fill = '#000000')
        self.c.create_oval(165, 325, 165+30, 325+30, width = 1.0, outline = '#000000', fill = '#FFFFFF')
        #石の数
        my_pieces_num = self.state.pieces.count(1)
        enemy_pieces_num = self.state.enemy_pieces.count(1)
        if self.state.is_first_player():
            self.c.create_text(120, 340, text=str(my_pieces_num), font=('', 30))
            self.c.create_text(280, 340, text=str(enemy_pieces_num), font=('', 30))
            self.c.create_line(0, 358, 160, 358, width=3.0, fill='#FF0000')
        else:
            self.c.create_text(120, 340, text=str(enemy_pieces_num), font=('', 30))
            self.c.create_text(280, 340, text=str(my_pieces_num), font=('', 30))
            self.c.create_line(160, 358, 320, 358, width=3.0, fill='#FF0000')

        #you,aiの表示
        if self.human_first:
            self.c.create_text(40, 340, text='(YOU)', font=('', 20), anchor=tk.W)
            self.c.create_text(200, 340, text='(AI)', font=('', 20), anchor=tk.W)
        else:
            self.c.create_text(40, 340, text='(AI)', font=('', 20), anchor=tk.W)
            self.c.create_text(200, 340, text='(YOU)', font=('', 20), anchor=tk.W)


if __name__ == '__main__':

    args = parse_arg()
    model_path = args.MODEL
    model = load_model(model_path)

    f = GameUI(model=model, human_first=None)
    f.pack()
    f.mainloop()