#visualize用のrecordsを作成する

from utils import write_data, move_to_idx
from game import State
import argparse
import struct

# def parse_arg():
#     parser = argparse.ArgumentParser(description="WTHOR reader.")
#     parser.add_argument("FILES", type=str, nargs="+", help="WTHOR files.")
#     return parser.parse_args()

def unpack_common_header(bytes):
    v = struct.unpack("<4bihh4b", bytes)
    common_header = {
        "file_year_upper": v[0],
        "file_year_lower": v[1],
        "file_month": v[2],
        "file_date": v[3],
        "num_games": v[4],
        "num_record": v[5],
        "game_year": v[6],
        "board_size": v[7],
        "game_type": v[8],
        "depth": v[9],
        "reserve": v[10],
    }
    return common_header

def unpack_game_header(bytes):
    v = struct.unpack("<hhhbb", bytes)
    game_header = {
        "game_id": v[0],
        "black_player_id": v[1],
        "white_player_id": v[2],
        "black_stones": v[3],
        "black_stones_theoretical": v[4],
    }
    return game_header

def unpack_play_record(bytes):
    return struct.unpack("<60b", bytes)

def read_wthor_file(file):
    with open(file, "rb") as f:
        common_header = unpack_common_header(f.read(16))
        games = []
        for i in range(common_header["num_games"]):
            game = {}
            game["header"] = unpack_game_header(f.read(8))
            game["play"] = unpack_play_record(f.read(60))
            games.append(game)
    return (common_header, games)

def show_play_record(record):
    num_alpha = ["a", "b", "c", "d", "e", "f", "g", "h"]
    record_str = []
    for move in record:
        #upperとlowerの順番に注意
        upper = move // 10
        lower = move % 10
        record_str.append("{}{}".format(num_alpha[lower - 1], upper))
    return record_str

def read_wthor_files(files):
    records = []
    for file in files:
        (_, games) = read_wthor_file(file)
        for game in games:
            records.append(show_play_record(game["play"]))
    return records

# # １試合を4試合分に増加させる
# # 90度回転したものでは、実際に存在しない盤面が発生するので注意
# def augment_records(records):
#     #増加させる際の対応表を2種類作成('0'→'a, '0'→'8')
#     reverse_dict, rotate_dict = {}, {}
#     for i in range(8):
#         reverse_dict[str(i+1)] = chr(97+i)
#         reverse_dict[chr(97+i)] = str(i+1)

#         rotate_dict[str(i+1)] = str(8-i)
#         rotate_dict[chr(97+i)] = chr(97+(7-i))

#     #斜め45度を軸に反転
#     def reverse_move(move):
#         return reverse_dict[move[1]] + reverse_dict[move[0]]
#     #中心を軸に180度回転
#     def rotate_move(move):
#         return rotate_dict[move[0]] + rotate_dict[move[1]]

#     augmented_records = []
#     for i, record in enumerate(records):
#         reversed_record, rotated_record, reversed_rotated_record = [], [], []
#         for move in record:
#             if move != 'h0':
#                 reversed_record.append(reverse_move(move))
#                 rotated_move = rotate_move(move)
#                 rotated_record.append(rotated_move)
#                 reversed_rotated_record.append(reverse_move(rotated_move))
#             else:
#                 reversed_record.append('h0')
#                 rotated_record.append('h0')
#                 reversed_rotated_record.append('h0')
  
#         augmented_records.append(record)
#         augmented_records.append(reversed_record)
#         augmented_records.append(rotated_record)
#         augmented_records.append(reversed_rotated_record)
        
#         print('\rMakeRecords {} / {}'.format(i+1, len(records)), end='')
#     print('')
#     return augmented_records

if __name__ == "__main__":
    records = read_wthor_files(['./data/wthor/WTH_2017.wtb'])

    save_path = './data/records2017.pickle'
    write_data(records, save_path)