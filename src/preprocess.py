import copy
import json
import os
from random import random, shuffle

RATIO1 = 0.8
RATIO2 = 0.9

weapon_index_dict = {}
# w = set()

def weapon2index(weapon_list):
    # global w
    global weapon_index_dict

    res = []
    for weapon in weapon_list:
        if weapon in weapon_index_dict:
            res.append(weapon_index_dict[weapon])
        # else:
        #     w.add(weapon)
    
    return res

def process_data(data):
    # data for each player (10 in total)
    processed_data = {} # [data, label]

    prev_opponent_weapons = {}
    default_weapons = {}

    for round in range(1, 31):
        if str(round) not in data:
            break

        round_end_weapons = {}

        teams = data[str(round)]["teams"]
        for _, team in teams.items():
            players = team["players"]
            for _, player in players.items():
                player_name = player["player_name"]
                if round == 1:
                    processed_data[player_name] = []

                is_terrorist = int(player["team_number"]) == 2

                round_start = player["round_start"]
                weapon_start = round_start["weapons"].split(',')
                weapon_start = weapon2index(weapon_start)
                if round == 1 or round == 16:
                    default_weapons[player_name] = weapon_start

                # store round_end weapons for next round data
                round_end = player["round_end"]
                if not round_end["weapons"]:
                    # player is dead
                    round_end_weapons[player_name] = default_weapons[player_name]
                else:
                    round_end_weapons[player_name] = weapon2index(round_end["weapons"].split(','))

                if round == 1 or round == 16:
                    continue

                # round is not 1 or 16, add round data to result
                player_data = []
                # player's team
                player_data.append([0 if is_terrorist else 1])
                # player's weapons at round start
                player_data.append(weapon_start)
                # player's money at round start, divided by 1k for normalization
                player_data.append([round_start["account"] / 1000])
                # player's performance score at round start, divided by 10*round_num for normalization
                player_data.append([player["player_score"] / (round * 10)])
                # team vs opponent score
                T, CT = data[str(round)]["TvsCT"].split("vs")
                if is_terrorist:
                    player_data.append([int(T) / 15, int(CT) / 15])
                else:
                    player_data.append([int(CT) / 15, int(T) / 15])

                teammate_data = []
                for _, p2 in players.items():
                    if player_name == p2["player_name"]:
                        continue
                    teammate_weapons = weapon2index(p2["round_freeze_end"]["weapons"].split(','))
                    teammate_money = [p2["round_freeze_end"]["account"] / 1000]
                    teammate_score = [p2["player_score"] / (round * 10)]
                    # teammates' money, weapon and score after purchasing
                    teammate_data.append([teammate_weapons, teammate_money, teammate_score])
                
                player_data.append(teammate_data)
                    
                # opponets' data
                opponents_data = []
                for _, t2 in teams.items():
                    for _, p2 in t2["players"].items():
                        if int(p2["team_number"]) != player["team_number"]:
                            opponent_money = [p2["round_start"]["account"] / 1000]
                            opponent_score = [p2["player_score"] / (round * 10)]
                            # teammates' money score at round start, weapons last round end
                            opponents_data.append([prev_opponent_weapons[p2["player_name"]], opponent_money, opponent_score])

                player_data.append(opponents_data)

                # player's purchasing actions
                player_label = []
                for p in player["pickup"]:
                    player_label.append(p)
                player_label.sort(key=lambda x: x["timestamp"])
                player_label = [x["equip_name"] for x in player_label]
                player_label.append("End")
                player_label = weapon2index(player_label)
                
                # add data to result
                processed_data[player_name].append((player_data, player_label))

        prev_opponent_weapons = copy.deepcopy(round_end_weapons)

    # print(w)
    return processed_data

def read_dataset(data_dir):
    train_set = []
    val_set = []
    test_set = []

    global weapon_index_dict
    with open("./data/weapon_index.json") as f:
        weapon_index_dict = json.load(f)

    for file in os.listdir(data_dir):
        print(file)
        with open("./data/0-40/" + file) as f:
            data = json.load(f)

        processed_data = process_data(data) # len == 10

        # TODO: set random seed
        rand = random()
        if 0 <= rand < RATIO1:
            for _, pd in processed_data.items():
                train_set.append(pd)
        elif RATIO1 <= rand < RATIO2:
            for _, pd in processed_data.items():
                val_set.append(pd)
        else:
            for _, pd in processed_data.items():
                test_set.append(pd)
                # with open("./res.json", 'w') as f:
                #     json.dump(pd, f, indent=4)

    shuffle(train_set)
    shuffle(val_set)
    shuffle(test_set)

    print("train set: ", len(train_set), end=" ")
    print("val set: ", len(val_set), end=" ")
    print("test set: ", len(test_set))

    return train_set, val_set, test_set
