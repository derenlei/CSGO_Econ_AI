import copy
import json
import numpy as np
import os
import random

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
        else:
            if weapon == "Five-Seven":
                # a misspelled word
                res.append(weapon_index_dict["Five-SeveN"])
            # else:
            #     w.add(weapon)
    
    return res

def process_data(data):
    # data for each player (10 in total)
    processed_data = {} # [data, label]

    prev_opponent_weapons = {}
    default_weapons = {}

    for round in range(1, 31):
        # print(round)
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

                if player["team_number"] is None:
                    return None

                is_terrorist = int(player["team_number"]) == 2

                round_start = player["round_start"]
                weapon_start = round_start["weapons"].split(',')
                if round_start["has_defuser"]:
                    weapon_start.append("defuser")
                if round_start["armor"] > 0:
                    if round_start["has_helmet"]:
                        weapon_start.append("vesthelm")
                    else:
                        weapon_start.append("vest")
                weapon_start = weapon2index(weapon_start)
                if round == 1 or round == 16:
                    default_weapons[player_name] = weapon_start

                # store round_end weapons for next round data
                round_end = player["round_end"]
                if round_end["weapons"] is None:
                    # player is dead
                    round_end_weapons[player_name] = default_weapons[player_name]
                else:
                    weapon_end = round_end["weapons"].split(',')
                    if round_end["has_defuser"]:
                        weapon_end.append("defuser")
                    if round_end["armor"] > 0:
                        if round_end["has_helmet"]:
                            weapon_end.append("vesthelm")
                        else:
                            weapon_end.append("vest")
                    round_end_weapons[player_name] = weapon2index(weapon_end)

                if round == 1 or round == 16:
                    continue

                # round is not 1 or 16, add round data to result only if data is valid
                player_data = []
                # player's team, 0 for terrorist and i for terrorist
                player_data.append([0 if is_terrorist else 1])
                # player's weapons at round start
                player_data.append(weapon_start)
                # player's money at round start, divided by 1k for normalization
                player_data.append([int(round_start["account"]) / 1000])
                # player's performance score at round start, divided by 10*round_num for normalization
                player_data.append([int(round_start["player_score"]) / (round * 10)])
                # team vs opponent score
                if data[str(round)]["TvsCT"] is None or not isinstance(data[str(round)]["TvsCT"], str):
                    # data anomaly 
                    continue

                T, CT = data[str(round)]["TvsCT"].split("vs")
                if is_terrorist:
                    player_data.append([int(T) / 15, int(CT) / 15])
                else:
                    player_data.append([int(CT) / 15, int(T) / 15])

                teammate_data = []
                valid = True
                for _, p2 in players.items():
                    if player_name == p2["player_name"]:
                        continue

                    if p2["round_freeze_end"]["weapons"] is None:
                        # data anomaly 
                        valid = False
                        break
                        
                    weapon_freeze_end = p2["round_freeze_end"]["weapons"].split(',')
                    if p2["round_freeze_end"]["has_defuser"]:
                        weapon_freeze_end.append("defuser")
                    if p2["round_freeze_end"]["armor"] > 0:
                        if p2["round_freeze_end"]["has_helmet"]:
                            weapon_freeze_end.append("vesthelm")
                        else:
                            weapon_freeze_end.append("vest")
                    teammate_weapons = weapon2index(weapon_freeze_end)
                    teammate_money = [int(p2["round_freeze_end"]["account"]) / 1000]
                    if p2["round_start"]["player_score"] is None:
                        # data anomaly 
                        valid = False
                        break
                        
                    teammate_score = [int(p2["round_start"]["player_score"]) / (round * 10)]
                    # teammates' money, weapon and score after purchasing
                    teammate_data.append([teammate_weapons, teammate_money, teammate_score])
                
                if not valid:
                    continue
                player_data.append(teammate_data)
                    
                # opponets' data
                valid = True
                opponents_data = []
                for _, t2 in teams.items():
                    for _, p2 in t2["players"].items():
                        if p2["team_number"] is None:
                            valid = False
                            break

                        if int(p2["team_number"]) != int(player["team_number"]):
                            opponent_money = [int(p2["round_start"]["account"]) / 1000]
                            opponent_score = [int(p2["round_start"]["player_score"]) / (round * 10)]
                            # teammates' money score at round start, weapons last round end
                            opponents_data.append([prev_opponent_weapons[p2["player_name"]], opponent_money, opponent_score])

                if not valid:
                    continue
                player_data.append(opponents_data)

                # player's purchasing actions
                player_label = []
                for p in player["pickup"]:
                    if p["possibly_get_from"] is None:
                        # currently only consider purchase, no pickup
                        player_label.append(p)
                if len(player_label) > 10:
                    # might be a noisy data
                    continue

                player_label.sort(key=lambda x: x["timestamp"])
                player_label = [x["equip_name"] for x in player_label]
                player_label.append("End")
                player_label = weapon2index(player_label)
                
                # add data to result
                processed_data[player_name].append((player_data, player_label))

        prev_opponent_weapons = copy.deepcopy(round_end_weapons)

    return processed_data

def read_dataset(data_dir):
    # print(data_dir)
    processed_dir = data_dir[:-4] + "p.npy"
    # print(processed_dir)
    if os.path.exists(processed_dir):
        dataset = np.load(processed_dir)
        train_set, val_set, test_set = dataset

        print("train set: ", len(train_set), end=" ")
        print("val set: ", len(val_set), end=" ")
        print("test set: ", len(test_set))

        return train_set, val_set, test_set

    global weapon_index_dict
    with open("./data/weapon_index.json") as f:
        weapon_index_dict = json.load(f)

    data = np.load(data_dir)

    processed_data = []
    for match in data:
        match_data = process_data(match) # len == 10
        if match_data is None:
            continue

        processed_data.append(match_data)

    random.seed(4164)
    random.shuffle(processed_data)

    train_set = []
    val_set = []
    test_set = []

    total = len(processed_data)
    for i, match_data in enumerate(processed_data):
        if 0 <= i < int(RATIO1 * total):
            for _, md in match_data.items():
                train_set.append(md)
        elif int(RATIO1 * total) <= i < int(RATIO2 * total):
            for _, md in match_data.items():
                val_set.append(md)
        else:
            for _, md in match_data.items():
                test_set.append(md)

    print("train set: ", len(train_set), end=" ")
    print("val set: ", len(val_set), end=" ")
    print("test set: ", len(test_set))

    # global w
    # print(w)
    # with open('./res.json', 'w') as f:
    #     json.dump(train_set[0], f, indent=4)

    np.save(processed_dir, (train_set, val_set, test_set))

    return train_set, val_set, test_set
