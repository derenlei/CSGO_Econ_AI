from src.preprocess import read_dataset
from src.utils import *

import json
import numpy as np
from tqdm import tqdm


DATA_DIR = 'data/2600-15156.npy'

def greedy(is_terrorist, weapons, money, w_attr, grenades, weapon_index):
    # current weapons
    grenades_start = []
    primary_weapon_start = None
    for weapon in weapons:
        if 1 <= w_attr["type"][weapon] <= 5:
            primary_weapon_start = weapon
        if w_attr["type"][weapon] == 6:
            grenades_start.append(weapon)

    purchase_list = []
    # Buy primary weapon
#     if is_terrorist and money >= w_attr["price"][weapon_index["AK-47"]]:
#     	if primary_weapon_start is None or w_attr["price"][weapon_index["AK-47"]] > w_attr["price"][primary_weapon_start]:
#     		money -= w_attr["price"][weapon_index["AK-47"]]
#     		purchase_list.append(weapon_index["AK-47"])
#     if not is_terrorist and money >= w_attr["price"][weapon_index["M4A4"]]:
#     	if primary_weapon_start is None or w_attr["price"][weapon_index["M4A4"]] > w_attr["price"][primary_weapon_start]:
#     		money -= w_attr["price"][weapon_index["M4A4"]]
#     		purchase_list.append(weapon_index["M4A4"])

    max_price = 0
    max_price_weapon = primary_weapon_start
    for w in range(44):
        if 1 <= w_attr["type"][w] <= 5 and max_price < w_attr["price"][w] <= money:
            if is_terrorist and w_attr["t"][w] == 1:
                max_price = w_attr["price"][w]
                max_price_weapon = w
            if not is_terrorist and w_attr["ct"][w] == 1:
                max_price = w_attr["price"][w]
                max_price_weapon = w
    if max_price_weapon is not None:
        money -= max_price
        purchase_list.append(max_price_weapon)
    
    # Buy Grenades (max 4)
    cnt = len(grenades_start)
    for g in grenades:
        if cnt == 4:
            break
        if g not in grenades_start and w_attr["price"][g] <= money:
            if is_terrorist and w_attr["t"][g] == 1:
                money -= w_attr["price"][g]
                purchase_list.append(g)
                cnt += 1
            if not is_terrorist and w_attr["ct"][g] == 1:
                money -= w_attr["price"][g]
                purchase_list.append(g)
                cnt += 1
        # 2 Flashbang
        if cnt < 4 and g == weapon_index["Flashbang"] and grenades_start.count(weapon_index["Flashbang"]) == 1:
            money -= w_attr["price"][g]
            purchase_list.append(g)
            cnt += 1

    # Buy Gear
    if weapon_index["vest"] not in weapons:
        if money >= w_attr["price"][weapon_index["vesthelm"]]:
            money -= w_attr["price"][weapon_index["vesthelm"]]
            purchase_list.append(weapon_index["vesthelm"])
        elif money >= weapon_index["vest"]:
            money -= weapon_index["vest"]
            purchase_list.append(weapon_index["vest"])

    if not is_terrorist and weapon_index["defuser"] not in weapons and money >= w_attr["price"][weapon_index["defuser"]]:
        money -= w_attr["price"][weapon_index["defuser"]]
        purchase_list.append(weapon_index["defuser"])

    if weapon_index["Zeus x27"] not in weapons and money >= w_attr["price"][weapon_index["Zeus x27"]]:
        money -= w_attr["price"][weapon_index["Zeus x27"]]
        purchase_list.append(weapon_index["Zeus x27"])

    return purchase_list
    

def process_match_data(match_data, weapon_attribute, grenades, weapon_index, weapon_type):
    f1_accum = 0.0
    eco_diff_accum = 0.0
    acc_gun_accum = 0.0
    acc_grenade_accum = 0.0
    acc_equip_accum = 0.0
    round_accum = 0

    for player_data in match_data:
        for i, round_data in enumerate(player_data):
            if i == 0 or i == 15:
                continue
            data, label = round_data[0], round_data[1]

            is_terrorist = True if data[0][0] == 0 else False
            weapons = data[1]
            money = data[2][0] * 1000

            # print(is_terrorist, weapons, money)       
            purchase = greedy(is_terrorist, weapons, money, weapon_attribute, grenades, weapon_index)

            f1 = get_accuracy(purchase, label)
            acc_type = get_acc_type(purchase, label, weapon_type)
            eco_diff = get_finance_diff(purchase, label, money, weapon_attribute["price"])
            f1_accum += f1
            eco_diff_accum += eco_diff
            acc_gun_accum += acc_type[0]
            acc_grenade_accum += acc_type[1]
            acc_equip_accum += acc_type[2]

            round_accum += 1

    return f1_accum, eco_diff_accum, acc_gun_accum, acc_grenade_accum, acc_equip_accum, round_accum

def main():
    _, _, test_set = read_dataset(DATA_DIR)

    weapon_price = np.load("./data/action_money.npy", allow_pickle=True)
    weapon_type = np.load("./data/action_type.npy", allow_pickle=True)
    mask = np.load("./data/mask.npz", allow_pickle=True)
    ct_mask = mask["ct_mask"]
    t_mask = mask["t_mask"]

    weapon_attribute = { "price": weapon_price, "ct": ct_mask, 
                        "t": t_mask, "type": weapon_type }

    with open("./data/weapon_index.json") as f:
        weapon_index = json.load(f)
    
    grenades = []
    for w in range(44):
        if weapon_type[w] == 6:
            grenades.append(w)
    grenades.sort(key=lambda x: weapon_price[x], reverse=True)
    # print(grenades)

    f1_accum = 0.0
    eco_diff_accum = 0.0
    acc_gun_accum = 0.0
    acc_grenade_accum = 0.0
    acc_equip_accum = 0.0

    round_accum = 0
    for data in tqdm(test_set):
        f1, eco_diff, acc_gun, acc_grenade, acc_equip, round = process_match_data(data, weapon_attribute, grenades, weapon_index, weapon_type)

        f1_accum += f1
        eco_diff_accum += eco_diff
        acc_gun_accum += acc_gun
        acc_grenade_accum += acc_grenade
        acc_equip_accum += acc_equip

        round_accum += round

    # print(f1_accum, round_accum)
    print("test_set avg f1 score: ", f1_accum / round_accum)
    print("test_set acc gun: ", acc_gun_accum / round_accum)
    print("test_set acc grenade: ", acc_grenade_accum / round_accum)
    print("test_set acc equip: ", acc_equip_accum / round_accum)
    print("test_set eco diff: ", eco_diff_accum / round_accum)

if __name__ == "__main__":
    main()
