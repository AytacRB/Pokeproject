import requests
import pandas as pd
import re
from datetime import datetime
import os
import ast


url = 'https://replay.pokemonshowdown.com/search.json?format=gen9ou'

def get_rawdata(url):
# Get latest 50 replays
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36 OPR/115.0.0.0'}


    r = requests.get(url, headers=headers)
    response_list = r.json()


    # Individual replay data
    battle_list = []
    for i in range(len(response_list)):
        b = response_list[i]['id']
        url = f'https://replay.pokemonshowdown.com/{b}.json'
        r = requests.get(url, headers=headers)
        battle = r.json()
        battle_list.append(battle)
        
    df = pd.DataFrame(battle_list)
    df['timestamp_scrape'] = datetime.now()

    if type(df.players.iloc[0]) != list:
        df['players'] = df['players'].apply(ast.literal_eval)
    

    p1_list = []
    p2_list = []
    win_list = []

    for i in range(len(df)):
        # Find teamcomps
        pattern = r"(?<=\|clearpoke)([\s\S]*?)(?=\|teampreview)"
        matches = re.findall(pattern, df['log'].iloc[i])

        # Extract teamcomps
        pattern = r"\|poke\|(p1|p2)\|([^|,]+)"
        matches = re.findall(pattern, matches[0])

        p1 = [name for player, name in matches if player == "p1"]
        p2 = [name for player, name in matches if player == "p2"]
        p1_list.append(p1)
        p2_list.append(p2)

        # Extract winner
        pattern = r"(?<=\|win\|)[^\\\n]+"
        matches = re.findall(pattern, df.log.iloc[i])
        if df.players.iloc[i][0] == matches[0]:
            win_list.append(1)
        elif df.players.iloc[i][1] == matches[0]:
            win_list.append(2)
        else:
            win_list.append(None)


    df['p1'] = p1_list
    df['p2'] = p2_list
    df['winner'] = win_list

    if type(df.p1.iloc[0]) != list:
        df['p1'] = df['p1'].apply(ast.literal_eval)

    if type(df.p2.iloc[0]) != list:
        df['p2'] = df['p2'].apply(ast.literal_eval)

    print(type(df.p1.iloc[0]), type(df.p2.iloc[0]), type(df.players.iloc[0]))

    return df
    
def save_data_to_csv(df, file_path='pokedata.csv'):
    # Append new data to existing file or create new one if it doesn't exist
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)

data = get_rawdata(url)
save_data_to_csv(data)