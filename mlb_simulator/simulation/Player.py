"""
A super class for batters/pitchers. Given an MLB id or a Rotowire id, gathers
the remainder of the information from appropriate name maps. E.g., given an 
MLB id, find the rotowire id, name of the player, position, etc.
"""

import pandas as pd
import pathlib
import os

class Player:

    def __init__(self, mlb_id=None, rotowire_id=None, backtest_date=''):
        self.mlb_id = mlb_id
        self.rotowire_id = rotowire_id
        self.backtest_date = backtest_date
        self.get_player_info()

    # if player defined using rotowire, get mapping to mlb id, vice versa
    def get_player_info(self):
        """
        Obtains the missing id for the player
        """

        name_mapping_df = pd.read_csv(os.path.join(pathlib.Path.home(),
                                                   'sports',
                                                   'mlb_simulator',
                                                   'data',
                                                   'raw',
                                                   'name_map.csv'))
        if self.mlb_id:
            name_mapping_df.set_index('MLBID', inplace=True)
            player = name_mapping_df.loc[[self.mlb_id]]
            self.rotowire_id = int(player['ROTOWIREID'].iloc[0])
            
        else:
            name_mapping_df.set_index('ROTOWIREID', inplace=True)
            player = name_mapping_df.loc[[self.rotowire_id]]
            self.mlb_id = int(player['MLBID'].iloc[0])

        # some players have multiple entries (just take first)
        if len(player) > 0:
            self.name = player['PLAYERNAME'].iloc[0]
            self.team = player['TEAM'].iloc[0]
            self.pos = player['POS'].iloc[0]

        else:
            self.name = player['PLAYERNAME']
            self.team = player['TEAM']
            self.pos = player['POS']


    def print_info(self):
        print(f'{self.name}, {self.pos} (mlbid: {self.mlb_id}, rotowireid: {
            self.rotowire_id})')


if __name__ == '__main__':
    players = [Player(mlb_id=665742),
               Player(rotowire_id=18749),
               Player(mlb_id=683003)]

    for player in players:
        player.print_info()

