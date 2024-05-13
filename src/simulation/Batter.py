import Player

class Batter(Player):
    def __init__(self, mlb_id, rotowire_id, batting_avg):
        super().__init__(mlb_id, rotowire_id)
        self.batting_avg = batting_avg