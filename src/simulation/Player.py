class Player():
    def __init__(self, mlb_id, rotowire_id):
        self.mlb_id = mlb_id
        self.rotowire_id = rotowire_id
        self.name = self.get_name()
        self.team = self.get_team()

    # if player defined using rotowire, get mapping to mlb id, etc.
    def get_remaining_ids(self):
        pass
    
    def get_name(self):
        pass

    def get_team(self):
        pass