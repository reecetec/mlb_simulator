import Player

class Pitcher(Player):
    def __init__(self, mlb_id, rotowire_id, pitches_thrown):
        super().__init__(mlb_id, rotowire_id)
        self.pitches_thrown = pitches_thrown


    def fit_pitch_generator(self):
        pass

    def fit_pitch_sequencer(self):
        pass

    def generate_pitch_characteristics(self):
        pass

    def generate_pitch_type(self):
        pass

    def generate_pitch(self):
        pass
