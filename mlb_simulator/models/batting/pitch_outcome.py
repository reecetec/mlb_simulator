""" 
this model determines the pitch outcome, meaning was the ball hit,
strike, ball, foul, or hit into play?
"""
from mlb_simulator.data.data_utils import query_mlb_db
from mlb_simulator.models import model_utils as mu

pitch_outcome_query = lambda batter: f"""
{batter}
"""

print(pitch_outcome_query(123))
