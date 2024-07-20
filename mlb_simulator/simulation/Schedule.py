"""
The schedule object will scrape daily projected lineups from rotowire. Creates
a list of Game objects.
"""

from mlb_simulator.simulation.Game import Game

import requests
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import statsapi

logger = logging.getLogger(__name__)
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)


class Schedule:

    ROTOWIRE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"
    ROTOWIRE_TOMORROW_SUFFIX = "?date=tomorrow"

    def __init__(self, for_today=True):
        self.for_today = for_today

        # rotowire has the schedule for the current day and the next day only
        if for_today:
            self.sched_date = datetime.now().date()
        else:
            self.sched_date = datetime.now().date() + timedelta(days=1)

        self._get()

    def __repr__(self):
        return f"Schedule(for_today={self.for_today})"

    def __len__(self):
        return len(self._games)

    def __getitem__(self, position):
        return self._games[position]

    def _get_venues(self):
        api_games = statsapi.schedule(date=self.sched_date.strftime("%Y-%m-%d"))

        # convert to map
        home_id_venue_map = {}
        for game_json in api_games:
            home_id_venue_map[game_json["home_id"]] = (
                game_json["venue_id"],
                game_json["venue_name"],
            )

        for game in self._games:
            venue_id, name = home_id_venue_map[game.home_team.team_id]
            setattr(game, "venue_id", venue_id)
            setattr(game, "venue_name", name)

    def _get(self):
        """
        Get the games for today. If for_today=False, get tomorrow's games.
        Create a Game object for each element in the schedule.
        """

        if not self.for_today:
            res = requests.get(
                Schedule.ROTOWIRE_URL + Schedule.ROTOWIRE_TOMORROW_SUFFIX
            )
        else:
            res = requests.get(Schedule.ROTOWIRE_URL)

        if res.status_code == 200:
            game_times, game_soups = self._process_res(res)
            self._games = [
                Game(game_times[i], game_soups[i]) for i in range(len(game_times))
            ]

            if len(self._games) == 0:
                logger.info(f"No games on {self.sched_date}")
                exit()

            self._get_venues()

        else:
            logger.critical("Failed to get schedule from Rotowire")
            exit()

    def _process_res(self, res):
        """
        Process the raw request to extract a list of game times and
        game lineups.
        """

        html_content = res.text
        soup = BeautifulSoup(html_content, "html.parser")
        lineup_boxes = soup.find_all("div", class_="lineup__box")
        game_times = soup.find_all("div", class_="lineup__time")

        def format_dt(raw_time):
            time_text = raw_time.get_text(strip=True)
            time = datetime.strptime(time_text, "%I:%M %p ET").time()
            final_datetime = datetime.combine(self.sched_date, time)
            return final_datetime

        # convert game times to dt
        # last 2 elements are not actual lineup boxes
        dt_times = [format_dt(raw_time) for raw_time in game_times[:-2]]

        return dt_times, lineup_boxes[:-2]


if __name__ == "__main__":
    s = Schedule(for_today=False)
    for game in s:
        print(game)
    # s[9].fit()
