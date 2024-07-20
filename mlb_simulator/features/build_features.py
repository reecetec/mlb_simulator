from mlb_simulator.data.data_utils import query_mlb_db
import pandas as pd

# pitch characteristics to be generated/put into pitch outcome model
_pitch_characteristics = [
    "release_speed",
    "release_spin_rate",
    "plate_x",
    "plate_z",
]
_pitch_outcome_game_state_features = "p_throws strikes balls".split()

PITCH_OUTCOME_FEATURES = _pitch_characteristics + _pitch_outcome_game_state_features
PITCH_OUTCOME_TARGET_COL = "pitch_outcome"


def format_backtest_date(backtest_date):
    """
    format the user input backtest date to sqlite query format
    """

    if backtest_date is not None:
        backtest_date = f"and game_date <= date('{backtest_date}')"
    else:
        backtest_date = ""
    return backtest_date


def get_game_state_t_prob_data():

    query_str = f"""
    select game_pk, inning, inning_topbot, at_bat_number, pitch_number,
        outs_when_up, post_bat_score - bat_score as runs_scored, stand,
    CASE
        WHEN events IN ('single') THEN 'single'
        WHEN events IN ('double') THEN 'double'
        WHEN events IN ('triple') THEN 'triple'
        WHEN events IN ('home_run') THEN 'home_run'
        WHEN events IN ('field_out') THEN 'field_out'
        WHEN events IN ('ground_out', 'force_out') THEN 'ground_out'
        WHEN events IN ('fly_out', 'sac_fly') THEN 'fly_out'
        WHEN events IN ('double_play', 'grounded_into_double_play',
                        'sac_fly_double_play') THEN 'double_play'
        WHEN events IN ('triple_play') THEN 'triple_play'
        WHEN events IN ('field_error') THEN 'fielding_error'
        WHEN events IN ('fielders_choice') THEN 'fielders_choice'
        ELSE NULL
    END AS simplified_outcome,
    batter, on_1b, on_2b, on_3b
    from Statcast 
    where game_year > 2020
    and inning < 9 /* avoid OT innings, handle manually */
    order by game_date, game_pk, inning, at_bat_number, pitch_number
    ;
    """

    dataset = query_mlb_db(query_str)

    return dataset


def get_hit_classification_data(venue_name, backtest_date=None):

    backtest_date = format_backtest_date(backtest_date)

    # get all game pks for this venue
    venue_query = f"""
    select game_pk from VenueGamePkMapping
    where venue_name = '{venue_name}'
    """
    game_pks = query_mlb_db(venue_query)["game_pk"].astype(str)
    sql_game_pks = ", ".join(game_pks)

    statcast_query = f"""
    SELECT
        CASE
            WHEN events IN ('single') THEN 'single'
            WHEN events IN ('double') THEN 'double'
            WHEN events IN ('triple') THEN 'triple'
            WHEN events IN ('home_run') THEN 'home_run'
            WHEN events IN ('field_out') THEN 'field_out'
            WHEN events IN ('ground_out', 'force_out') THEN 'ground_out'
            WHEN events IN ('fly_out', 'sac_fly') THEN 'fly_out'
            WHEN events IN ('double_play', 'grounded_into_double_play',
                            'sac_fly_double_play') THEN 'double_play'
            WHEN events IN ('triple_play') THEN 'triple_play'
            WHEN events IN ('field_error') THEN 'fielding_error'
            WHEN events IN ('fielders_choice') THEN 'fielders_choice'
            ELSE NULL
        END AS simplified_outcome,
        game_pk, batter,
        case when inning_topbot='Top' then home_team else away_team
                      end as 'fielding_team',
        game_year, outs_when_up, stand, 
        case when on_1b is not null then 1 else 0 end as on_1b,
        case when on_2b is not null then 1 else 0 end as on_2b,
        case when on_3b is not null then 1 else 0 end as on_3b,    
        launch_speed, launch_angle,
          ROUND((-(180 / PI()) * atan2(hc_x - 130, 213 - hc_y) + 90))
                      as spray_angle
    FROM
        Statcast
    WHERE game_pk in ({sql_game_pks})
    and game_year > 2020
    and type='X'
    and simplified_outcome &
    game_year & outs_when_up & of_fielding_alignment &
    launch_speed & launch_angle & spray_angle is not null
    {backtest_date}
    ORDER BY GAME_DATE ASC;
    """
    target_col = "simplified_outcome"

    df = query_mlb_db(statcast_query)

    speed_df = query_mlb_db("select mlb_id as batter, speed from PlayerSpeed;")
    oaa_df = query_mlb_db(
        """
        select o.year as 'game_year', t.statcast_name as 'fielding_team',
                          o.oaa_rhh_standardized, o.oaa_lhh_standardized
        from TeamOAA o
        left join TeamIdMapping t on o.entity_id = t.team_id
    """
    )
    oaa_df["game_year"] = oaa_df["game_year"].astype(int)

    df = df.merge(speed_df, how="left", on="batter")
    df = df.merge(oaa_df, how="left", on=["game_year", "fielding_team"])

    df["speed"] = df["speed"].astype(float)
    df["speed"] = df["speed"].fillna(df["speed"].mean())

    df["oaa"] = df.apply(
        lambda row: (
            row["oaa_rhh_standardized"]
            if row["stand"] == "R"
            else row["oaa_lhh_standardized"]
        ),
        axis=1,
    )

    df = df.drop(
        [
            "batter",
            "game_pk",
            "oaa_rhh_standardized",
            "oaa_lhh_standardized",
            "fielding_team",
        ],
        axis=1,
    )

    return df, target_col


def get_pitch_sequencing_data(pitcher, backtest_date=None):
    """ """

    # format input backtest date as query
    backtest_date = format_backtest_date(backtest_date)

    # only care about RECENT pitch arsenal - what has this pitcher thrown in
    # last 1000 pithhes?
    pitch_arsenal_query = f"""
        select distinct pitch_type from (
        select pitch_type 
        from Statcast 
        where pitcher = "{pitcher}"
        and pitch_type <> 'PO'
        and pitch_type is not null
        and game_type <> 'E' || 'S' /* avoid games with likely 
                                        experimentation */
        order by game_date desc limit 1000
        )
    """
    pitch_arsenal = query_mlb_db(pitch_arsenal_query)["pitch_type"]
    sql_pitch_arsenal = ", ".join(pitch_arsenal)

    query_str = f"""
        SELECT game_year, pitch_type, batter, pitch_number, strikes, balls,
        outs_when_up, stand,
            CASE
                when on_1b is not null then 1
                else 0
            END AS on_1b,
            CASE
                when on_2b is not null then 1
                else 0
            END AS on_2b,
            CASE
                when on_3b is not null then 1
                else 0
            END AS on_3b,
        LAG(pitch_type) OVER (PARTITION BY game_pk, pitcher,
                              at_bat_number ORDER BY pitch_number)
        AS prev_pitch,
        ROW_NUMBER() OVER (PARTITION BY game_pk, pitcher
                           ORDER BY at_bat_number, pitch_number)
        AS cumulative_pitch_number
        FROM Statcast
        WHERE pitcher = {pitcher}
            and pitch_type in ('{("', '").join(pitch_arsenal)}')
            AND pitch_type IS NOT NULL
        {backtest_date}
        ORDER BY game_year, at_bat_number, pitch_number
    """
    target_col = "pitch_type"
    pitcher_df = query_mlb_db(query_str).set_index("batter")

    # get datasets for batters standardized woba and strike pct into
    # pitchers arsenal
    batter_query = lambda table: f"""select batter, {sql_pitch_arsenal} from {table}"""
    strike_df = (
        query_mlb_db(batter_query("BatterStrikePctByPitchType"))
        .set_index("batter")
        .add_suffix("_strike")
    )
    woba_df = (
        query_mlb_db(batter_query("BatterAvgWobaByPitchType"))
        .set_index("batter")
        .add_suffix("_woba")
    )

    df = pitcher_df.merge(strike_df, left_index=True, right_index=True, how="left")
    df = df.merge(woba_df, left_index=True, right_index=True, how="left")
    df.reset_index(drop=True, inplace=True)

    return df, target_col, pitch_arsenal


def get_hit_outcome_data(batter_id: int | None, backtest_date=None) -> pd.DataFrame:
    """ """

    # format input backtest date as query
    backtest_date = format_backtest_date(backtest_date)

    query_str = f"""
        select 
            launch_speed, launch_angle,
            ROUND((-(180 / PI()) * atan2(hc_x - 130, 213 - hc_y) + 90))
            as spray_angle
        from Statcast
        {f"where batter={batter_id}" 
         if batter_id is not None else 
         "where stand='R' and sz_top >= 3.3 and sz_top <= 3.5 and sz_bot <= 1.7 and sz_bot >= 1.5"}
        and description = 'hit_into_play'
        and
            launch_speed &
            launch_angle &
            spray_angle    
        is not null
        {backtest_date}
        order by game_date asc, at_bat_number asc, pitch_number asc;
    """

    data = query_mlb_db(query_str)

    return data


def get_pitch_outcome_data(
    batter_id: int | None, backtest_date=None
) -> tuple[pd.DataFrame, str]:
    """Function to get training data for a batter's pitch outcome model

    For a given player, returns the model features for the pitch outcome model
    (PITCH_OUTCOME_FEATURES) and the pitch_outcome (the target for this model).
    If a backtest date is supplied, will only return data before the backtest
    date.

    Args:
        batter_id (int): the batter's mlb id
        backtest_date (str): get data up to backtest_date (default None)

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: a dataframe containing the query results
            - str: the name of the target col for this model

    Example:
        >>> df = get_pitch_outcome_data(665742, backtest_date='2022-01-01')
        >>> df.head()
    """

    # format input backtest date as query
    backtest_date = format_backtest_date(backtest_date)

    query_str = f"""
        select 
            case
                when description='swinging_strike' or
                     description='swinging_strike_blocked' or
                     description='called_strike' or description='foul_tip' or
                     description='swinging_pitchout'
                then 'strike'
                when description='foul' or
                     description='foul_pitchout' 
                then 'foul'
                when description='ball' or
                     description='blocked_ball' or
                     description='pitchout' 
                then 'ball'
                when description='hit_by_pitch' then 'hit_by_pitch'
                when description='hit_into_play' then 'hit_into_play'
                else NULL
            end as {PITCH_OUTCOME_TARGET_COL},
            {', '.join(PITCH_OUTCOME_FEATURES)}
        from Statcast
        {f"where batter={batter_id}" 
         if batter_id is not None else 
         "where stand='R' and sz_top >= 3.3 and sz_top <= 3.5 and sz_bot <= 1.7 and sz_bot >= 1.5"}
        and {' & '.join(PITCH_OUTCOME_FEATURES)} 
        is not null
        {backtest_date}
        order by game_date asc, at_bat_number asc, pitch_number asc;
    """

    df = query_mlb_db(query_str)

    return df, PITCH_OUTCOME_TARGET_COL


if __name__ == "__main__":

    kukuchi = 579328
    jones = 683003

    df, target_col = get_pitch_outcome_data(665742, backtest_date="2022-01-01")
    print(target_col)
    print(df.head())
