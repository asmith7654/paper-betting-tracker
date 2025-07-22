"""
results.py

The file fetches sports game results using both The-Odds-API and TheSportsDB API. Results are appended to 
DataFrames containing paper bets in a column called "Result".

Author: Andrew Smith
Date: July 2025
"""
# ---------------------------------------- Imports and Variables --------------------------------------------
import pandas as pd
from theodds_results import get_finished_games, map_league_to_key
from sportsdb_results import get_finished_games_from_thesportsdb

sports_with_results = [
    "americanfootball_cfl",
    "americanfootball_ncaaf",
    "americanfootball_nfl",
    "americanfootball_nfl_preseason",
    "americanfootball_ufl",
    "aussierules_afl",
    "baseball_mlb",
    "basketball_euroleague",
    "basketball_nba",
    "basketball_nba_preseason",
    "basketball_wnba",
    "basketball_ncaab",
    "icehockey_nhl",
    "rugbyleague_nrl",
    "soccer_argentina_primera_division",
    "soccer_australia_aleague",
    "soccer_austria_bundesliga",
    "soccer_belgium_first_div",
    "soccer_brazil_campeonato",
    "soccer_brazil_serie_b",
    "soccer_chile_campeonato",
    "soccer_china_superleague",
    "soccer_denmark_superliga",
    "soccer_efl_champ",
    "soccer_england_efl_cup",
    "soccer_england_league1",
    "soccer_england_league2",
    "soccer_epl",
    "soccer_fa_cup",
    "soccer_fifa_world_cup",
    "soccer_fifa_world_cup_womens",
    "soccer_fifa_club_world_cup",
    "soccer_finland_veikkausliiga",
    "soccer_france_ligue_one",
    "soccer_france_ligue_two",
    "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2",
    "soccer_germany_liga3",
    "soccer_greece_super_league",
    "soccer_italy_serie_a",
    "soccer_italy_serie_b",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    "soccer_league_of_ireland",
    "soccer_mexico_ligamx",
    "soccer_netherlands_eredivisie",
    "soccer_norway_eliteserien",
    "soccer_poland_ekstraklasa",
    "soccer_portugal_primeira_liga",
    "soccer_spain_la_liga",
    "soccer_spain_segunda_division",
    "soccer_spl",
    "soccer_sweden_allsvenskan",
    "soccer_sweden_superettan",
    "soccer_switzerland_superleague",
    "soccer_turkey_super_league",
    "soccer_uefa_europa_conference_league",
    "soccer_uefa_champs_league",
    "soccer_uefa_champs_league_qualification",
    "soccer_uefa_europa_league",
    "soccer_uefa_european_championship",
    "soccer_uefa_euro_qualification",
    "soccer_uefa_nations_league",
    "soccer_conmebol_copa_america",
    "soccer_conmebol_copa_libertadores",
    "soccer_usa_mls"
]


# ------------------------------------------- Main Pipeline -----------------------------------------------
if __name__ == "__main__":
    # Create file paths
    bet_str = "master_avg_bets"
    input_csv = f"{bet_str}.csv"
    output_csv = f"{bet_str}_results.csv"

    # Pull results from the odd api
    df = pd.read_csv(input_csv)
    keys = map_league_to_key(df)

    # Loop through keys
    for key in keys:
        df = get_finished_games(df, key)

    print("\nCompleted The-Odds-API pull, now pulling from TheSportsDB.\n")

    # Pull remaining results from sportsdb
    df = get_finished_games_from_thesportsdb(df)


    # Write to .csv
    df.to_csv(output_csv,index=False)