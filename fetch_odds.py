"""
fetch_odds.py

The file fetches sports betting odds using The Odds API. Pulls in odds from any designated sport,
region, or market. Organizes a dataframe with odds to contain one row per outcome, with columns being
bookmakers and essential information.

Author: Andrew Smith
Date: July 2025
"""

import requests
import pandas as pd
from datetime import datetime
from dateutil import parser
import pytz

#sports keys
upcoming = "upcoming"
kbo = "baseball_kbo"
mlb = "baseball_mlb"
ncaa_baseball = "baseball_ncaa"
wnba = "basketball_wnba"
brazil_serie_a = "soccer_brazil_campeonato"
brazil_serie_b = "soccer_brazil_serie_b"
super_league_china = "soccer_china_superleague"
japan_league = "soccer_japan_j_league"
mls = "soccer_usa_mls"
cfl = "americanfootball_cfl"
aussie = "aussierules_afl"
npb = "baseball_npb"
boxing = "boxing_boxing"
cricket = "cricket_t20_blast"
lacrosse = "lacrosse_pll"
rugby = "rugbyleague_nrl"
mma = "mma_mixed_martial_arts"
euroleague = "soccer_uefa_european_championship"
finland = "soccer_finland_veikkausliiga"
nhl = "icehockey_nhl"
sweden_hockey = "icehockey_sweden_hockey_league"
mexico = "soccer_mexico_ligamx"
ireland = "soccer_league_of_ireland"

API_KEY = "bd61470183edaddac927f82180a28fa1"
SPORT = upcoming
REGIONS = "us,uk,eu,au"
MARKETS = "h2h"
ODDS_FORMAT = "decimal"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def fetch_odds() -> pd.DataFrame:
    """
    Goes to The Odds API and pulls odds from the designated constants above. Returns a dataframe with odds
    information.

    Returns:
        pd.Dataframe: A dataframe from The Odds API containing odds information.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"⚠️ Failed to fetch odds: {response.status_code} - {response.text}")
        return pd.DataFrame()

    data = response.json()
    print(f"✅ Pulled {len(data)} games.\n")

    # 🧱 List of known betting exchanges to skip
    exchange_blocklist = {
        "Smarkets",
        "Betfair",
        "Matchbook",
        "Betfair Sportsbook"
    }

    rows = []
    eastern = pytz.timezone("US/Eastern")

    for game in data:
        try:
            home = game["home_team"]
            away = game["away_team"]

            utc_dt = datetime.fromisoformat(game["commence_time"][:-1]).replace(tzinfo=pytz.utc)
            local_dt = utc_dt.astimezone(eastern)
            start_time = local_dt.strftime(DATE_FORMAT)

            for book in game.get("bookmakers", []):
                book_name = book["title"]

                # ⛔ Skip betting exchanges
                if any(exchange.lower() in book_name.lower() for exchange in exchange_blocklist):
                    continue

                last_update_utc = parser.isoparse(book["last_update"])
                last_update_local = last_update_utc.astimezone(eastern)
                last_update_str = last_update_local.strftime(DATE_FORMAT)

                for market in book.get("markets", []):
                    if market["key"] != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        rows.append({
                            "match": f"{away} @ {home}",
                            "start time": start_time,
                            "team": outcome["name"],
                            "bookmaker": book_name,
                            "odds": outcome["price"],
                            "last update": last_update_str
                        })
        except Exception as e:
            print(f"Error parsing game: {e}")
            continue

    df = pd.DataFrame(rows)
    return df


def organize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Organizes dataframes from The Odds API such that each row is an outcome, and columns are bookmakers
    and essential info.

    Args:
        df (pd.Dataframe): A dataframe containing odds from The Odds API.
    
    Returns:
        pd.Dataframe: A dataframe where each row is an outcome, and columns are bookmakers
    and essential info.
    """
    unique_matches = df["match"].unique().tolist()
    bookmakers = df["bookmaker"].unique().tolist()
    columns = ["match", "start time", "team", "last update"] + bookmakers + ["best odds"]
    new_df = pd.DataFrame(columns=columns)

    for match in unique_matches:
        matching_rows = df[df["match"] == match]
        start_time = matching_rows.iloc[0]["start time"]
        last_update = matching_rows.iloc[0]["last update"]
        teams = matching_rows["team"].unique().tolist()  # now includes Draw if present

        for team in teams:
            team_rows = matching_rows[matching_rows["team"] == team]
            odds_row = {bm: None for bm in bookmakers}

            for bm in bookmakers:
                bm_rows = team_rows[team_rows["bookmaker"] == bm]
                if not bm_rows.empty:
                    odds_row[bm] = bm_rows.iloc[0]["odds"]

            odds_values = [v for v in odds_row.values() if v is not None]
            best = max(odds_values) if odds_values else None

            row = {
                "match": match,
                "start time": start_time,
                "team": team,
                "last update": last_update,
                **odds_row,
                "best odds": best
            }

            new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)

    return new_df


if __name__ == "__main__":
    df_odds = fetch_odds()
    organized_df = organize(df_odds)
    print(organized_df.head())
    organized_df.to_csv("odds.csv", index=False)
