"""
fetch_odds.py

The file fetches sports betting odds using The-Odds-API. Pulls in odds from any designated sport,
region, or market. Organizes a DataFrame with odds to contain one row per outcome, with columns being
bookmakers and essential information.

Author: Andrew Smith
Date: July 2025
"""
# ---------------------------------------- Imports and Variables ------------------------------------------ #
import requests
import pandas as pd
from datetime import datetime
from dateutil import parser
import pytz

# Sports keys
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

API_KEY = "7ca177e18aa6a5230dddc27a238e3f73"
SPORT = upcoming
REGIONS = "us,uk,eu,au"
MARKETS = "h2h"
ODDS_FORMAT = "decimal"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# -------------------------- Odds fetcher (Rows are one set of odds for one outcome) ---------------------- #
def fetch_odds() -> pd.DataFrame:
    """
    Fetches head-to-head (h2h) betting odds from The Odds API.

    Returns:
        pd.DataFrame: A DataFrame containing bookmaker odds data for each outcome in each game.
                      Each row represents one team's odds from one bookmaker.
    """
    # Construct API URL and parameters
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT
    }

    # Send request to the API
    response = requests.get(url, params=params)

    # Log API usage details for debugging/rate limiting
    print("Requests Remaining:", response.headers.get("x-requests-remaining"))
    print("Requests Used:", response.headers.get("x-requests-used"))

    # Return empty DataFrame if request fails
    if response.status_code != 200:
        print(f"Failed to fetch odds: {response.status_code} - {response.text}")
        return pd.DataFrame()

    # Parse the JSON response
    data = response.json()
    print(f"Pulled {len(data)} games")

    # List of betting exchanges to exclude from results
    exchange_blocklist = {
        "Smarkets",
        "Betfair",
        "Matchbook",
        "Betfair Sportsbook"
    }

    rows = []  # Will contain each row of the resulting DataFrame
    eastern = pytz.timezone("US/Eastern")  # Used to localize times

    for game in data:
        try:
            # Get home and away teams
            home = game["home_team"]
            away = game["away_team"]

            # Get league for results fetching
            league = game["sport_title"]

            # Convert game start time from UTC to Eastern time
            utc_dt = datetime.fromisoformat(game["commence_time"][:-1]).replace(tzinfo=pytz.utc)
            local_dt = utc_dt.astimezone(eastern)
            start_time = local_dt.strftime(DATE_FORMAT)

            for book in game.get("bookmakers", []):
                book_name = book["title"]

                # Skip betting exchanges
                if any(exchange.lower() in book_name.lower() for exchange in exchange_blocklist):
                    continue

                # Convert last update time to Eastern
                last_update_utc = parser.isoparse(book["last_update"])
                last_update_local = last_update_utc.astimezone(eastern)
                last_update_str = last_update_local.strftime(DATE_FORMAT)

                # Only process "h2h" markets for now
                for market in book.get("markets", []):
                    if market["key"] != "h2h":
                        print("Non-h2h market found, refactor code")
                        continue

                    for outcome in market.get("outcomes", []):
                        # Append a row with all necessary data
                        rows.append({
                            "match": f"{away} @ {home}",
                            "league": league,
                            "start time": start_time,
                            "team": outcome["name"],
                            "bookmaker": book_name,
                            "odds": outcome["price"],
                            "last update": last_update_str
                        })

        except Exception as e:
            # Catch and report errors during game parsing
            print(f"Error parsing game: {e}")
            continue

    # Convert collected rows into a pandas DataFrame
    df = pd.DataFrame(rows)
    return df


# ------------------------- Organizer (Rows are outcomes with all available odds) ------------------------- #
def organize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Organizes dataframes from fetch_odds() such that each row is an outcome, and columns are bookmakers
    and essential info.

    Args:
        df (pd.DataFrame): A dataframe containing odds from The Odds API.
    
    Returns:
        pd.DataFrame: A dataframe where each row is an outcome, and columns are bookmakers
        and essential info.
    """
    # Store all rows in a list, avoiding repeated DataFrame concatenation
    rows = []

    # Get all unique matches and bookmakers from the original dataframe
    unique_matches = df["match"].unique().tolist()
    bookmakers = df["bookmaker"].unique().tolist()

    # Define the column structure for the organized dataframe
    columns = ["match", "league", "start time", "team", "last update"] + bookmakers + ["best odds", "best bookmaker"]
    new_df = pd.DataFrame(columns=columns)

    # Iterate through each unique match
    for match in unique_matches:
        # Filter rows that correspond to the current match
        matching_rows = df[df["match"] == match]

        # Extract static info (assumed same across rows for a given match)
        league = matching_rows.iloc[0]["league"]
        start_time = matching_rows.iloc[0]["start time"]
        last_update = matching_rows.iloc[0]["last update"]

        # Get the unique teams or outcomes (e.g., Team A win, draw, Team B win)
        teams = matching_rows["team"].unique().tolist()

        # Process each outcome (team) for the current match
        for team in teams:
            # Get all rows for this team in this match
            team_rows = matching_rows[matching_rows["team"] == team]

            # Initialize a dictionary to hold odds for each bookmaker
            odds_row = {bm: None for bm in bookmakers}

            # Variables to track the best odds and the corresponding bookmaker
            best = None
            best_bm = None

            # For each bookmaker, find and record their odds for this outcome
            for bm in bookmakers:
                # Filter for this specific bookmaker's row
                bm_row = team_rows[team_rows["bookmaker"] == bm]

                # If data exists, extract the odds
                if not bm_row.empty:
                    odds = bm_row.iloc[0]["odds"]
                    odds_row[bm] = odds

                    # Update best odds if this is the highest seen so far
                    if best is None or odds > best:
                        best = odds
                        best_bm = bm

            # Combine all data into a single row for the new DataFrame
            row = {
                "match": match,
                "league": league,
                "start time": start_time,
                "team": team,
                "last update": last_update,
                **odds_row,  # Unpack bookmaker odds into the row
                "best odds": best,
                "best bookmaker": best_bm
            }
            rows.append(row)

    # Create a single DataFrame from the collected rows
    new_df = pd.DataFrame(rows, columns=columns)
    return new_df


# ------------------------------------------ Main Pipeline ------------------------------------------ #
if __name__ == "__main__":
    df_odds = fetch_odds()
    organized_df = organize(df_odds)
    organized_df.to_csv("odds.csv", index=False)
