"""
sportsdb_results.py

The file fetches sports game results using TheSportsDB API. Results are appended to DataFrames containing paper
bets in a column called "Result".

Author: Andrew Smith
Date: July 2025
"""
# ---------------------------------------- Imports and Variables --------------------------------------------
import requests
import pandas as pd
import time

THESPORTSDB_API_KEY = "123"


# ----------------------------------------- Formatting Helpers ----------------------------------------------
def _start_date(ts) -> str:
    """
    Convert a timestamp / datetime-like / ISO string to "YYYY-MM-DD".

    Args:
        ts (Any): Timestamp to convert to date.

    Returns:
        str: Date string.
    """
    return pd.to_datetime(ts).strftime("%Y-%m-%d")


def _format_match_for_thesportsdb(match: str) -> str:
    """
    Convert a match string (Lakers @ Celtics) to a correctly formatted match string (Lakers_vs_Celtics).

    Args:
        match (str): The name of the match.

    Returns:
        str: A list containing each individual team.
    """
    if "@" in match:
        teams = [t.strip() for t in match.split("@")]
        formatted = f"{teams[1]} vs {teams[0]}"
    elif "vs" in match.lower():
        formatted = match
    else:
        return match.replace(" ", "_")
    return formatted.replace(" ", "_")


# ------------------------------------------- Results Fetcher -----------------------------------------------
def _get_results(match: str, date: str) -> str:
    """
    Find the results of a game.

    Args:
        match (str): The name of the match.
        date (str): The date of the match.

    Returns:
        str: The outcome of the game.
    """
    url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/searchevents.php?e={match}&d={date}"
    try:
        # Request url
        resp = requests.get(url)
        if resp.status_code != 200:
            return "API Error"
        
        # Store data and check if it does not exist
        data = resp.json()
        if not data or "event" not in data or not data["event"]:
            return "Not Found"

        # Store event
        event = data["event"][0]

        # Store teams
        home = event.get("strHomeTeam", "Home")
        away = event.get("strAwayTeam", "Away")

        # Store scores
        home_score = event.get("intHomeScore")
        away_score = event.get("intAwayScore")

        print(f"{home} (H) vs {away} (A): {home_score}-{away_score}")

        if home_score is None or away_score is None:
            return "Pending"

        # Results
        home_score, away_score = int(home_score), int(away_score)
        if home_score > away_score:
            return home
        elif away_score > home_score:
            return away
        else:
            return "Draw"

    except Exception as e:
        print(f"Error fetching match: {e}")
        return "Error"


# --------------------------------------- Results Column Function -------------------------------------------
def get_finished_games_from_thesportsdb(df: pd.DataFrame) -> pd.DataFrame:
    if "Result" not in df.columns:
        df["Result"] = "Not Found"

    fetches = 0

    for idx, row in df.iterrows():
        if row["Result"] != "Not Found":
            continue

        match = _format_match_for_thesportsdb(row["Match"])
        date = _start_date(row["Start Time"])
        result = _get_results(match, date)
        fetches += 1
        df.at[idx, "Result"] = result

        if fetches % 30 == 0:
            # Every 30 requests, wait 60 seconds
            print("Pausing for 60 seconds to respect API rate limits...")
            time.sleep(60)

    return df


# ------------------------------------------- Main Pipeline -----------------------------------------------
if __name__ == "__main__":
    input_csv = "results.csv"
    df = pd.read_csv(input_csv)
    df = get_finished_games_from_thesportsdb(df)
    df.to_csv("results2.csv", index=False)
