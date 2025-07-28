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
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

THESPORTSDB_API_KEY = "123"
# https://www.thesportsdb.com/api/v1/json/123/searchevents.php?e={match}&d={date}
# https://www.thesportsdb.com/api/v1/json/123/searchevents.php?e=Atlanta_Braves_vs_Texas_Rangers


# ----------------------------------------- Formatting Helpers ----------------------------------------------
def _start_date(ts) -> str:
    """
    Convert a timestamp / datetime-like / ISO string to "YYYY-MM-DD",
    adjusted 4 hours forward (e.g., from UTC to EDT).

    Args:
        ts (Any): Timestamp to convert to date.

    Returns:
        str: Date string.
    """
    dt = pd.to_datetime(ts) + pd.Timedelta(hours=4)
    return dt.strftime("%Y-%m-%d")



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


# -------------------------------------- Time Since Start Filter --------------------------------------------
def _time_since_start(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    Filter out games that started less than "thresh" hours ago.

    Args:
        df (pd.DataFrame): A results DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing only games that started over "thresh" hours ago.
    """
    # Get the current time
    current_time = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
    current_time_obj = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")

    # Convert "Start Time" column to datetime objects
    df['Start Time'] = pd.to_datetime(df['Start Time'], format="%Y-%m-%d %H:%M:%S")

    # Create conditions for removal
    cutoff = current_time_obj - timedelta(days=thresh)

    # Filter out games that started less than 12 hours ago (for API bug)
    mask = (df["Start Time"] <= cutoff)
    df = df[mask]

    return df


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

    # Only loop through games that started less than 12 hours ago
    indices = _time_since_start(df,0.5).index.tolist()

    # Track API requests to respect rate limits
    fetches = 0

    for i in indices:
        row = df.iloc[i]
        existing_result = row.get("Result")

        # Skip rows that already have a result other than "Not Found"
        if existing_result not in ["Not Found", "Pending", "API Error"]:
            continue

        match = _format_match_for_thesportsdb(row["Match"])
        date = _start_date(row["Start Time"])
        result = _get_results(match, date)
        fetches += 1
        df.at[i, "Result"] = result

        if fetches % 30 == 0:
            # Every 30 requests, wait 60 seconds
            print("\nPausing for 60 seconds to respect SportsDB API rate limits...\n")
            time.sleep(60)

    return df


# ------------------------------------------- Main Pipeline -----------------------------------------------
if __name__ == "__main__":
    input_csv = "results.csv"
    df = pd.read_csv(input_csv)
    df = get_finished_games_from_thesportsdb(df)
    df.to_csv("results2.csv", index=False)
