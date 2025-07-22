"""
find_bets.py

The file fetches odds using a separate file called fetch_odds.py, then identifies profitable bets from them
using four different strategies. The strategies are comparing odds to the average fair odds of an outcome,
computing the Z-score and modified Z-score of the odds of an outcome, and comparing odds to the fair odds of 
Pinnacle sportsbook (a known "sharp" sportsbook). Profitable bets are then saved into a master .csv file.

Author: Andrew Smith
Date: July 2025
"""
# ---------------------------------------------- Imports ------------------------------------------------ #
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from fetch_odds import fetch_odds, organize


# ------------------------------------- Start time to date helper --------------------------------------- #
def _start_date(ts) -> str:
    """
    Convert a timestamp / datetime-like / ISO string to "YYYY-MM-DD".

    Args:
        ts (Any): Timestamp to convert to date.

    Returns:
        str: Date string.
    """
    return pd.to_datetime(ts).strftime("%Y-%m-%d")


# --------------------------------------- Bookmaker Columns Helper --------------------------------------- #
def _find_bookmakers(df: pd.DataFrame,
                     additional_cols: list[str] | None = None) -> list[str]:
    """
    Given a DataFrame, and possibly a list of non-bookmaker columns, find the bookmaker columns.

    Args:
        df (pd.DataFrame): Any DataFrame with bookmaker columns.
        additional_cols (list[str]): A list of columns to not include in the bookmaker columns.

    Returns:
        list[str]: The bookmaker column names for a DataFrame.
    """
    # Create a list of columns that may be interpreted as a float or int that we do not want
    excluded_cols = ["Best Odds","Start Time", "Last Update"]
    if additional_cols:
        excluded_cols += additional_cols

    bm_cols = [
        c for c in df.select_dtypes(include=["float", "int"]).columns
        if c not in excluded_cols
    ]
    return bm_cols


# ----------------------------------------- Cleaning functions ------------------------------------------- #
def _prettify_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes column names uppercase and replaces underscores with spaces.

    Args: 
        df (pd.DataFrame): Any DataFrame with a header row that needs to be formatted.

    Returns:
        pd.DataFrame: A DataFrame with nicely formatted headers.
    """
    mapping = {c: c.replace("_", " ").title() for c in df.columns}
    return df.rename(columns=mapping)


def _clean_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any odds equal to 1.0 with NaN.
    
    Args:
        df (pd.DataFrame): A DataFrame with bookmaker columns. Need to edit bm_cols if DataFrame includes vig columns.

    Returns:
        pd.DataFrame: A DataFrame with all float or int values that are equal to 1 converted to NaN.
    """
    bm_cols = _find_bookmakers(df)
    df[bm_cols] = df[bm_cols].where(df[bm_cols] != 1, np.nan)
    return df


def _safe_float(x) -> float:
    """
    Attempt to convert x to a float. Returns -np.inf if conversion fails.

    Args:
        x (Any): The input to convert.

    Returns:
        float: A float value or -np.inf if conversion fails.
    """
    try:
        return float(x)
    except (ValueError, TypeError):
        return -np.inf


def _requirements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame such that rows with <5 bookmakers are dropped, and rows with decimal odds over 50
    are dropped.

    Args:
        df (pd.DataFrame): A DataFrame that has been fetched and organized, with no additional information
            added (no vig free odds, no z score info, etc.).

    Returns:
        pd.DataFrame: A DataFrame that has been filtered with the aforementioned requirements.
    """
    df = df.copy()
    bm_cols = _find_bookmakers(df)

    def _row_filter(row):
        # Count number of non-nan rows
        num_valid = sum(
            pd.notna(row[b]) and isinstance(_safe_float(row[b]), float)
            for b in bm_cols
        )
        # Check minimum non-nan odds condition
        if num_valid < 5:
            return False
        
        # Check maximum odds condition
        if row["Best Odds"] > 50:
            return False
        
        return True

    filtered_df = df[df.apply(_row_filter, axis=1)]
    print(f"Filtered: {len(df)} → {len(filtered_df)} rows")
    return filtered_df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove odds equal to one, make headers neat, and filter out rows that have less than 5 bookmakers or
    have decimal odds greater than 50.

    Args:
        df (pd.DataFrame): A DataFrame that has been fetched and organized, with no additional information
            added (no vig free odds, no z score info, etc.).

    Returns:
        pd.DataFrame: A DataFrame that meets all the requirements for bet finding analysis.
    """
    df = _clean_odds(df)
    df = _prettify_headers(df)
    df = _requirements(df)
    return df


# ----------------------------------------- Dataframe appender -------------------------------------------- #
def _unique_cols(filename: str):
    """
    Return list of columns that are unique to a filename.

    Args:
        filename (str): The name of the file.

    Returns:
        Any: A list of column names, or None if no match.
    """
    mapping = {
        "master_avg_full.csv": ["Avg Edge Pct", "Fair Odds Avg"],
        "master_mod_zscore_full.csv": ["Modified Z Score"],
        "master_pin_full.csv": ["Pinnacle Fair Odds", "Pin Edge Pct"],
        "master_zscore_full.csv": ["Z Score"]
    }
    return mapping.get(filename)


def _append_unique(df_to_append: pd.DataFrame,
                   csv_path: str,) -> None:
    """
    Append rows to csv_path unless a row with the same defining attributes (Match, Date) already
    exists. Adds a "Scrape Time" column. 

    Args:
        df_to_append (pd.DataFrame): DataFrame with the new bets to be added.
        csv_path (str): Path leading to the master data set where bets will be appended.
    """
    df_to_append = df_to_append.copy()
    df_to_append["Scrape Time"] = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")

    # If no CSV yet, write it
    if not os.path.exists(csv_path):
        df_to_append.to_csv(csv_path, index=False)
        print(f"Created {csv_path} ({len(df_to_append)} rows)")
        return

    # Load existing, align schemas (for full .csv files, where new columns may arise)
    existing = pd.read_csv(csv_path)

    # Union of columns (keeps order: old cols first, then any new ones)
    all_cols = list(existing.columns)
    for c in df_to_append.columns:
        if c not in all_cols:
            all_cols.append(c)

    # Move specific columns to the end
    cols_to_move = (_unique_cols(csv_path) or []) + ["Best Odds", "Best Bookmaker", "Result", "Scrape Time"] 
    all_cols = [c for c in all_cols if c not in cols_to_move] + [c for c in cols_to_move if c in all_cols]

    # Add missing cols with NaN
    for c in all_cols:
        if c not in existing.columns and c != "Result":
            existing[c] = np.nan
        elif c not in existing.columns and c == "Result":
            existing[c] = "Not Found"
        if c not in df_to_append.columns and c != "Result":
            df_to_append[c] = np.nan
        elif c not in df_to_append.columns and c == "Result":
            df_to_append[c] = "Not Found"

    # Re‑order both DataFrames
    existing = existing[all_cols]
    df_to_append = df_to_append[all_cols]

    # Build a set of existing keys (match, date)
    existing_keys = {
        (row["Match"], _start_date(row["Start Time"]))
        for _, row in existing.iterrows()
    }

    # Find rows whose (match, date) combo is new
    new_rows_mask = df_to_append.apply(
        lambda r: (r["Match"], _start_date(r["Start Time"])) not in existing_keys,
        axis=1,
    )
    new_rows = df_to_append[new_rows_mask]


    if new_rows.empty:
        print(f"No new rows for {csv_path}, only duplicates")
        return

    # Concatenate & rewrite entire CSV
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"Appended {len(new_rows)} rows to {csv_path}")


# ----------------------------- Logging profitable bets into master .csv files ----------------------------- #
def log_unique_bets(summary_df: pd.DataFrame,
                    csv_path: str) -> None:
    """
    Deduplicate summary_df so only the most profitable side per
    (Match, Start Time) is kept, then append to csv_path.

    Args:
        summary_df (pd.DataFrame): A DataFrame of paper bets with only essential information included.
        csv_path (str): Path leading to the master data set of minimal bet information.
    """
    conflict_key = ["Match", "Start Time"]

    # Infer score column name
    for candidate in ("Avg Edge Pct", "Z Score", "Modified Z Score", "Pin Edge Pct"):
        if candidate in summary_df.columns:
            score_col = candidate
            break
    else:
        raise ValueError("No score column found in summary DataFrame.")

    # Drop bets that conflict
    best_side = (
        summary_df.sort_values(score_col, ascending=False)
                  .drop_duplicates(subset=conflict_key, keep="first")
    )
    _append_unique(best_side, csv_path)


def log_full_rows(source_df: pd.DataFrame,
                  summary_df: pd.DataFrame,
                  csv_path: str) -> None:
    """
    Copy all columns from source_df that correspond to bets in summary_df,
    except the vigfree columns, and append to full_csv_path.

    Args:
        source_df (pd.DataFrame): A DataFrame containing all information available for each bet.
        summary_df (pd.DataFrame): A DataFrame of paper bets with only essential information included.
        full_csv_path: Path leading to the master data set of maximal bet information.
    """
    # Combine source and summary df based on key columns
    key = ["Match", "Team", "Start Time"]
    merged = pd.merge(summary_df[key], source_df, on=key, how="left")

    # Drop columns we do not want carrying over
    drop_prefixes = ("Vigfree ",)
    keep_cols = [c for c in merged.columns if not c.startswith(drop_prefixes)]
    output = merged[keep_cols]

    # Infer score column name for sorting
    for candidate in ("Avg Edge Pct", "Z Score", "Modified Z Score", "Pin Edge Pct"):
        if candidate in output.columns:
            score_col = candidate
            break
    else:
        raise ValueError("No score column found in summary DataFrame.")

    # Sort to match "log_unique_bets"
    sorted = (output.sort_values(score_col, ascending=False))

    _append_unique(sorted, csv_path)


# ----------------------------------- Vig‑free implied probabilities ----------------------------------- #
def add_vig_free_implied_probs(df: pd.DataFrame,
                               min_outcomes: int | None = None) -> pd.DataFrame:
    """
    Adds "Vigfree <Bookmaker>" columns that contain the implied probability of a bookmakers odds, after
    removing the vig. A bookmaker is processed only if it has odds for all outcomes in the match
    (or at least "min_outcomes" if you pass one). This is because all outcomes are needed for 
    normalization of probabilites, or "de-vigging".

    Args:
        df (pd.DataFrame): A cleaned DataFrame with odds.
        min_outcomes (int): The minimum number of outcomes of a match for the function to run.

    Returns:
        pd.DataFrame: A DataFrame where each bookmaker for each match either has fair probs added, or nan
            added.
    """
    df = df.copy()
    bm_cols = _find_bookmakers(df)

    for bm in bm_cols:
        # Create vigfree columns for each bookmaker
        vf_col = f"Vigfree {bm}"
        df[vf_col] = np.nan

        for match, sub in df.groupby("Match", sort=False):
            # Calculate how many sides of an outcome there are, and after dropping na vals and odds less than 
            # or equal to zero for a bm, check to see if all sides of an outcome are offered (or at least 
            # min_outcomes are offered)
            needed = len(sub) if min_outcomes is None else min_outcomes
            odds = sub[bm].where(sub[bm] > 0).dropna()
            if len(odds) < needed:        
                continue                

            # Create fair probability and add to DataFrame
            probs = 1 / odds
            probs /= probs.sum()          
            df.loc[odds.index, vf_col] = probs.values
    return df


# --------------------------------------------  Average‑edge  -------------------------------------------- #
def _vigfree_test(bm_cols: list[str],
                  row: pd.Series,
                  max: int) -> bool:
    """
    Tests to see if the amount of bookmakers with odds but no vigfree odds is over max.

    Args:
        bm_cols (list[str]): A list of bookmakers in the row.
        row (pd.Series): A row for an outcome.
        max (int): The maximum number of bookmakers in a row who have odds, but no
            vig-free odds.

    Returns:
        bool: Returns True if the amount of bookmakers with odds but no vigfree odds is less than or
            equal to max, returns False otherwise.
    """
    missing_vf_with_odds = 0

    # Count how many columns do not meet conditon
    for bm in bm_cols:
        if pd.notna(row[bm]):  # Has odds
            vf_col = f"Vigfree {bm}"
            if vf_col in row and pd.isna(row[vf_col]): # Does not have vig free odds
                missing_vf_with_odds += 1

    if missing_vf_with_odds > max:
        return False
    else:
        return True


def add_avg_edge_info(df: pd.DataFrame,
                      edge_threshold: float = 0.05,
                      max_missing_vf_with_odds: int = 2) -> pd.DataFrame:
    """
    Checks if "Best Odds" are a higher payout than the average fair odds of the bookmakers. Creates columns
    "Avg Edge Pct" and "Fair Odds Avg". Values are only added to columns if no more than 
    max_missing_vf_with_odds bookmakers have odds but lack a vig-free probability (otherwise filled with None).
    This is to limit bets where the average fair odds does not represent the sample of odds available.

    Args:
        df (pd.DataFrame): A DataFrame that is cleaned and contains vig-free implied odds.
        edge_threshold (float): The lowest edge percentage acceptable to place a bet.
        max_missing_vf_with_odds (int): The maximum number of bookmakers in a row who have odds, but no
            vig-free implied odds.

    Returns:
        pd.DataFrame: A DataFrame with columns "Avg Edge Pct" and "Fair Odds Avg". Columns
            are filled with expected results or None.
    """
    df = df.copy()
    vf_cols = [c for c in df.columns if c.startswith("Vigfree ")]
    bm_cols = _find_bookmakers(df, vf_cols)
    best_edge, fair_odds_list = [], []

    for _, row in df.iterrows():
        # Test to see how many bookmakers have odds but no vig-free prob
        pass_vig_test = _vigfree_test(bm_cols, row, max_missing_vf_with_odds)
        if not pass_vig_test:
            best_edge.append(None); fair_odds_list.append(None); continue

        # Collect probabilities
        probs = [row[c] for c in vf_cols if pd.notnull(row[c])]
        if not probs:
            best_edge.append(None); fair_odds_list.append(None); continue

        # Calculate fair odds and append
        fair_odds = 1 / np.mean(probs)
        fair_odds_list.append(round(fair_odds, 3))

        # Calculate edge
        best_odds = row["Best Odds"]
        edge = best_odds / fair_odds - 1

        # Check edge threshold
        if edge > edge_threshold:
            best_edge.append(round(edge * 100, 2))
        else:
            best_edge.append(None)

    df["Fair Odds Avg"] = fair_odds_list
    df["Avg Edge Pct"] = best_edge
    return df


# --------------------------------------- Z‑score outliers --------------------------------------- #
def add_largest_zscore_info(df: pd.DataFrame, 
                            z_thresh: float = 2) -> pd.DataFrame:
    """
    Checks if "Best Odds" Z-score is greater than z_thresh. Creates column "Z Score".

    Args:
        df (pd.DataFrame): A cleaned DataFrame.
        z_thresh (float): The lowest Z-score acceptable for a profitable bet.

    Returns:
        pd.DataFrame: A DataFrame with added column "Z Score". The value in the column is the Z-score of 
            the best odds of the row if higher than the threshold.
    """
    df = df.copy()
    bm_cols = _find_bookmakers(df)
    zscores = []

    for _, row in df.iterrows():
        # Collect odds
        odds = [row[c] for c in bm_cols if pd.notnull(row[c])]
        if not odds:
            zscores.append(None); continue

        # Note best odds
        best_odds = row["Best Odds"]

        # Calculate Z-score
        mean = np.mean(odds)
        sd = np.std(odds, ddof=1)
        z = np.maximum(0, best_odds - mean) / (sd or 1e-6)

        # Append if Z-score meets threshold and is less than 6 (too extreme)
        if z > z_thresh and z < 6:
            zscores.append(round(z, 2))
        else:
            zscores.append(None)

    df["Z Score"] = zscores
    return df


# ---------------------------------------  Modified Z‑score outliers --------------------------------------- #
def add_largest_mod_zscore_info(df: pd.DataFrame,
                                z_thresh: float = 2) -> pd.DataFrame:
    """
    Checks if "Best Odds" Modified Z-score is greater than z_thresh. Creates column "Modified Z Score".

    Args:
        df (pd.DataFrame): A cleaned DataFrame.
        z_thresh (float): The lowest Z-score acceptable for a profitable bet.

    Returns:
        pd.DataFrame: A DataFrame with added column "Modified Z Score". The value in the column is the 
            Z-score of the best odds of the row if higher than the threshold.
    """
    df = df.copy()
    bm_cols = _find_bookmakers(df)
    mod_zscores = []

    for _, row in df.iterrows():
        # Collect odds
        odds = [row[c] for c in bm_cols if pd.notnull(row[c])]
        if not odds:
            mod_zscores.append(None); continue

        # Note best odds
        best_odds = row["Best Odds"]

        # Calculate Modified Z-score
        median = np.median(odds)
        mad = np.median(np.abs(odds - median)) or 1e-6
        z = 0.6745 * np.maximum(0, best_odds - median) / mad

        # Append if Z-score meets threshold and is less than 6 (too extreme)
        if z > z_thresh and z < 6:
            mod_zscores.append(round(z, 2))
        else:
            mod_zscores.append(None)

    df["Modified Z Score"] = mod_zscores
    return df


# ---------------------------------------------  Pinnacle edge --------------------------------------------- #
def add_pinnacle_edge_info(df: pd.DataFrame,
                           pinnacle_col: str = "Pinnacle",
                           edge_threshold: float = 0.05) -> pd.DataFrame:
    """
    Checks if "Best Odds" are a greater payout than Pinnacle sportsbook's fair odds. Creates columns
    "Pinnacle Fair Odds" and "Pin Edge Pct".

    Args:
        df (pd.DataFrame): A cleaned DataFrame with fair implied odds of each bookmaker.
        pinnacle_col (str): The title of the column for Pinnacle's odds.
        edge_threshold (float): The lowest edge percentage acceptable for odds.

    Returns:
        pd.DataFrame: A DataFrame with columns "Pinnacle Fair Odds" and "Pin Edge Pct". Columns
            are filled with expected results or None.
    """
    df = df.copy()
    vf_pin = f"Vigfree {pinnacle_col}"
    if vf_pin not in df.columns: # Throw error if no Vigfree Pinnacle column
        raise ValueError(f"{vf_pin} column missing - run vig-free step first")
    pin_fair, edge_pct = [], [],

    for _, row in df.iterrows():
        # Note Pinnacles fair probability
        pin_prob = row[vf_pin]
        if pd.isna(pin_prob) or pin_prob <= 0:
            pin_fair.append(None); edge_pct.append(None); continue

        # Calculate the fair odds
        fair_odds = 1 / pin_prob
        pin_fair.append(round(fair_odds, 3))

        # Calculate edge
        best_odds = row["Best Odds"]
        edge = best_odds / fair_odds - 1

        # Check edge threshold
        if edge > edge_threshold:
            edge_pct.append(round(edge * 100, 2))
        else:
            edge_pct.append(None)

    df["Pinnacle Fair Odds"] = pin_fair
    df["Pin Edge Pct"]       = edge_pct
    return df


# ------------------------------------------ Summary Dataframes ------------------------------------------ #
def summarize_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy summary of average-edge bets.

    Args:
        df (pd.DataFrame): A DataFrame that contains average-edge info.

    Returns:
        pd.DataFrame: DataFrame with only 6 essential columns per average-edge bet.
    """
    rows = []

    # Return empty DataFrame if input is None or empty
    if df is None or df.empty:
        return pd.DataFrame()

    for _, r in df.iterrows():
        if pd.isna(r["Avg Edge Pct"]) or pd.isna(r["Fair Odds Avg"]):
            continue

        rows.append({
            "Match": r["Match"],
            "League": r["League"],
            "Team": r["Team"],
            "Start Time": r["Start Time"],
            "Avg Edge Book": r["Best Bookmaker"],
            "Avg Edge Odds": r["Best Odds"],
            "Avg Edge Pct": r["Avg Edge Pct"],
            "Result": r["Result"],
        })
    return pd.DataFrame(rows)


def summarize_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy summary of Z-score bets.

    Args:
        df (pd.DataFrame): A DataFrame that contains Z-score info.

    Returns:
        pd.DataFrame: DataFrame with only 6 essential columns per Z-score bet.
    """
    rows = []

    # Return empty DataFrame if input is None or empty
    if df is None or df.empty:
        return pd.DataFrame()

    for _, r in df.iterrows():
        if pd.isna(r["Z Score"]):
            continue

        rows.append({
            "Match": r["Match"],
            "League": r["League"],
            "Team": r["Team"],
            "Start Time": r["Start Time"],
            "Outlier Book": r["Best Bookmaker"],
            "Outlier Odds": r["Best Odds"],
            "Z Score": r["Z Score"],
            "Result": r["Result"],
        })
    return pd.DataFrame(rows)


def summarize_mod_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy summary of Modified Z-score bets.

    Args:
        df (pd.DataFrame): A DataFrame that contains Modified Z-score info.

    Returns:
        pd.DataFrame: DataFrame with only 6 essential columns per Modified Z-score bet.
    """
    rows = []

    # Return empty DataFrame if input is None or empty
    if df is None or df.empty:
        return pd.DataFrame()

    for _, r in df.iterrows():
        if pd.isna(r["Modified Z Score"]):
            continue

        rows.append({
            "Match": r["Match"],
            "League": r["League"],
            "Team": r["Team"],
            "Start Time": r["Start Time"],
            "Outlier Book": r["Best Bookmaker"],
            "Outlier Odds": r["Best Odds"],
            "Modified Z Score": r["Modified Z Score"],
            "Result": r["Result"],
        })
    return pd.DataFrame(rows)


def summarize_pin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy summary of Pinnacle-edge bets.

    Args:
        df (pd.DataFrame): A DataFrame that contains Pinnacle odds info.
    
    Returns:
        pd.DataFrame: DataFrame with only 7 essential columns per Pinnacle bet.
    """
    rows = []

    # Return empty DataFrame if input is None or empty
    if df is None or df.empty:
        return pd.DataFrame()

    for _, r in df.iterrows():
        if pd.isna(r["Pinnacle Fair Odds"]) or pd.isna(r["Pin Edge Pct"]):
            continue

        rows.append({
            "Match": r["Match"],
            "League": r["League"],
            "Team": r["Team"],
            "Start Time": r["Start Time"],
            "Pinnacle Edge Book": r["Best Bookmaker"],
            "Pinnacle Edge Odds": r["Best Odds"],
            "Pin Edge Pct": r["Pin Edge Pct"],
            "Pinnacle Fair Odds": r["Pinnacle Fair Odds"],
            "Result": r["Result"],
        })
    return pd.DataFrame(rows)


# ------------------------------------------ Main Pipeline ------------------------------------------ #
if __name__ == "__main__":
    # 1. Fetch, organize, and clean -----------------------------------------------------------
    df = fetch_odds()
    organized = organize(df)
    print(organized)
    print(organzied.columns)
    cleaned = clean(organized)
    print(cleaned)
    print(cleaned.columns)

    # 2‑A Average-edge ------------------------------------------------------------------------
    #Compute vig‑free first
    vf_df = add_vig_free_implied_probs(cleaned)
    avg_df = add_avg_edge_info(vf_df, edge_threshold=0.05)
    avg_summary = summarize_avg(avg_df)

    if not avg_summary.empty:
        log_unique_bets(avg_summary, "master_avg_bets.csv")
        log_full_rows(avg_df, pd.read_csv("master_avg_bets.csv"), "master_avg_full.csv")
    else:
        print("No bets found for average-edge bets")

    # 2‑B Z‑scores ----------------------------------------------------------------------------
    z_df = add_largest_zscore_info(cleaned, z_thresh=2)
    z_summary = summarize_zscores(z_df)

    if not z_summary.empty:
        log_unique_bets(z_summary, "master_zscore_bets.csv")
        log_full_rows(z_df, pd.read_csv("master_zscore_bets.csv"), "master_zscore_full.csv")
    else:
        print("No bets found for Z-score bets")

    # 2‑C Modified Z‑scores -------------------------------------------------------------------
    mod_z_df = add_largest_mod_zscore_info(cleaned, z_thresh=2)
    mod_z_summary = summarize_mod_zscores(mod_z_df)

    if not mod_z_summary.empty:
        log_unique_bets(mod_z_summary, "master_mod_zscore_bets.csv")
        log_full_rows(mod_z_df, pd.read_csv("master_mod_zscore_bets.csv"), "master_mod_zscore_full.csv")
    else:
        print("No bets found for Modified Z-score bets")

    # 2‑D  Pinnacle-edge  ---------------------------------------------------------------------
    if "Pinnacle" in cleaned.columns and cleaned["Pinnacle"].notna().any():
        pin_df = add_pinnacle_edge_info(vf_df, edge_threshold=0.05)
        pin_summary = summarize_pin(pin_df)

        if not pin_summary.empty:
            log_unique_bets(pin_summary, "master_pin_bets.csv")
            log_full_rows(pin_df, pd.read_csv("master_pin_bets.csv"), "master_pin_full.csv")
        else:
            print("No bets found for Pinnacle-edge bets")
    else:
        print("No Pinnacle odds found - skipping Pinnacle-edge step")
