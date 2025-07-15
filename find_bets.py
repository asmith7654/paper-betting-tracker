"""
find_bets.py

The file fetches odds using a separate file called fetch_odds.py, then identifies profitable bets from them
using three different strategies. The strategies are comparing odds to the average fair odds of an outcome,
computing the modified Z-score of the odds of an outcome, and comparing odds to the fair odds of Pinnacle
sportsbook (a known "sharp" sportsbook). Profitable bets are then saved into a master .csv file.

Author: Andrew Smith
Date: July 2025
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
from fetch_odds import fetch_odds, organize

# --------------------------------------- Helper functions --------------------------------------- #
def prettify_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes column names Title-case and replaces underscores with spaces.

    Args: 
        df (pd.DataFrame): Any data frame with a header row.
    """
    mapping = {c: c.replace("_", " ").title() for c in df.columns}
    return df.rename(columns=mapping)


def clean_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any odds equal to 1.0 with NaN.
    
    Args:
        df (pd.Dataframe): A dataframe where most/all numeric columns are odds.
    """
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].where(df[numeric_cols] != 1, np.nan)
    return df


def min_bookmakers_mask(row: pd.Series, min_bm: int = 5) -> bool:
    """
    Return True if row has at least min_bm non‑na numeric bookmaker odds.
    
    Args:
        row (pd.Series): A row of odds for an outcome.
        min_bm (int): The minimum number of bookmakers desired for sample size purposes.
    """
    count = sum(isinstance(v, (float, int)) and not pd.isna(v) for v in row.values)
    return count >= min_bm

def _safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return -np.inf

def df_requirements(df: pd.DataFrame) -> pd.DataFrame:
    vf_cols = [c for c in df.columns if c.startswith("Vigfree ")]
    exclude_cols = set(vf_cols).union({"Best Odds"})
    bm_cols = [
        c for c in df.select_dtypes(include=["float", "int", "object"]).columns
        if c not in exclude_cols
    ]

    def row_filter(row):
        num_valid = sum(
            pd.notna(row[b]) and isinstance(_safe_float(row[b]), float)
            for b in bm_cols
        )
        if num_valid < 5:
            return False
        if any(
            pd.notna(row[b]) and _safe_float(row[b]) > 50
            for b in bm_cols
        ):
            return False
        return True

    filtered_df = df[df.apply(row_filter, axis=1)].copy()
    print(f"Filtered: {len(df)} → {len(filtered_df)} rows")
    return filtered_df


def _append_unique(df_to_append: pd.DataFrame,
                   csv_path: str,
                   key_cols: list[str]) -> None:
    """
    Append rows to csv_path unless a row with the same key_cols tuple already
    exists. Adds a `Scrape Time` column automatically. 

    Args:
        df_to_append (pd.Dataframe): Dataframe with the new bets to be added.
        csv_path (str): Path leading to the master data set of minimal bet information.
        key_cols (list[str]): Columns that uniquely identify a possible bet. For example, "Match"
            and "Start Time".
    """
    df_to_append = df_to_append.copy()
    df_to_append["Scrape Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_to_append = prettify_headers(df_to_append)

    # --- If no CSV yet, just write it -------------------------------------
    if not os.path.exists(csv_path):
        df_to_append.to_csv(csv_path, index=False)
        print(f"Created {csv_path} ({len(df_to_append)} rows)")
        return

    # --- Load existing, align schemas -------------------------------------
    existing = pd.read_csv(csv_path)

    # Union of columns (keeps order: old cols first, then any new ones)
    all_cols = list(existing.columns)
    for c in df_to_append.columns:
        if c not in all_cols:
            all_cols.append(c)

    # Add missing cols with NaN
    for c in all_cols:
        if c not in existing.columns:
            existing[c] = np.nan
        if c not in df_to_append.columns:
            df_to_append[c] = np.nan

    # Re‑order both DataFrames
    existing   = existing[all_cols]
    df_to_append = df_to_append[all_cols]

    # --- Deduplicate on key_cols ------------------------------------------
    # ----- build a set of existing keys (match, date) ------------------
    existing_keys = {
        (row["Match"], start_date(row["Start Time"]))
        for _, row in existing.iterrows()
    }

    # ----- find rows whose (match, team, DATE) combo is new ------------------
    new_rows_mask = df_to_append.apply(
        lambda r: (r["Match"], start_date(r["Start Time"])) not in existing_keys,
        axis=1,
    )
    new_rows = df_to_append[new_rows_mask]


    if new_rows.empty:
        print(f"No new rows for {csv_path}")
        return

    # --- Concatenate & rewrite entire CSV ---------------------------------
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"Appended {len(new_rows)} rows to {csv_path} (file rewritten)")


def start_date(ts) -> str:
    """
    Convert a timestamp / datetime-like / ISO string to 'YYYY-MM-DD'.

    Examples
    --------
    >>> start_date("2025-07-13 04:00:44")
    '2025-07-13'
    >>> start_date(pd.Timestamp("2025-07-13 04:00:00"))
    '2025-07-13'
    """
    return pd.to_datetime(ts).strftime("%Y-%m-%d")


# ----------------------------- Logging profitable bets into master .csv files ----------------------------- #
def log_unique_bets(summary_df: pd.DataFrame, csv_path: str) -> None:
    """
    Deduplicate summary_df so only the most profitable side per
    (Match, Start Time) is kept, then append to csv_path.

    Args:
        summary_df (pd.Dataframe): A dataframe of paper bets with only essential information included.
        csv_path (str): Path leading to the master data set of minimal bet information.
    """
    if summary_df.empty:
        print("Summary DataFrame empty, nothing to log.")
        return

    conflict_key = ["Match", "Start Time"]
    # infer score column name
    for candidate in ("Avg Edge Pct", "Z Score", "Edge Vs Pinnacle Pct"):
        if candidate in summary_df.columns:
            score_col = candidate
            break
    else:
        raise ValueError("No score column found in summary DataFrame.")

    best_side = (
        summary_df.sort_values(score_col, ascending=False)
                  .drop_duplicates(subset=conflict_key, keep="first")
    )
    _append_unique(best_side, csv_path, ["Match", "Start Time"])


def log_full_rows(source_df: pd.DataFrame,
                  summary_df: pd.DataFrame,
                  full_csv_path: str) -> None:
    """
    Copy all columns from source_df that correspond to bets in summary_df,
    except the bulky vigfree / diagnostic columns, and append to full_csv_path.

    Args:
        source_df (pd.Dataframe): A dataframe containing all information available for each bet.
        summary_df (pd.Dataframe): A dataframe of paper bets with only essential information included.
        full_csv_path: Path leading to the master data set of maximal bet information.
    """
    if summary_df.empty:
        print("No summaries to copy to full log.")
        return

    key = ["Match", "Team", "Start Time"]   # <- add Team back
    merged = pd.merge(summary_df[key], source_df, on=key, how="left")

    # Drop columns we do not want carrying over
    drop_prefixes = ("Vigfree ",)
    score_col = None
    for candidate in ("Avg Edge Pct", "Z Score", "Edge Vs Pinnacle Pct"):
        if candidate in summary_df.columns:
            score_col = candidate
            break
    else:
        raise ValueError("No score column found in summary DataFrame.")
    
    if score_col == "Avg Edge Pct":
        drop_exact    = {
            "Largest Outlier Book", "Z Score",
            "Best Avg Book",
            "Pinnacle Edge Book", "Pin Edge Pct"
        }
    elif score_col == "Edge Vs Pinnacle Pct":
        drop_exact    = {
            "Largest Outlier Book", "Z Score",
            "Best Avg Book", "Avg Edge Pct",
            "Pinnacle Edge Book",
        }
    else:
        drop_exact    = {
            "Largest Outlier Book",
            "Best Avg Book", "Avg Edge Pct",
            "Pinnacle Edge Book", "Pin Edge Pct",
            "Vigfree Best Odds"
        }
    
    keep_cols = [c for c in merged.columns
                 if not c.startswith(drop_prefixes) and c not in drop_exact]

    _append_unique(merged[keep_cols], full_csv_path, ["Match", "Start Time"])


# ----------------------------------- Vig‑free implied probabilities ----------------------------------- #
def add_vig_free_implied_probs(df: pd.DataFrame,
                               min_outcomes: int | None = None) -> pd.DataFrame:
    """
    Adds 'Vigfree <Bookmaker>' columns that contain the implied probability of a bookmakers odds, after
    removing the vig. A bookmaker is processed only if it has odds for all outcomes in the match
    (or at least `min_outcomes` if you pass one). This is because all outcomes are needed for 
    normalization of probabilites, or "de-vigging".

    Args:
        df (pd.Dataframe): A dataframe whose float or int columns are odds.
        min_outcomes (int): The minimum number of outcomes of a match for the function to run.
    """
    df = df.copy()
    _EXCLUDE_EXACT = {"Best Odds"}
    bm_cols = [
        c for c in df.select_dtypes(include=["float", "int"]).columns
        if c not in _EXCLUDE_EXACT
    ]

    for bm in bm_cols:
        vf_col = f"Vigfree {bm}"
        df[vf_col] = np.nan

        for match, sub in df.groupby("Match", sort=False):
            needed = len(sub) if min_outcomes is None else min_outcomes
            odds = sub[bm].where(sub[bm] > 0).dropna()

            if len(odds) < needed:        
                continue                

            probs = 1 / odds
            probs /= probs.sum()          
            df.loc[odds.index, vf_col] = probs.values
    return df



# --------------------------------------------  Average‑edge  -------------------------------------------- #
def add_avg_edge_info(df: pd.DataFrame,
                      edge_threshold: float = 0.05,
                      max_missing_vf_with_odds: int = 2) -> pd.DataFrame:
    """
    Identifies odds that are a higher payout than the average fair odds of each bookmaker. Creates columns
    'Best Avg Book', 'Avg Edge Pct', and 'Fair Odds Avg'. A row is only kept if no more than 
    max_missing_vf_with_odds bookmakers have odds but lack a vig-free probability. This is 
    to limit bets where the average fair odds does not represent the sample of odds available,
    because they contain a high number of odds that can not be normalized/"de-vigged".

    Args:
        df (pd.Dataframe): A dataframe whose float or int columns are odds, and contains vig-free implied odds.
        edge_threshold (float): The lowest expected value percentage acceptable for odds.
        max_missing_vf_with_odds (int): The maximum number of bookmakers in a row who have odds, but no
            vig-free implied odds.
    """
    df = df.copy()

    _EXCLUDE_EXACT = {"Best Odds"}
    vf_cols = [c for c in df.columns if c.startswith("Vigfree ")]
    bm_cols = [
        c for c in df.select_dtypes(include=["float", "int"]).columns
        if c not in vf_cols and c not in _EXCLUDE_EXACT
    ]

    best_book, best_edge, fair_odds_list = [], [], []

    for _, row in df.iterrows():
        # Count how many bookmakers have odds but no vig-free prob
        missing_vf_with_odds = 0
        for bm in bm_cols:
            if pd.notna(row[bm]):  # has odds
                vf_col = f"Vigfree {bm}"
                if vf_col in row and pd.isna(row[vf_col]):
                    missing_vf_with_odds += 1

        if missing_vf_with_odds > max_missing_vf_with_odds:
            best_book.append(None)
            best_edge.append(None)
            fair_odds_list.append(None)
            continue

        # Use only valid vig-free probabilities
        probs = [row[c] for c in vf_cols if pd.notnull(row[c])]
        if not probs:
            best_book.append(None)
            best_edge.append(None)
            fair_odds_list.append(None)
            continue

        fair_odds = 1 / np.mean(probs)
        fair_odds_list.append(round(fair_odds, 3))

        # Look for the best edge
        row_best_edge, row_best_book = -np.inf, None
        for bm in bm_cols:
            price = row[bm]
            if pd.isna(price):
                continue
            edge = price / fair_odds - 1
            if edge > row_best_edge and edge >= edge_threshold:
                row_best_edge = edge
                row_best_book = bm

        best_book.append(row_best_book)
        best_edge.append(round(row_best_edge * 100, 2) if row_best_book else None)

    df["Best Avg Book"] = best_book
    df["Avg Edge Pct"] = best_edge
    df["Fair Odds Avg"] = fair_odds_list
    return df



def summarize_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy summary of average‑edge bets, one row per profitable bet.
    Excludes rows where the best odds > 50.

    Args:
        df (pd.Dataframe): A dataframe that contains average edge info.
    """
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["Best Avg Book"]) or pd.isna(r["Avg Edge Pct"]) or pd.isna(r["Fair Odds Avg"]):
            continue
        price = r[r["Best Avg Book"]]
        rows.append({
            "Match": r["Match"],
            "Team": r["Team"],
            "Start Time": r["Start Time"],
            "Avg Edge Book": r["Best Avg Book"],
            "Avg Edge Odds": price,
            "Avg Edge Pct": r["Avg Edge Pct"],
        })
    return pd.DataFrame(rows)


# --------------------------------------- Z‑score outliers --------------------------------------- #
def add_largest_zscore_info(df: pd.DataFrame, z_thresh: float = 2) -> pd.DataFrame:
    """
    Identifies odds that are greater than z_thresh modified Z-score. Creates columns
    'Largest Outlier Book' and 'Z Score'.

    Args:
        df (pd.Dataframe): A dataframe whose float or int columns are odds.
        z_thresh (float): The lowest Z-score possible for a profitable bet.
    """
    df = df.copy()

    _EXCLUDE_EXACT = {"Best Odds"}
    vf_cols = [c for c in df.columns if c.startswith("Vigfree ")]
    bm_cols = [
        c for c in df.select_dtypes(include=["float", "int"]).columns
        if c not in vf_cols and c not in _EXCLUDE_EXACT
    ]

    names, scores = [], []

    for _, row in df.iterrows():
        odds = row[bm_cols].dropna().values
        mean = np.mean(odds)
        sd = np.std(odds,ddof=1)
        z = np.maximum(0, odds - mean) / (sd or 1e-6)
        #when z score is particularly high, do not consider
        z[z > 6] = 0 
        idx = np.where(z > z_thresh)[0]
        if idx.size == 0:
            names.append(None); scores.append(None); continue
        best = idx[np.argmax(z[idx])]
        names.append(bm_cols[best]); scores.append(round(z[best], 2))

    df["Largest Outlier Book"] = names
    df["Z Score"]              = scores
    return df


def summarize_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy summary of outlier bets, one row per profitable bet.
    Excludes rows where the best odds > 50.

    Args:
        df (pd.Dataframe): A dataframe that contains Z-score info.
    """
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["Largest Outlier Book"]) or pd.isna(r["Z Score"]):
            continue
        price = r[r["Largest Outlier Book"]]
        rows.append({
            "Match": r["Match"],
            "Team": r["Team"],
            "Start Time": r["Start Time"],
            "Outlier Book": r["Largest Outlier Book"],
            "Outlier Odds": price,
            "Z Score": r["Z Score"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------  Modified Z‑score outliers --------------------------------------- #
def add_largest_mod_zscore_info(df: pd.DataFrame, z_thresh: float = 2) -> pd.DataFrame:
    """
    Identifies odds that are greater than z_thresh modified Z-score. Creates columns
    'Largest Outlier Book' and 'Z Score'.

    Args:
        df (pd.Dataframe): A dataframe whose float or int columns are odds.
        z_thresh (float): The lowest Z-score possible for a profitable bet.
    """
    df = df.copy()

    _EXCLUDE_EXACT = {"Best Odds"}
    vf_cols = [c for c in df.columns if c.startswith("Vigfree ")]
    bm_cols = [
        c for c in df.select_dtypes(include=["float", "int"]).columns
        if c not in vf_cols and c not in _EXCLUDE_EXACT
    ]

    names, scores = [], []

    for _, row in df.iterrows():
        odds = row[bm_cols].dropna().values
        median = np.median(odds)
        z = 0.6745 * np.maximum(0, odds - median) / (np.median(np.abs(odds - median)) or 1e-6)
        #when z score is particularly high, do not consider
        z[z > 6] = 0 
        idx = np.where(z > z_thresh)[0]
        if idx.size == 0:
            names.append(None); scores.append(None); continue
        best = idx[np.argmax(z[idx])]
        names.append(bm_cols[best]); scores.append(round(z[best], 2))

    df["Largest Outlier Book"] = names
    df["Modified Z Score"]     = scores
    return df


def summarize_mod_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy summary of outlier bets, one row per profitable bet.
    Excludes rows where the best odds > 50.

    Args:
        df (pd.Dataframe): A dataframe that contains Z-score info.
    """
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["Largest Outlier Book"]) or pd.isna(r["Modified Z Score"]):
            continue
        price = r[r["Largest Outlier Book"]]
        rows.append({
            "Match": r["Match"],
            "Team": r["Team"],
            "Start Time": r["Start Time"],
            "Outlier Book": r["Largest Outlier Book"],
            "Outlier Odds": price,
            "Modified Z Score": r["Modified Z Score"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------  Pinnacle edge --------------------------------------------- #
def add_pinnacle_edge_info(df: pd.DataFrame,
                           pinnacle_col: str = "Pinnacle",
                           edge_threshold: float = 0.05) -> pd.DataFrame:
    """
    Identifies odds that are a greater payout than Pinnacle sportsbook's fair odds. Creates columns
    'Pinnacle Fair Odds', 'Pinnacle Edge Book', and "Pin Edge Pct".

    Args:
        df (pd.Dataframe): A dataframe whose float or int columns are odds, and contains Pinnacle odds.
        pinnacle_col (str): The title of the column for Pinnacle's odds.
        edge_threshold (float): The lowest expected value percentage acceptable for odds.
    """
    df = df.copy()
    vf_pin = f"Vigfree {pinnacle_col}"
    if vf_pin not in df.columns:
        raise ValueError(f"{vf_pin} column missing – run vig‑free step first")

    _EXCLUDE_EXACT = {"Best Odds"}
    vf_cols = [c for c in df.columns if c.startswith("Vigfree ")]
    bm_cols = [
        c for c in df.select_dtypes(include=["float", "int"]).columns
        if c not in vf_cols and c not in _EXCLUDE_EXACT
    ]

    pin_fair, best_book, best_edge = [], [], []

    for _, row in df.iterrows():
        pin_prob = row[vf_pin]
        if pd.isna(pin_prob) or pin_prob <= 0:
            pin_fair.append(None); best_book.append(None); best_edge.append(None); continue

        fair_odds = 1 / pin_prob
        pin_fair.append(round(fair_odds, 3))

        row_best_edge, row_best_book = -np.inf, None
        for bm in bm_cols:
            if bm == pinnacle_col:
                continue
            price = row[bm]
            if pd.isna(price):
                continue
            edge = price / fair_odds - 1
            if edge > row_best_edge and edge >= edge_threshold:
                row_best_edge, row_best_book = edge, bm

        best_book.append(row_best_book)
        best_edge.append(round(row_best_edge * 100, 2) if row_best_book else None)

    df["Pinnacle Fair Odds"] = pin_fair
    df["Pinnacle Edge Book"] = best_book
    df["Pin Edge Pct"]       = best_edge
    return df


def summarize_pinnacle_edge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build tidy summary of Pinnacle-edge bets, one row per profitable bet.
    Excludes rows where the best odds > 50.

    Args:
        df (pd.Dataframe): A dataframe that contains Pinnacle odds info.
    """
    rows = []
    for match, sub in df.groupby("Match"):
        sub = sub.dropna(subset=["Pinnacle Edge Book", "Pin Edge Pct"])
        if sub.empty:
            continue
        best = sub.loc[sub["Pin Edge Pct"].idxmax()]
        price = best[best["Pinnacle Edge Book"]]
        if pd.isna(price):
            continue
        rows.append({
            "Match": match,
            "Team": best["Team"],
            "Start Time": best["Start Time"],
            "Pinnacle Edge Book": best["Pinnacle Edge Book"],
            "Pinnacle Edge Odds": price,
            "Edge Vs Pinnacle Pct": best["Pin Edge Pct"],
            "Pinnacle Fair Odds": best["Pinnacle Fair Odds"],
        })
    return pd.DataFrame(rows)



# --------------------------------------------------------------------------- #
#  MAIN PIPELINE                                                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # 1. Fetch, organize, and filter -----------------------------------------
    organized = prettify_headers(clean_odds(organize(fetch_odds())))
    filtered = df_requirements(organized)

    # 2. Compute vig‑free
    vf_df = add_vig_free_implied_probs(filtered)
    #vf_df.to_csv("vig.csv",index=False)

    # 3‑A Average edge -------------------------------------------------------
    avg_df      = add_avg_edge_info(vf_df, edge_threshold=0.05)
    avg_summary = summarize_avg(avg_df)
    #avg_summary.to_csv("avg_edge.csv", index=False)

    if not avg_summary.empty:
        log_unique_bets(avg_summary, "master_avg_bets.csv")
        log_full_rows(avg_df, pd.read_csv("master_avg_bets.csv"), "master_avg_full.csv")

    # 3‑B Z‑scores -----------------------------------------------------------
    z_df        = add_largest_zscore_info(vf_df, z_thresh=2)
    z_summary   = summarize_zscores(z_df)
    #z_summary.to_csv("z.csv", index=False)

    if not z_summary.empty:
        log_unique_bets(z_summary, "master_zscore_bets.csv")
        log_full_rows(z_df, pd.read_csv("master_zscore_bets.csv"), "master_zscore_full.csv")

    # 3‑C Modified Z‑scores ---------------------------------------------------
    mod_z_df        = add_largest_mod_zscore_info(vf_df, z_thresh=2)
    mod_z_summary   = summarize_mod_zscores(mod_z_df)
    #mod_z_summary.to_csv("mod_z.csv", index=False)

    if not mod_z_summary.empty:
        log_unique_bets(mod_z_summary, "master_mod_zscore_bets.csv")
        log_full_rows(mod_z_df, pd.read_csv("master_mod_zscore_bets.csv"), "master_mod_zscore_full.csv")

    # 3‑D  Pinnacle edge  ------------------------------------------------------
    if "Pinnacle" in vf_df.columns and vf_df["Pinnacle"].notna().any():
        pin_df      = add_pinnacle_edge_info(vf_df, edge_threshold=0.05)
        pin_summary = summarize_pinnacle_edge(pin_df)

        if not pin_summary.empty:
            log_unique_bets(pin_summary, "master_pin_bets.csv")
            log_full_rows(pin_df, pin_summary, "master_pin_full.csv")
    else:
        print("⚠️  No Pinnacle odds found – skipping Pinnacle‑edge step")