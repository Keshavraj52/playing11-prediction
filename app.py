import streamlit as st
import pandas as pd

def load_data():
    """Function to upload and read CSV files."""
    deliveries_file = st.file_uploader("Upload deliveries.csv", type=["csv"])
    matches_file = st.file_uploader("Upload matches.csv", type=["csv"])
    
    if deliveries_file and matches_file:
        deliveries_df = pd.read_csv(deliveries_file)
        matches_df = pd.read_csv(matches_file)
        return deliveries_df, matches_df
    return None, None

def analyze_data(deliveries_df):
    """Analyze batting, bowling, and all-rounder performance."""
    # Batting performance
    batting_stats = deliveries_df.groupby("batsman").agg(
        total_runs=("batsman_runs", "sum"),
        balls_faced=("ball", "count"),
        dismissals=("player_dismissed", "count")
    ).reset_index()
    batting_stats["strike_rate"] = (batting_stats["total_runs"] / batting_stats["balls_faced"]) * 100
    batting_stats["batting_avg"] = batting_stats["total_runs"] / batting_stats["dismissals"].replace(0, 1)
    
    # Bowling performance
    bowling_stats = deliveries_df.groupby("bowler").agg(
        wickets=("is_wicket", "sum"),
        balls_bowled=("ball", "count"),
        runs_conceded=("total_runs", "sum")
    ).reset_index()
    bowling_stats["economy_rate"] = (bowling_stats["runs_conceded"] / bowling_stats["balls_bowled"]) * 6
    bowling_stats["bowling_avg"] = bowling_stats["runs_conceded"] / bowling_stats["wickets"].replace(0, 1)
    
    # Identify all-rounders
    all_rounders = set(batting_stats["batter"]).intersection(set(bowling_stats["bowler"]))
    top_all_rounders = batting_stats[batting_stats["batter"].isin(all_rounders)].merge(
        bowling_stats, left_on="batter", right_on="bowler"
    ).sort_values(by=["wickets", "total_runs"], ascending=[False, False]).head(3)
    
    # Select top players
    top_batsmen = batting_stats.sort_values(by=["total_runs", "strike_rate"], ascending=[False, False]).head(4)
    top_bowlers = bowling_stats.sort_values(by=["wickets", "economy_rate"], ascending=[False, True]).head(4)
    
    return top_batsmen, top_bowlers, top_all_rounders

def main():
    st.title("Best 11 Players Predictor")
    deliveries_df, matches_df = load_data()
    
    if deliveries_df is not None and matches_df is not None:
        st.success("Files uploaded successfully!")
        top_batsmen, top_bowlers, top_all_rounders = analyze_data(deliveries_df)
        
        # Display results
        st.subheader("🏏 Top 4 Batsmen")
        st.dataframe(top_batsmen[["batter", "total_runs", "strike_rate", "batting_avg"]])
        
        st.subheader("🎯 Top 4 Bowlers")
        st.dataframe(top_bowlers[["bowler", "wickets", "economy_rate", "bowling_avg"]])
        
        st.subheader("🔥 Top 3 All-rounders")
        st.dataframe(top_all_rounders[["batter", "total_runs", "strike_rate", "wickets", "economy_rate"]])
    
if __name__ == "__main__":
    main()