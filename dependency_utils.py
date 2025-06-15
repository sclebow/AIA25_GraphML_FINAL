# This is a python script that parses a user provided csv file with task attributes and dependency information
# This file is refered to as the WBS (Work Breakdown Structure)
# The script will build a dictionary from the WBS file and create utility functions to access the data and calculate time estimates 

import pandas as pd

WBS_FILE_PATH = 'wbs_data.csv'  # Path to the WBS file

def load_wbs(file_path):
    """
    Load the Work Breakdown Structure (WBS) from an Excel file.
    
    Args:
        file_path (str): Path to the Excel file containing the WBS.
        
    Returns:
        pd.DataFrame: DataFrame containing the WBS data.
    """
    try:
        wbs_df = pd.read_csv(file_path)

        # Print some information about the loaded DataFrame
        print(f"WBS loaded successfully with {len(wbs_df)} rows and {len(wbs_df.columns)} columns.")
        print("Columns in WBS:", wbs_df.columns.tolist())
        # Optionally, display the first few rows of the DataFrame
        print("First few rows of WBS:")
        print(wbs_df.head())

        return wbs_df
    except Exception as e:
        print(f"Error loading WBS file: {e}")
        return None

df = load_wbs(WBS_FILE_PATH)

