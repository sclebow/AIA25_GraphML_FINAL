# This is a python script that parses a user provided csv file with task attributes and dependency information
# This file is refered to as the WBS (Work Breakdown Structure)
# The script will build a dictionary from the WBS file and create utility functions to access the data and calculate time estimates 

import pandas as pd

def get_wbs_from_directory(directory):
    """
    Get the latest WBS file from the specified directory.
    This function searches for CSV or Excel files in the given directory and returns the path to the most recently modified file.
    If no WBS files are found, it raises a FileNotFoundError.
    If the directory does not exist, it raises a FileNotFoundError.

    :param directory: Directory to search for WBS files.
    :return: Path to the latest WBS file.    
    """

    import os
    import glob

    # Ensure the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")

    # Get all CSV or excel files in the directory
    wbs_files = glob.glob(os.path.join(directory, '*.csv')) + glob.glob(os.path.join(directory, '*.xlsx'))

    if not wbs_files:
        raise FileNotFoundError("No WBS files found in the specified directory.")

    # Sort files by modification time and return the latest one
    latest_file = max(wbs_files, key=os.path.getmtime)
    return latest_file

def load_wbs(directory):
    """
    Load the Work Breakdown Structure (WBS) from a file in the specified directory.
    This function retrieves the latest WBS file from the directory, reads it into a pandas DataFrame,
    and returns the DataFrame. It also prints some information about the loaded data.
    If the file cannot be loaded, it prints an error message and returns None.

    :param directory: Directory where the WBS file is located.
    :return: DataFrame containing the WBS data.
    """
    # Get the latest WBS file from the specified directory
    file_path = get_wbs_from_directory(directory)
    print(f"Loading WBS from file: {file_path}")

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
