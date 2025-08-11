import os
import pandas as pd
import numpy as np
from datetime import datetime

class SleepDataProcessor:
    def __init__(self, file_path, user_id):
        #Initialize  file path and user ID
        self.file_path = file_path
        self.user_id = user_id
        self.dfs = {}
        self.clean_dfs = {}

    def load_data(self):
        #Separate dataframes
        df = pd.read_csv(self.file_path)

        self.dfs['df_sleep'] = df[['startdate', 'enddate', 'Sleep state']]
        self.dfs['df_heart_rate'] = df[['Heart Rate date', 'Heart Rate']]
        self.dfs['df_respiration_rate'] = df[['Respiration rate date', 'Respiration rate']]
        self.dfs['df_snoring'] = df[['Snoring date', 'Snoring']]

    def remove_duplicates(self):
        #Remove duplicates
        key_columns = {
            "df_sleep": "startdate",
            "df_heart_rate": "Heart Rate date",
            "df_respiration_rate": "Respiration rate date",
            "df_snoring": "Snoring date",
            "df_sdnn": "sdnn_1 date"
        }

        for name, df in self.dfs.items():
            self.clean_dfs[name] = df.drop_duplicates(subset=[key_columns[name]], keep='first')

    def process_sleep_data(self):
        # Sleep preprocessing
        df_sleep = self.clean_dfs['df_sleep'].copy()
        df_sleep = df_sleep[df_sleep['startdate'] > 0]
        df_sleep['startdate'] = pd.to_datetime(df_sleep['startdate'], unit='s', utc=True) 
        df_sleep['enddate'] = pd.to_datetime(df_sleep['enddate'], unit='s', utc=True) 
        df_sleep['date'] = (df_sleep['startdate'] - pd.Timedelta(hours=12)).dt.date
        df_sleep['duration'] = (df_sleep['enddate'] - df_sleep['startdate']).dt.total_seconds()
        df_duration = df_sleep.groupby(['date', 'Sleep state']).agg({
            'duration': 'sum',             
        }).reset_index().pivot(index='date', columns='Sleep state', values='duration').reset_index()#.unstack(fill_value=0)
        df_dates = df_sleep.groupby(['date']).agg({      
            'startdate': 'min',      
            'enddate': 'max'         
        }).reset_index()
        sleep_summary = df_duration.merge(df_dates, on=['date'])
        sleep_summary.rename(columns={0: "Awake (s)", 1: "Light Sleep (s)", 2: "Deep Sleep (s)", 
                                      3: "REM Sleep (s)", 5: "Unspecified Sleep (s)"}, inplace=True)
        
        # Daily aggregated snoring
        df_snore = self.clean_dfs['df_snoring'].copy()
        df_snore['Snoring date'] = pd.to_datetime(df_snore['Snoring date'], unit='s', utc=True) 
        df_snore['date'] = (df_snore['Snoring date'] - pd.Timedelta(hours=12)).dt.date
        snore_summary = df_snore.groupby('date')['Snoring'].sum()

        #Daily agreggated hr and rr 
        def process_metric(df, timestamp_col, value_col, metric_name):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s', utc=True)
            df['date'] = (df[timestamp_col] - pd.Timedelta(hours=12)).dt.date
            summary = df.groupby('date')[value_col].agg(['mean', 'min', 'max'])
            return summary.rename(columns={'mean': f'Avg {metric_name}', 'min': f'Min {metric_name}', 'max': f'Max {metric_name}'})

        heart_summary = process_metric(self.clean_dfs['df_heart_rate'], 'Heart Rate date', 'Heart Rate', 'Heart Rate')
        resp_summary = process_metric(self.clean_dfs['df_respiration_rate'], 'Respiration rate date', 'Respiration rate', 'Respiration Rate')

        #Merging daily agreggated data
        final_summary = sleep_summary.merge(heart_summary, on=['date'], how='outer') \
                                     .merge(resp_summary, on=['date'], how='outer') \
                                     .merge(snore_summary, on=['date'], how='outer')

        final_summary['User ID'] = self.user_id  # Add user ID 
        self.resilient_dataset = final_summary
        return final_summary
        return sleep_summary

    def run_pipeline(self):
        #Pipeline running
        self.load_data()
        self.remove_duplicates()
        final_summary = self.process_sleep_data()
        return final_summary  # Return processed data for aggregation

def process_multiple_users(base_directory):
    #Process multiple folders
    all_summaries = []  # List to collect summaries

    for folder in sorted(os.listdir(base_directory)):  # Sort to process users in order
        folder_path = os.path.join(base_directory, folder)
        file_path = os.path.join(folder_path, "Sleepmat_intra_activity.csv")

        if os.path.isdir(folder_path) and os.path.exists(file_path): 
            print(f"Processing user: {folder}")

            processor = SleepDataProcessor(file_path, user_id=folder)
            summary_df = processor.run_pipeline()  # Run pipeline & get summary
            
            if summary_df is not None:
                all_summaries.append(summary_df)

    # Combine all user summaries into one dataframe
    if all_summaries:
        complete_summary = pd.concat(all_summaries).reset_index().drop(columns=['index','date'])
        columns_to_move = ['User ID','startdate', 'enddate']
        remaining_columns = [col for col in complete_summary.columns if col not in columns_to_move]
        new_order = columns_to_move + remaining_columns
        complete_summary = complete_summary[new_order]
        complete_summary.to_csv("./data/resilient_sleep_mat.csv", index=False)
        print(" summary saved as 'resilient_sleep_mat.csv'.")

    else:
        print("No valid data in the specified directory.")
     
