from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
import typing as t
import torch
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(44 + worker_id)
    
# DEFAULTS

@dataclass
class RandomState:
    seed: int = None

    def next(self, n_seeds: int = 1):
        generator = np.random.default_rng(self.seed)
        new_seed, *seeds = generator.integers(0, 2**32 - 1, size=n_seeds + 1)
        self.seed = new_seed
        seeds = [int(s) for s in seeds]
        if n_seeds == 1:
            return seeds[0]
        return seeds
    
@dataclass
class MainArgs:

    data_dir: str = os.path.join(".", "data")  # data directory
    id_col: str = "user_id"  # id column
    date_col: str = "start_date"  # start date column
    date_end_col: str = "end_date (UTC)"  # end date column (UTC)
    gender_col: str = "gender" # gender column
    age_col: str = "age" # age column
    timezone_col: str = "timezone"
    
    awake_duration_col: str = "awake_duration (s)"  # awake duration column
    bed_time_period_col_to_name: str = "bed_time_period" # the column name for the bed time period (including time out of bed during overall sleep)
    max_bed_time_period: int = 86400
    sleep_states = [
        "rem_duration (s)",
        "light_duration (s)",
        "deep_duration (s)",
        awake_duration_col,
    ]
    bed_time_period_cols_to_sum = sleep_states
    sleep_offset: int = 6  # this is the number of hours to subtract from
    time_in_bed_col_to_name: str = "time_in_bed"  # the column name for the time in bed
    time_out_of_bed_col_to_name: str = (
        "time_out_of_bed"  # the column name for the time out of bed
    )
    time_to_bed_col_to_name: str = (
        "time_to_bed"  # the column name for the time to bed
    )
    time_to_rise_col_to_name: str = (
        "time_to_rise"  # the column name for the time to rise
    )

    hr_and_rr_cols = [ # convert any 0s in these columns to NaNs
        "hr_average",
        "hr_min",
        "hr_max",
        "rr_average",
        "rr_min",
        "rr_max",
        ]

    snoring_col: str = "snoring_time (s)"  # snoring column

    zeroes_to_nan = [  # convert the 0s in these columns to NaNs if all values are 0
        [
            "rem_duration (s)",
            "light_duration (s)",
            "deep_duration (s)",
            awake_duration_col,
            snoring_col,
            bed_time_period_col_to_name,
            time_to_bed_col_to_name,
            time_to_rise_col_to_name,
        ],
        [
            "rem_duration (s)",
            "light_duration (s)",
            "deep_duration (s)",
            awake_duration_col,
            bed_time_period_col_to_name,
            time_to_bed_col_to_name,
            time_to_rise_col_to_name,
        ],
        [
            "rem_duration (s)",
            "light_duration (s)",
            "deep_duration (s)",
            awake_duration_col,
        ],
    ]
    
    # columns to use as X
    x_cols = [
        bed_time_period_col_to_name,
        time_to_bed_col_to_name,
        time_to_rise_col_to_name,
        time_in_bed_col_to_name,
        time_out_of_bed_col_to_name,
        "rem_duration (s)",
        "light_duration (s)",
        "deep_duration (s)",
        awake_duration_col,
        snoring_col,
        "hr_average",
        "hr_min",
        "hr_max",
        "rr_average",
        "rr_min",
        "rr_max",
    ]

    # functions to aggregate the columns
    # when calculating daily values
    agg_dict = {
        gender_col: "first",
        age_col: "first",
        bed_time_period_col_to_name: "sum",
        time_to_bed_col_to_name: "max",
        time_to_rise_col_to_name: "min",
        time_in_bed_col_to_name: "sum",
        time_out_of_bed_col_to_name: "sum",
        "rem_duration (s)": "sum",
        "light_duration (s)": "sum",
        "deep_duration (s)": "sum",
        awake_duration_col: "sum",
        snoring_col: "sum",
        "hr_average": "mean",
        "hr_min": "mean",
        "hr_max": "mean",
        "rr_average": "mean",
        "rr_min": "mean",
        "rr_max": "mean",
    }

    day_col_to_name: str = "date"  # the column name for the day

    age_bins = [-np.inf, 50, 60, 70, 80, 90, np.inf]  # age bins using intervals [.,.)
    age_bin_col_to_name: str = "age_bin"  # the column name for the age bin
    
    impute_window: int = 7  # imputation window
    impute_min_periods: int = 5  # min periods for imputation
    impute_center: bool = False  # whether to center the imputation window or not
    
    n_consecutive_days: int = 30  # number of consecutive days to use for training
    n_shift: int = 10 # number of days to shift rolling window by

    minder_demographics_id_col: str = "id"
    minder_demographics_birth_year_col: str = "birth_date"
    minder_demographics_gender_col: str = "gender"

    minder_id_col: str = "patient_id"  # id column for minder data
    minder_date_col: str = "start_date"  # date column for minder data
    minder_date_end_col: str = "end_date"  # date column for minder data
    minder_sleep_state_col: str = "state"  # sleep state column for minder data
    
    # how to aggregate the columns
    # in minder when calculating per period
    # values. These features will then be aggregated
    # daily using the agg_dict above
    minder_agg_func: t.Callable = lambda df: pd.Series(
        {
            # sum weighted by duration
            "snoring": df["snoring"].multiply(df["duration"]).sum(),
            # average weighted by duration
            "hr_average": (
                df["heart_rate"]
                .multiply(df["duration"])
                .divide(df["duration"].sum())
                .sum()
                .round(0)
            ),
            "hr_min": df["heart_rate"].min(),
            "hr_max": df["heart_rate"].max(),
            # average weighted by duration
            "rr_average": (
                df["respiratory_rate"]
                .multiply(df["duration"])
                .divide(df["duration"].sum())
                .sum()
                .round(0)
            ),
            "rr_min": df["respiratory_rate"].min(),
            "rr_max": df["respiratory_rate"].max(),
        }
    )
    minder_columns_matched = {
        minder_id_col: id_col,
        minder_date_col: date_col,
        minder_date_end_col: date_end_col,
        "AWAKE": awake_duration_col,
        "DEEP": "deep_duration (s)",
        "LIGHT": "light_duration (s)",
        "REM": "rem_duration (s)",
        "snoring": snoring_col,
        "hr_average": "hr_average",
        "hr_min": "hr_min",
        "hr_max": "hr_max",
        "rr_average": "rr_average",
        "rr_min": "rr_min",
        "rr_max": "rr_max",
    }

    resilient_demographics_id_col: str = "ID"
    resilient_demographics_birth_year_col: str = "Date of Birth"
    resilient_demographics_gender_col: str = "Sex"

    resilient_id_col: str = "User ID"  # id column for resilient data
    resilient_date_col: str = "startdate"  # date column for resilient data
    resilient_date_end_col: str = "enddate"  # date column for resilient data
    resilient_sleep_state_col: str = "state"  # sleep state column for resilient data

    resilient_columns_matched = {
        resilient_id_col: id_col,
        resilient_date_col: date_col,
        resilient_date_end_col: date_end_col,
        "Awake (s)": awake_duration_col,
        "Deep Sleep (s)": "deep_duration (s)",
        "Light Sleep (s)": "light_duration (s)",
        "REM Sleep (s)": "rem_duration (s)",
        "Snoring": snoring_col,
        "Avg Heart Rate": "hr_average",
        "Min Heart Rate": "hr_min",
        "Max Heart Rate": "hr_max",
        "Avg Respiration Rate": "rr_average",
        "Min Respiration Rate": "rr_min",
        "Max Respiration Rate": "rr_max",
    }
    
    data_name: str = "withings"  # which dataset to use
    #reset_train_val_test_split: bool = False  # whether to reset the train-val-test split or not
    reset_train_test_split: bool = False  # whether to reset the train-test split or not

    #n_jobs: int = 1  # number of jobs to run in parallel
    #results_dir: str = os.path.join("results")  # data directory

    #label_col: str = "label"  # label column
    #source_col: str = "source"  # source column

    load_preprocessed_data: bool = (
        True  # whether to load preprocessed data or to recalculate the processed data
    )
    withings_data_saved_file_name: str = (
        "withings_data_preprocessed.parquet"  # withings data file name
    )
    minder_data_saved_file_name: str = (
        "minder_data_preprocessed.parquet"  # minder data file name
    )
    resilient_data_saved_file_name: str = (
        "resilient_data_preprocessed.parquet"  # resilient data file name
    )
