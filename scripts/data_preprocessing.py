import numpy as np
import pandas as pd
from scipy import stats
import pytz
import typing as t
import os
import sklearn.model_selection as skms


def load_withings_data(args) -> pd.DataFrame:
    """
    Specific function for loading the withings sleep data.

    This will load the files from the data directory.

    Files loaded:
    - withings_sleep_dataset.csv
    - withings_sleep_dataset_older_users.csv
    - withings_users_gender.csv

    They are returned in the order above.
    """
    withings_sleep_dataset = pd.read_csv(
        os.path.join(args.data_dir, "withings_sleep_dataset.csv"),
        index_col=0,
    )
    withings_sleep_dataset_older_users = pd.read_csv(
        os.path.join(args.data_dir, "withings_sleep_dataset_older_users.csv"),
        index_col=0,
    )
    withings_users_gender = pd.read_csv(
        os.path.join(args.data_dir, "withings_users_gender.csv"),
        sep=";",
    )

    return withings_sleep_dataset, withings_sleep_dataset_older_users, withings_users_gender


def prepare_withings_data_for_processing(
    args,
    withings_sleep_dataset: pd.DataFrame,
    withings_sleep_dataset_older_users: pd.DataFrame,
    withings_users_gender: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function simply merges the different sleep datasets from
    withings.

    Arguments
    ---------

    - args: dataclass:
        The arguments for the model.
    - withings_sleep_dataset: pd.DataFrame
    - withings_sleep_dataset_older_users: pd.DataFrame
    - withings_users_gender: pd.DataFrame


    Returns
    -------
    out: pd.DataFrame:
        The merged sleep data, ready for processing.
    """
    data = (
        pd.concat(
            [withings_sleep_dataset, withings_sleep_dataset_older_users]
        ).drop_duplicates()  # drops all data with the same values
        # remove conflicting days and ids
        .drop_duplicates(subset=[args.id_col, args.date_col], keep=False)
        # get the gender info from the other dataset, and only include the ids that appear in the concatenated dataset AND in this csv 
        .merge(
            withings_users_gender[[args.id_col, args.gender_col]],
            on=args.id_col,
            how="inner",
        )
        
    )
    # get only the users whose age is 50 or greater
    data = data[data[args.age_col] >= 50]
    return data


def time_to_angles(
    time,
    total_seconds_in_day: int = 24 * 60**2,
) -> float:    
    """
    Converts times of day to angles in degrees.

    Arguments
    ---------
    time:
        The time of day to convert to seconds to be converted to angles in degrees.

    total_seconds_in_day: int:
        The total number of seconds in a day.
        Defaults to :code:`24 * 60**2`.

    Returns
    -------
    out: float:
        The angle in degrees.
    """
    seconds = time.hour * 60**2 + time.minute * 60 + time.second
    fraction = seconds / total_seconds_in_day
    return fraction * 360

def local_time_to_angle(
    row,
    datetime_col,
    timezone_col):
    """

    Arguments
    ---------
    row:
        A single row of the DataFrame.

    datetime_col: int:
        The time of day to convert to seconds to be converted to angles in degrees.

    timezone_col:
        The timezone (country) by which to convert UTC time to local time. 

    Returns
    -------
    out: float:
        Converts the local time to angle in degrees.
    """
    local_tz = pytz.timezone(row[timezone_col])
    local_time = row[datetime_col].tz_convert(local_tz).time()
    return time_to_angles(local_time)

def values_all_to_nan(
    df: pd.DataFrame, subset: t.List[str], value: t.Any = 0
) -> pd.DataFrame:
    """
    Convert all values in the given :code:`subset` to
    NaNs if, across the row, they are all equal to :code:`value`.

    Arguments
    ---------
    df : pd.DataFrame:
        Dataframe to process.

    subset : list of str:
        List of columns to process.

    value : Any:
        Value to check for.


    Returns
    -------
    out: pd.DataFrame:
        Dataframe with values converted to NaNs.
    """
    df_out = df.copy()

    # check if all values are equal to value
    df_out[subset] = df_out[subset].mask(df_out[subset].eq(value).all(axis=1))
    return df_out


def values_all_to_nan_subsets(
    df: pd.DataFrame, subsets: t.List[t.List[str]], value: t.Any = 0
) -> pd.DataFrame:
    """
    Convert all values in the given :code:`subsets` to
    NaNs if, across the row for each subset, they are all equal to :code:`value`.
    This iterates over the function :code:`values_all_to_nan` for each subset
    in :code:`subsets`.

    Arguments
    ---------
    df : pd.DataFrame:
        Dataframe to process.

    subsets : list of list of str:
        List of list of columns to process.

    value : Any:
        Value to check for.


    Returns
    -------
    out: pd.DataFrame:
        Dataframe with values converted to NaNs.
    """
    df_out = df.copy()
    for subset in subsets:
        df_out = values_all_to_nan(df_out, subset, value)
    return df_out


def subset_apply(df, subset, *args, **kwargs):
    """
    Apply a function to a subset of a dataframe, then
    return the whole dataframe.
    """
    df_out = df.copy()
    df_out[subset] = df_out[subset].apply(*args, **kwargs)
    return df_out


def subset_pipe(df, subset, *args, **kwargs):
    """
    Pipe a function to a subset of a dataframe, then
    return the whole dataframe.
    """
    df_out = df.copy()
    df_out[subset] = df_out[subset].pipe(*args, **kwargs)
    return df_out


def rolling_impute(
    df: pd.DataFrame,
    window: int,
    on: str,
    min_periods: int = 1,
    subset: t.List[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Impute the missing values in a dataframe by
    taking the rolling mean of the values.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'id': [1] * 10,
    ...     'date': pd.date_range('2020-01-01', '2020-01-10'),
    ...     'value': range(1, 11),
    ...     'value2': range(1, 11),
    ... })
    >>> df.loc[[2, 5, 7, 8, 9], 'value'] = np.nan
    >>> rolling_impute(df, window=3, on='date', min_periods=1, subset=['value'])


    Arguments
    ---------
    - df : pd.DataFrame:
        The dataframe to impute.

    - window : int:
        The window size to use for the rolling mean.

    - on : str:
        The column to use for the rolling mean.

    - min_periods : int:
        The minimum number of periods to use for the rolling mean.
        Defaults to :code:`1`.

    - subset : list of str:
        The columns to impute. :code:`None` imputes all columns.
        Defaults to :code:`None`.

    - **kwargs:
        Keyword arguments to pass to :code:`pd.DataFrame.rolling`.

    Returns
    -------
    out: pd.DataFrame:
        The imputed dataframe.
    """
    if subset is None:
        subset = df.columns

    df_out = df.set_index(on).sort_index()

    rolling_values = (
        df[subset + [on]]
        .set_index(on)
        .sort_index()
        .rolling(window, min_periods=min_periods, **kwargs)
        .mean()
    )

    df_out[subset] = (
        df.reset_index().set_index(on).combine_first(rolling_values)[subset]
    )

    return df_out.reset_index()


def clean_data(
    args,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to process the files loaded using the :code:`load_withings_data` function.

    This function will:
    - get the date of sleep by subtracting args.sleep_offset hours from the time and date of the sleep start
    - calculate the time between the start of the sleep and the end of the sleep as bed time period
    - calculate the time that the user went to bed as an angle
    - calculate the time that the user woke up as an angle
    - remove the rows in which the sum of the sleep states is more than the bed time period
    - remove those rows where the snoring time is more than time in bed
    - remove those rows where the bed time period is more than the max bed time period
    - calculate the total time in bed from the sleep states
    - calculate the total time out of bed during the sleep period
    - replace any 0s in some columns with NaNs 
    - replace the 0s in some columns with NaNs if all values are 0
    - sort the values by id and date
    - keep the columns we want


    Arguments
    ---------

    - args: dataclass:
        The arguments for the model.
    
    - data: pd.DataFrame:
        The data.
    """
    data = (
        data.assign(
            **{
                args.date_col: lambda df: pd.to_datetime(df[args.date_col]),
                args.date_end_col: lambda df: pd.to_datetime(df[args.date_end_col]),
            }
        )
        # we get the date of sleep by subtracting args.sleep_offset hours
        # from the time and date of the sleep start
        .assign(
            **{
                args.day_col_to_name: lambda df: pd.to_datetime(
                    (df[args.date_col] - pd.Timedelta(hours=args.sleep_offset)).dt.date
                )
            },
        )
        .assign(
            **{
                # calculate the time between the start of the sleep and end of sleep
                args.bed_time_period_col_to_name: lambda df: (
                    (df[args.date_end_col] - df[args.date_col]).dt.seconds
                ),
                # calculate the time that the user went to bed as an angle
                args.time_to_bed_col_to_name: lambda df: df.apply(
                lambda row: local_time_to_angle(row, args.date_col, args.timezone_col), axis=1),
                # calculate the time that the user got out of bed as an angle
                args.time_to_rise_col_to_name: lambda df: df.apply(
                lambda row: local_time_to_angle(row, args.date_end_col, args.timezone_col), axis=1)
            }
        )
        # remove the rows in which the sum of the sleep states is more
        # than the bed time period
        .loc[
            lambda df: df[args.sleep_states].sum(axis=1)
            < df[args.bed_time_period_col_to_name]
        ]
        # remove those rows where the snoring time is more than time in bed
        .loc[
            lambda df: df[args.snoring_col] < df[args.bed_time_period_col_to_name]
        ]
        # remove those rows where the bed time period is
        # more than args.max_bed_time_period
        .loc[
            lambda df: df[args.bed_time_period_col_to_name] < args.max_bed_time_period]
        .assign(
            **{
                # calculate the total time in bed from the sleep states
                args.time_in_bed_col_to_name: lambda df: (
                    df[args.sleep_states].sum(axis=1)
                ),
            }
        )
        .assign(
            **{
                # calculate the total time out of bed during the sleep
                # by subtracting the time in bed from the total time
                # between the start of the sleep and the end of the sleep
                args.time_out_of_bed_col_to_name: lambda df: (
                    df[args.bed_time_period_col_to_name]
                    - df[args.time_in_bed_col_to_name])
            }
        )
        # replace any 0s in these columns with NaNs
        .replace({col: {'0':np.nan, 0:np.nan} for col in args.hr_and_rr_cols})
        # replace the 0s in the following columns with NaNs
        # if all values are 0
        .pipe(values_all_to_nan_subsets, subsets=args.zeroes_to_nan)
        # sort the values by id and date
        .sort_values([args.id_col, args.date_col])
        # keep the columns we want
        [
            [
                args.id_col,
                args.day_col_to_name,
                args.gender_col,
                args.age_col,
            ]
            + args.x_cols
        ]
    )

    return data


def agg_and_fill_dates(
    args,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to continue to process the files loaded using the :code:`load_withings_data` function.

    This function will:
    - group by id and date of sleep, and aggregate the numeric columns


    Arguments
    ---------

    - args: dataclass:
        The arguments for the model.

    - data: pd.DataFrame:
        The data.
    """
    data = (
        data
        # group by id and date of sleep, and aggregate the numeric columns
        .groupby([args.id_col, args.day_col_to_name])
        .agg(args.agg_dict)
        .reset_index()
        # group by id, and fill in the missing dates.
        # We back fill the id, gender and age since
        # we assume they are the same as the day before
        .groupby([args.id_col], group_keys=False)
        .apply(
            lambda df: (
                df.set_index(args.day_col_to_name)
                .reindex(
                    pd.date_range(
                        df[args.day_col_to_name].min(), df[args.day_col_to_name].max()
                    )
                )
                .assign(
                    **{
                        args.id_col: lambda s: s[args.id_col].bfill(),
                        args.gender_col: lambda s: s[args.gender_col].bfill(),
                        args.age_col: lambda s: s[args.age_col].bfill(),
                    }
                )
                .rename_axis(args.day_col_to_name)
                .reset_index()
                .set_index([args.id_col, args.day_col_to_name])
            )
        )
        .reset_index()
        .assign(
            **{
                args.day_col_to_name: lambda df: pd.to_datetime(
                    df[args.day_col_to_name]
                )
            }
        )
    )

    return data


def impute_missing_values(
    args,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to continue to process the files loaded using the :code:`load_withings_data` function.

    This function will:
    - fill in missing dates using the rolling mean
    - remove the remaining missing values


    Arguments
    ---------

    - args: dataclass:
        The arguments for the model.

    - data: pd.DataFrame:
        The data.
    """
    data = (
        data.sort_values([args.id_col, args.day_col_to_name])
        # fill in the missing values with a rolling mean. The values
        # that do not have at least impute_min_periods days of data in
        # a window of impute_window days will be left as NaN
        .groupby([args.id_col], group_keys=False).apply(
            lambda df: rolling_impute(
                df=df,
                window=args.impute_window,
                on=args.day_col_to_name,
                min_periods=args.impute_min_periods,
                # the rolling window will look like
                # [current-window, current)
                # this means it doesn't attempt to impute
                # the current day with the values of the current day
                # which would be a problem if the current day
                # was missing
                closed="left",
                center=args.impute_center,
                # fill NaN on the numeric columns
                subset=list(
                    df.drop(
                        columns=[
                            args.id_col,
                            args.day_col_to_name,
                            args.gender_col,
                            args.age_col,
                        ]
                    ).columns
                ),
            )
        )
        # remove the missing values that were not
        # imputed with the mean of the window
        .dropna()
        # ensure args.day_col_to_name is a datetime
        .assign(
            **{
                args.day_col_to_name: lambda df: pd.to_datetime(
                    df[args.day_col_to_name]
                )
            }
        )
        # drop duplicate days and id
        .drop_duplicates(subset=[args.id_col, args.day_col_to_name])
    )

    return data


def bin_age(
    args,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to continue to process the files loaded using the :code:`load_withings_data` function.

    This function will:
    - add a binned age column


    Arguments
    ---------

    - args: dataclass:
        The arguments for the model.

    - data: pd.DataFrame:
        The data.
    """
    data = data.assign(
        **{
            args.age_bin_col_to_name: lambda df: (
                df[[args.age_col]]
                .apply(
                    lambda x: pd.cut(
                        x,
                        bins=np.sort(args.age_bins),
                        right=False,
                        include_lowest=False,
                        ordered=True,
                    )
                )[args.age_col]
                .apply(lambda x: f"{x}")
            )
        }
    )

    return data


def filter_not_enough_consecutive_days(args, data: pd.DataFrame) -> pd.DataFrame:
    """
    Function to continue to process the files loaded using the :code:`load_withings_data` function.

    This function will:
    - remove the rows that are not part of a sequence of at least args.n_consecutive_days


    Arguments
    ---------

    - args: dataclass:
        The arguments for the model.

    - data: pd.DataFrame:
        The data.
    """
    data = (
        data
        # groupby ids to calculate previous day of data
        .sort_values([args.id_col, args.day_col_to_name])
        .groupby([args.id_col], group_keys=False)
        .apply(
            lambda df: (
                df.assign(prev_date=df[args.day_col_to_name].shift(1)).set_index(
                    [args.id_col]
                )
            )
        )
        # calculate the number of days between the
        # previous date and the current date
        .assign(
            days_to_prev_date=lambda df: (
                df[args.day_col_to_name] - df["prev_date"]
            ).dt.days
        )
        # the first days_to_prev_date for each ID will be
        # NaN since there is no eariler date
        # in that group. This means that they will
        # form the first of a new consecutive_days_group
        .assign(
            consecutive_days_group=lambda df: (df["days_to_prev_date"] != 1.0).cumsum()
        )
        .reset_index()
        # filter out the groups that do not have at least
        # args.n_consecutive_days of data. This is done by
        # counting the number of consecutive days in each group,
        # and then filtering out the groups that do not have
        # at least args.n_consecutive_days of data
        .pipe(
            lambda df: (
                df[
                    df["consecutive_days_group"].isin(
                        (
                            (
                                df["consecutive_days_group"].value_counts()
                                >= args.n_consecutive_days
                            )
                            .to_frame("n_cons_days")
                            .query("n_cons_days == True")
                            .index.to_list()
                        )
                    )
                ]
            )
        )
        .drop(columns=["prev_date", "days_to_prev_date", "consecutive_days_group"])
    )

    return data


def process_data(
    args,
    data: pd.DataFrame,
    filter_n_consecutive_days: bool = True,
) -> pd.DataFrame:
    """
    Function to process the files loaded using the :code:`load_withings_data` function.

    This function will:
    - get the date of sleep by subtracting args.sleep_offset hours from the time and date of the sleep start
    - calculate the time between the start of the sleep and the end of the sleep as bed time period
    - calculate the time that the user went to bed as an angle
    - calculate the time that the user woke up as an angle
    - remove the rows in which the sum of the sleep states is more than the bed time period
    - remove those rows where the snoring time is more than time in bed
    - remove those rows where the bed time period is more than the max bed time period
    - calculate the total time in bed from the sleep states
    - calculate the total time out of bed during the sleep period
    - replace any 0s in some columns with NaNs 
    - replace the 0s in some columns with NaNs if all values are 0
    - sort the values by id and date
    - keep the columns we want
    - group by id and date of sleep, and aggregate the numeric columns
    - fill in missing dates using the rolling mean
    - remove the remaining missing values
    - add a binned age column
    - keep the columns we want
    - remove the rows that are not part of a sequence of at least args.n_consecutive_days

    Arguments
    ---------

    - args: dataclass:
        The arguments for the model.

    - data: pd.DataFrame:
        The data to process.
    
    - filter_n_consecutive_days: bool:
        Whether to filter out the rows that are not part of a 
        sequence of at least args.n_consecutive_days.
        Defaults to :code:`True`.
    """
    data = clean_data(args, data)
    data = agg_and_fill_dates(args, data)
    data = impute_missing_values(args, data)
    data = bin_age(args, data)
    if filter_n_consecutive_days:
        data = filter_not_enough_consecutive_days(args, data)

    data = data[
        [
            args.id_col,
            args.day_col_to_name,
            args.gender_col,
            args.age_col,
            args.age_bin_col_to_name,
        ]
        + args.x_cols
    ]

    return data.reset_index(drop=True)


def load_and_process_withings_data(
    args,
):
    """
    This function will load the withings data and process it.
    If :code:`args.load_preprocessed_data` is :code:`True`,
    then this function will load the processed data from disk.

    Arguments
    ---------

    - args: dataclass:
        The arguments for the data loading and model.
    """
    file_path = os.path.join(args.data_dir, args.withings_data_saved_file_name)
    if args.load_preprocessed_data and os.path.exists(file_path):
        print(f"Loading preprocessed Withings data from {file_path}")
        withings_sleep_mat = pd.read_parquet(
            file_path,
        )
    else:
        print(f"Calculating and writing preprocessed Withings data to {file_path}")
        # load the withings dataset files
        withings_datasets = load_withings_data(args=args)

        # prepare the files for processing
        withings_sleep_mat = prepare_withings_data_for_processing(
            args, *withings_datasets
        )

        # process the files
        withings_sleep_mat = process_data(args, withings_sleep_mat)

        # save the processed file to disk
        withings_sleep_mat.to_parquet(file_path)

    return withings_sleep_mat


def load_minder_data(args) -> pd.DataFrame:
    """
    Specific function for loading the minder sleep data.

    This will load the files from the data directory.

    Files loaded:
    - minder_sleep_mat.csv
    """
    cols = list(
        pd.read_csv(os.path.join(args.data_dir, "minder_sleep_mat.csv"), nrows=1)
    )

    cols_to_drop = [
        "id",
        "device_type",
        "home_id",
        "heart_rate_unit",
        "respiratory_rate_unit",
    ]

    minder_sleep_mat_raw = pd.read_csv(
        os.path.join(args.data_dir, "minder_sleep_mat.csv"),
        usecols=[c for c in cols if c not in cols_to_drop],
        low_memory=False,
    )

    minder_demographics = pd.read_csv(
        os.path.join(args.data_dir, "minder_demographics.csv"),
        usecols=[
            args.minder_demographics_id_col,
            args.minder_demographics_birth_year_col,
            args.minder_demographics_gender_col,
        ],
    )

    return minder_sleep_mat_raw, minder_demographics


def prepare_minder_data_for_processing(
    args,
    minder_sleep_mat: pd.DataFrame,
    minder_demographics: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function simply prepare the minder sleep data for processing
    by formatting it in the same way as the withings data.

    Arguments
    ---------

    - args: dataclass:
        The arguments for the model.

    - minder_sleep_mat: pd.DataFrame:
        The minder sleep data.

    - minder_demographics: pd.DataFrame:
        The minder demographics data.


    Returns
    -------
    out: pd.DataFrame:
        The minder sleep data formatted in the same way as the withings data.
    """
    sleep_mat_periods = (
        minder_sleep_mat
        # ensure that the date columns are in the correct format
        .assign(
            **{
                args.minder_date_col: lambda df: pd.to_datetime(
                    df[args.minder_date_col]
                ).dt.round("1ms"),
                args.minder_date_end_col: lambda df: (
                    pd.to_datetime(df[args.minder_date_end_col]).dt.round("1ms")
                    - pd.Timedelta(seconds=1)
                ),
            }
        )
        # sort the values by id and date
        .sort_values(by=[args.minder_id_col, args.minder_date_col])
        # calculate the sleep_segments.
        # each period has at least one sleep mat measurement each hour
        .groupby(args.minder_id_col, group_keys=False)
        # The following will produce a column that is True if the difference
        # between the current date and the previous date is less than an
        # hour, and False otherwise. It will
        # also be True if the date is the first for an ID
        .apply(
            lambda df: (
                df.assign(
                    **{
                        "sleep_segment": lambda df: (
                            df[args.minder_date_end_col]
                            .subtract(df[args.minder_date_col].shift(1))
                            # gets the values that are not within an hour of
                            # the previous value and the NaT values and labels them as True
                            .le(pd.Timedelta(hours=1))
                            .apply(np.invert)
                        )
                    }
                )
            )
        ).assign(
            **{
                # calculate the sleep_segment number by doing a cumulative sum of the period segments
                "sleep_segment": lambda df: df["sleep_segment"].cumsum(),
                # calculate the duration of each period
                "duration": lambda df: (
                    df[args.minder_date_end_col]
                    .subtract(df[args.minder_date_col])
                    .dt.total_seconds()
                ),
            }
        )
        # rename the snoring column and replace boolean values
        .rename(
            {"snoring": {"true": 1, "false": 0, True: 1, False: 0}},
        )
    )

    # calculate the start and end dates for each period
    sleep_mat_start_end = sleep_mat_periods.groupby(
        [args.minder_id_col, "sleep_segment"], group_keys=False
    ).agg(
        **{
            args.minder_date_col: (args.minder_date_col, "min"),
            args.minder_date_end_col: (args.minder_date_end_col, "max"),
        }
    )

    # calculate the sleep states for each period by summing the durations of sleep
    # period spent in each state
    sleep_mat_states = (
        sleep_mat_periods.groupby(
            [args.minder_id_col, "sleep_segment", args.minder_sleep_state_col],
            group_keys=False,
        )["duration"]
        .sum()
        .unstack()
    )

    # calculate the vitals for each period by calculating a weighted function
    # for each period using the duration of each measurement as the weights
    sleep_mat_vitals = sleep_mat_periods.groupby(
        [args.minder_id_col, "sleep_segment"],
    ).apply(args.minder_agg_func)

    # combine the start and end dates, the sleep states, and the vitals
    # into one dataframe and reset the index
    sleep_mat_processed = (
        pd.concat(
            [
                sleep_mat_start_end,
                sleep_mat_states,
                sleep_mat_vitals,
            ],
            axis=1,
        )
        .reset_index()
        .drop(columns=["sleep_segment"])
        .rename(columns=args.minder_columns_matched)
    )

    sleep_mat_processed_with_demographics = (
        sleep_mat_processed
        # merge the demographics data
        .merge(
            (
                minder_demographics.rename(
                    columns={args.minder_demographics_id_col: args.id_col}
                )
            ),
            on=args.id_col,
            how="left",
        )
        # calculate the age
        .assign(
            **{
                args.age_col: lambda x: (
                    x[args.date_col].dt.year
                    - x[args.minder_demographics_birth_year_col]
                )
            }
        )
        # rename the columns to match withings naming
        .rename(columns={args.minder_demographics_gender_col: args.gender_col})
    )

    sleep_mat_processed_with_demographics['timezone'] = 'Europe/London'
    
    return sleep_mat_processed_with_demographics


def load_and_process_minder_data(
    args,
):
    """
    This function will load the minder data and process it.
    If :code:`args.load_preprocessed_data` is :code:`True`,
    then this function will load the processed data from disk.

    Arguments
    ---------

    - args: dataclass:
        The arguments for the data loading and model.
    """
    file_path = os.path.join(args.data_dir, args.minder_data_saved_file_name)
    if args.load_preprocessed_data and os.path.exists(file_path):
        print(f"Loading preprocessed Minder data from {file_path}")
        minder_sleep_mat = pd.read_parquet(
            file_path,
        )
    else:
        print(f"Calculating and writing preprocessed Minder data to {file_path}")
        # load the minder dataset files
        minder_datasets = load_minder_data(args=args)

        # prepare the files for processing
        minder_sleep_mat = prepare_minder_data_for_processing(args, *minder_datasets)

        # process the files
        minder_sleep_mat = process_data(args, minder_sleep_mat)

        # save the processed file to disk
        minder_sleep_mat.to_parquet(file_path)

    return minder_sleep_mat


def load_resilient_data(args) -> pd.DataFrame:
    """
    Specific function for loading the resilient sleep data.

    This will load the files from the data directory.

    Files loaded:
    - resilient_sleep_mat.csv
    """
    cols = list(
        pd.read_csv(os.path.join(args.data_dir, "resilient_sleep_mat.csv"), nrows=1)
    )

    resilient_sleep_mat_raw = pd.read_csv(
        os.path.join(args.data_dir, "resilient_sleep_mat.csv"),
        low_memory=False,
    )

    resilient_demographics = pd.read_csv(
        os.path.join(args.data_dir, "resilient_demographics.csv"),
        usecols=[
            args.resilient_demographics_id_col,
            args.resilient_demographics_birth_year_col,
            args.resilient_demographics_gender_col,
        ],
    )
    resilient_demographics[args.resilient_demographics_id_col] = resilient_demographics[args.resilient_demographics_id_col].astype(str)
    resilient_demographics[args.resilient_demographics_birth_year_col] = pd.to_datetime(resilient_demographics[args.resilient_demographics_birth_year_col])

    return resilient_sleep_mat_raw, resilient_demographics


def prepare_resilient_data_for_processing(
    args,
    resilient_sleep_mat: pd.DataFrame,
    resilient_demographics: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function simply prepare the resilient sleep data for processing
    by formatting it in the same way as the withings data.

    Arguments
    ---------

    - args: dataclass:
        The arguments for the model.

    - resilient_sleep_mat: pd.DataFrame:
        The resilient sleep data.

    - resilient_demographics: pd.DataFrame:
        The resilient demographics data.


    Returns
    -------
    out: pd.DataFrame:
        The resilient sleep data formatted in the same way as the withings data.
    """
    sleep_mat_processed = (
        resilient_sleep_mat
        # ensure that the date columns are in the correct format
        .assign(
            **{
                args.resilient_date_col: lambda df: pd.to_datetime(
                    df[args.resilient_date_col]
                ).dt.round("1ms"),
                args.resilient_date_end_col: lambda df: (
                    pd.to_datetime(df[args.resilient_date_end_col]).dt.round("1ms")
                    - pd.Timedelta(seconds=1)
                ),
            }
        )
        # sort the values by id and date
        .sort_values(by=[args.resilient_id_col, args.resilient_date_col])
        .rename(columns=args.resilient_columns_matched)
    )
    
    sleep_mat_processed_with_demographics = (
        sleep_mat_processed
        # merge the demographics data
        .merge(
            (
                resilient_demographics.rename(
                    columns={args.resilient_demographics_id_col: args.id_col}
                )
            ),
            on=args.id_col,
            how="left",
        )
        # calculate the age
        .assign(
            **{
                args.age_col: lambda x: (
                    x[args.date_col].dt.year
                    - x[args.resilient_demographics_birth_year_col].dt.year
                )
            }
        )
        # rename the columns to match withings naming
        .rename(columns={args.resilient_demographics_gender_col: args.gender_col})
    )
    
    sleep_mat_processed_with_demographics['timezone'] = 'Europe/London'
    
    return sleep_mat_processed_with_demographics


def load_and_process_resilient_data(
    args,
):
    """
    This function will load the resilient data and process it.
    If :code:`args.load_preprocessed_data` is :code:`True`,
    then this function will load the processed data from disk.

    Arguments
    ---------

    - args: dataclass:
        The arguments for the data loading and model.
    """
    file_path = os.path.join(args.data_dir, args.resilient_data_saved_file_name)
    if args.load_preprocessed_data and os.path.exists(file_path):
        print(f"Loading preprocessed Resilient data from {file_path}")
        resilient_sleep_mat = pd.read_parquet(
            file_path,
        )
    else:
        print(f"Calculating and writing preprocessed Resilient data to {file_path}")
        # load the resilient dataset files
        resilient_datasets = load_resilient_data(args=args)

        # prepare the files for processing
        resilient_sleep_mat = prepare_resilient_data_for_processing(args, *resilient_datasets)

        # process the files
        resilient_sleep_mat = process_data(args, resilient_sleep_mat)

        # save the processed file to disk
        resilient_sleep_mat.to_parquet(file_path)

    return resilient_sleep_mat


def load_train_test_idx(
    args, dataset: t.Literal["withings"], data=None, seed=None
) -> t.Tuple[np.ndarray, np.ndarray]:
    """
    This function will load the train and test idx from the
    data directory. If args.reset_train_test_split is :code:`True`,
    it will create a new train and test split and save it to the
    data directory. If this is the case, you will need to provide
    the data and seed arguments.

    Arguments
    ---------
    - args: dataclass:
        The arguments for the model.

    - dataset: t.Literal["withings"]:
        The dataset to load the train and test idx for.

    - data: pd.DataFrame:
        The data to use to create the train and test idx.
        Only required if args.reset_train_test_split is :code:`True`.
        Defaults to :code:`None`.

    - seed: int:
        The seed to use to create the train and test idx.
        Only required if args.reset_train_test_split is :code:`True`.
        Defaults to :code:`None`.

    """

    if dataset not in ["withings"]:
        raise ValueError("dataset must be withings")

    # where the train and test idx are stored
    file_path = os.path.join(args.data_dir, f"{dataset}_train_test_idx_{args.n_consecutive_days}.npz")

    if args.reset_train_test_split:
        if data is None:
            raise ValueError(
                "data must be provided if args.reset_train_test_split is True"
            )
        binned_target = data[args.age_bin_col_to_name]

        unique_ids = np.unique(data[args.id_col])

        binned_target_for_unique_ids = [binned_target[data[args.id_col] == unique_id].mode()[0] for unique_id in unique_ids]

        splitter = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)

        # Perform stratified split based on unique ids
        #for train_idx, test_idx in splitter.split(unique_ids, [np.unique(target) for target in binned_target_for_unique_ids]):
        for train_idx, test_idx in splitter.split(unique_ids, binned_target_for_unique_ids):
            id_train_unique = unique_ids[train_idx].tolist()
            id_test_unique = unique_ids[test_idx].tolist()

        # Now check age group counts per split and rebalance if needed
        def get_age_group_counts(id_list):
            ids_df = pd.DataFrame({
                'id': id_list,
                'age_group': [binned_target[data[args.id_col] == uid].mode()[0] for uid in id_list]
            })
            return ids_df['age_group'].value_counts(), ids_df

        train_counts, train_df = get_age_group_counts(id_train_unique)
        test_counts, test_df = get_age_group_counts(id_test_unique)

        all_groups = set(train_counts.index).union(set(test_counts.index))

        rng = np.random.default_rng(seed)

        for group in all_groups:
            n_train = train_counts.get(group, 0)
            n_test = test_counts.get(group, 0)

            # Shift from train → test
            if n_test < 5 and n_train > 5:
                needed = 5 - n_test
                candidates = train_df[train_df['age_group'] == group]['id'].tolist()
                to_move = rng.choice(candidates, size=needed, replace=False)
                for uid in to_move:
                    id_train_unique.remove(uid)
                    id_test_unique.append(uid)

            # Shift from test → train
            elif n_train < 5 and n_test > 5:
                needed = 5 - n_train
                candidates = test_df[test_df['age_group'] == group]['id'].tolist()
                to_move = rng.choice(candidates, size=needed, replace=False)
                for uid in to_move:
                    id_test_unique.remove(uid)
                    id_train_unique.append(uid)
            
        train_idx = np.where(data[args.id_col].isin(id_train_unique))[0]
        test_idx = np.where(data[args.id_col].isin(id_test_unique))[0]

        np.savez(
            file_path,
            train_idx=train_idx,
            test_idx=test_idx,
        )

    else:
        if not os.path.exists(file_path):
            raise ValueError(
                "Please ensure there is a file titled train_test_idx.npz"
                " at the data_dir specified. If not, please set"
                " reset_train_test_split to True and run this function"
                " again with the data and seed arguments given."
            )
        train_test_idx = np.load(file_path)
        train_idx = train_test_idx["train_idx"]
        test_idx = train_test_idx["test_idx"]

    return train_idx, test_idx


def make_input_roll(
    data: np.ndarray,
    sequence_length: int,
    shift: int,
) -> np.ndarray:
    """
    This function will produce an array that is a rolled
    version of the original data sequence. The original
    sequence must be 2D.

    Examples
    ---------

    .. code-block::

        >>> make_input_roll(np.array([[1, 2],[3, 4],[5, 6]]), sequence_length=2, shift=1)
        array([[[1,2],
                [3,4]],

                [[3,4],
                [5,6]]]

    Arguments
    ---------

    - data: numpy.ndarray:
        This is the data that you want transformed.
        Please use the shape (n_datapoints, n_features).

    - sequence_length: int:
        This is an integer that contains the length
        of each of the returned sequences.

    - shift: int:
        This is the number of days by which the window is shifted.


    Returns
    ---------

    - output: ndarray:
        This is an array with the rolled data.
    """
    assert type(sequence_length) == int, "Please ensure that sequence_length is an integer"
    assert type(shift) == int, "Please ensure that shift is an integer"

    if data.shape[0] < sequence_length:
        raise TypeError(
            "Please ensure that the input can be rolled "
            "by the specified sequence_length. Input size was "
            f"{data.shape} and the sequence_length was {sequence_length}."
        )

    # Calculate the number of windows
    num_windows = (data.shape[0] - sequence_length) // shift + 1
    if num_windows <= 0:
        raise ValueError("Cannot generate any windows with the given sequence_length and shift")

    # Prepare an array to hold the rolling windows
    output = []

    for start_idx in range(0, num_windows * shift, shift):
        # Ensure we don't exceed the bounds of the data
        if start_idx + sequence_length <= data.shape[0]:
            # Append the window slice (data[start_idx:start_idx+sequence_length])
            output.append(data[start_idx:start_idx + sequence_length])

    # Convert the output list to a numpy array
    return np.array(output)


def data_to_arrays(
    args, data: pd.DataFrame
) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function turns the data returned from the :code:`process_data` 
    function into arrays that can be used by the model. 
    The arrays will be rolled into sliding windows.

    This function will:
        - remove data that do not have at least \
            :code:`args.n_consecutive_days` of data
        - roll the data into sliding windows of \
            size :code:`args.n_consecutive_days`

    This function will return:
        - :code:`x`: the input data
        - :code:`dates`: the dates of the data
        - :code:`ids`: the ids of the data
        - :code:`gender`: the genders of the data
        - :code:`age`: the ages of the data

    In this order.
    """
    data_arrays_df = (
        data
        # groupby ids to calculate previous day of data
        .groupby([args.id_col], group_keys=False)
        .apply(
            lambda df: (
                df.assign(prev_date=df[args.day_col_to_name].shift(1)).set_index(
                    [args.id_col]
                )
            )
        )
        # calculate the number of days between the
        # previous date and the current date
        .assign(
            days_to_prev_date=lambda df: (
                df[args.day_col_to_name] - df["prev_date"]
            ).dt.days
        )
        # the first days_to_prev_date for each ID will be
        # NaN since there is no eariler date
        # in that group. This means that they will
        # form the first of a new consecutive_days_group
        .assign(
            consecutive_days_group=lambda df: (df["days_to_prev_date"] != 1.0).cumsum()
        )
        .reset_index()
        # group by consecutive_days_group and create the input roll
        .groupby(["consecutive_days_group"])  # we will roll over each group
        .apply(
            lambda df: [
                make_input_roll(df[args.x_cols].values, args.n_consecutive_days, args.n_shift),
                make_input_roll(
                    df[args.day_col_to_name].values, args.n_consecutive_days, args.n_shift
                ),
                df[args.id_col].values[args.n_consecutive_days - 1 :][
                    : (len(df) - args.n_consecutive_days) // args.n_shift + 1
                ],
                df[args.gender_col].values[args.n_consecutive_days - 1 :][
                    : (len(df) - args.n_consecutive_days) // args.n_shift + 1
                ],
                df[args.age_bin_col_to_name].values[args.n_consecutive_days - 1 :][
                    : (len(df) - args.n_consecutive_days) // args.n_shift + 1
                ],
                df[args.age_col].values[args.n_consecutive_days - 1 :][
                    : (len(df) - args.n_consecutive_days) // args.n_shift + 1
                ],
            ]
            if len(df) >= args.n_consecutive_days else []
        )
        .apply(pd.Series)
        .dropna()
    )

    X_data, dates_data, id_data, gender_data, age_data, raw_age_data = [
        np.concatenate(v.values, axis=0) for n, v in data_arrays_df.items()
    ]
    X_data = X_data.transpose(0, 2, 1)

    return X_data, dates_data, id_data, gender_data, age_data, raw_age_data


def calculate_statistics(
    data):
     """
    Converts 2D NumPy arrays with n_features, n_values into 1D vectors for each sample in
    the order as shown below:

    [mean_feat_1, mean_feat_2, ..., mean_feat_F,
    median_feat_1, median_feat_2, ..., median_feat_F,
    min_feat_1, min_feat_2, ..., min_feat_F,
    max_feat_1, max_feat_2, ..., max_feat_F,
    std_feat_1, std_feat_2, ..., std_feat_F,
    iqr_feat_1, iqr_feat_2, ..., iqr_feat_F]

    where F is the number of features.

    Each vector is then appended to a 2D NumPy array as a single row.

    Returns
    -------
    out: pd.DataFrame:
        A 2D NumPy array with shape (N, 6 x F) where N is the number of data samples and
        F is the number of features.
    """
    # Calculate statistics for each feature across the n timesteps
    results = []

    # Calculate each statistic
    for sample in data:
        mean = np.mean(sample, axis=1)
        median = np.median(sample, axis=1)
        min_val = np.min(sample, axis=1)
        max_val = np.max(sample, axis=1)
        std = np.std(sample, axis=1)
        iqr = stats.iqr(sample, axis=1)

        stats_vector = np.concatenate([mean, median, min_val, max_val, std, iqr], axis=0)
        results.append(stats_vector)

    final_result = np.array(results)
    return final_result
