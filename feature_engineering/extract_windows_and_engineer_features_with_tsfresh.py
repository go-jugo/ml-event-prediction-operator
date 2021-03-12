import datetime
import pandas as pd
import dask.dataframe as dd
from sklearn.utils import shuffle
import dask
import glob
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from ..monitoring.time_it import timing
from math import ceil
import copy
import random
from ..logger import get_logger

logger = get_logger(__name__.split(".", 1)[-1])


def calculate_windo(df, window_start_date, window_end_date, element, minimal_features, window_length, errorcode_col,
                     extract_negative_examples=False):

    df_window = df.loc[window_start_date:window_end_date].copy()

    global_timestamp = element[0]
    errorcode_value = element[1]

    df_window['id'] = 1

    if df_window.empty:
        logger.debug('Empty')
        if extract_negative_examples:
            return pd.DataFrame(data={'global_timestamp': [global_timestamp]})
        else:
            return

    if minimal_features:
        extracted_features = extract_features(df_window, column_id='id', n_jobs=0,
                                              default_fc_parameters=MinimalFCParameters())
    else:
        extracted_features = extract_features(df_window, column_id='id', column_sort='global_timestamp', n_jobs=0,
                                              default_fc_parameters=EfficientFCParameters())

    extracted_features['global_timestamp'] = global_timestamp
    extracted_features[errorcode_col] = errorcode_value
    extracted_features['samples_used'] = len(df_window)
    extracted_features['window_start'] = window_start_date
    extracted_features['window_end'] = window_end_date
    extracted_features['window_length'] = window_length

    return extracted_features


def get_processed_timestamp_list(errorcode, window_end, window_length, negative_examples=True, count=False):

    if negative_examples:
        processed_files = glob.glob('../data/Extracted_Examples_ts_fresh/errorcode_' + str(errorcode) +
                                    '_PW_' + str(window_end) + '_RW_' + str(window_length) + '_' + 'neg*.gzip')
    else:
        processed_files = glob.glob('../data/Extracted_Examples_ts_fresh/errorcode_' + str(errorcode) +
                                    '_PW_' + str(window_end) + '_RW_' + str(window_length) + '_' + 'pos*.gzip')

    processed_timestamp_list = []

    if count:
        if negative_examples:
            return len(processed_files) * 500
        else:
            counter = 0
            for file in processed_files:
                df_processed = pd.read_parquet(file)
                counter = counter + len(df_processed)
            return counter
    else:
        for file in processed_files:
            df_processed = pd.read_parquet(file)
            timestamp_list = df_processed['global_timestamp'].to_list()
            processed_timestamp_list.extend(timestamp_list)
        return processed_timestamp_list


def get_clean_errorcode_column_to_process(error_code_series, errorcode_col, errorcode, window_end, window_length, negative_examples=False):

    if negative_examples:
        target_errorcode_timestamp_list = list(error_code_series[error_code_series[errorcode_col] == errorcode].index.compute())
        exclude_windows = []
        for element in target_errorcode_timestamp_list:
            exclude_windows.append([element + datetime.timedelta(seconds=(window_end)),
                                    element + datetime.timedelta(seconds=(window_end + window_length))])
        df_process = error_code_series
        df_process = df_process.fillna(method='ffill')
        df_process = df_process.loc[df_process[errorcode_col] != errorcode].compute()
        df_process = df_process.dropna()
        for element in exclude_windows:
            mask = ~((df_process.index >= element[0]) & (df_process.index <= element[1]))
            df_process = df_process.loc[mask]
        return df_process

    else:
        df_process = error_code_series
        df_process = df_process.loc[df_process[errorcode_col] == errorcode].compute()
        target_errorcode_timestamp_list = [element for element in df_process.index]
        timestamps_to_process = [copy.deepcopy(target_errorcode_timestamp_list[0])]
        target_errorcode_timestamp_list.pop(0)
        for element in sorted(target_errorcode_timestamp_list):
            current_element = timestamps_to_process[-1]
            time_limit = element - datetime.timedelta(seconds=window_end + window_length)
            if current_element < time_limit:
                timestamps_to_process.append(element)
        df_process = df_process.loc[timestamps_to_process]
        return df_process

def calculate_window(df, minimal_features):

    df['id'] = 1

    if minimal_features:
        extracted_features = extract_features(df, column_id='id', n_jobs=0,
                                              default_fc_parameters=MinimalFCParameters())
    else:
        extracted_features = extract_features(df, column_id='id', n_jobs=0,
                                              default_fc_parameters=EfficientFCParameters())
    return extracted_features

@timing
def extract_windows_and_features(df, window_length, window_end, minimal_features = True, v_dask=True):

    extracted_features = []
    df = df.fillna(0)
    extracted_feature = dask.delayed(calculate_window)(df, minimal_features)
    extracted_features.append(extracted_feature)
    extracted_features = dask.compute(*extracted_features)
    df = pd.concat(extracted_features)
    df = df.reset_index(drop=True)

    return df







