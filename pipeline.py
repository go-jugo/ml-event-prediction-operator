# import pandas as pd
# import numpy as np
from types import SimpleNamespace
# from .data_merge import create_global_dataframe
from .data_cleansing import impute_missing_values, value_based_column_filter, one_hot_encode_categories, remove_correlated_features, remove_error_codes
from .feature_engineering import standardize_features, generate_error_code_indicators
from .feature_engineering.extract_windows_and_engineer_features_with_tsfresh import extract_windows_and_features
from .data_cleansing.adjust_sampling_frequency import adjust_sampling_frequency
from .ml_evaluation.eval import eval
from .feature_engineering.remove_global_timestamp import remove_global_timestamp
# from multiprocessing import Pool
from .config import v_dask, logging_level, logging_color
# from .feature_engineering.create_error_code_col import create_error_code_col
from .logger import init_logger, get_logger


init_logger(logging_level, logging_color)

logger = get_logger(__name__.split(".", 1)[-1])


def run_pipeline(df, config, clf):
    c = SimpleNamespace(**config)
    logger.debug('Data cleansing')
    df = value_based_column_filter.value_filter(df, v_dask=v_dask)
    df = adjust_sampling_frequency(df, sampling_frequency=c.sampling_frequency, v_dask=v_dask)
    df = impute_missing_values.impute_missing_values(df, v_dask=v_dask, string_imputation_method=c.imputations_technique_str,
                                                     numeric_imputation_method=c.imputation_technique_num)
    df = impute_missing_values.slice_valid_data(df, v_dask=v_dask)
    df = one_hot_encode_categories.one_hot_encode_categories(df, errorcode_col=c.target_col, v_dask=v_dask)
    df = remove_error_codes.remove_error_codes(df, dependent_variable=c.target_col)
    logger.debug('Feature engineering')
    df = extract_windows_and_features(df, window_length=c.ts_fresh_window_length, window_end=c.ts_fresh_window_end,
                                      minimal_features=c.ts_fresh_minimal_features)
    logger.debug('ML evaluation')
    df = standardize_features.standardize_features(df, errorcode_col=c.target_col, scaler=c.scaler)
    df = remove_global_timestamp(df)
    prediction = eval(df, clf=clf)
    return prediction