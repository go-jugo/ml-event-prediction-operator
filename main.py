from pipeline import run_pipeline
import pandas as pd
import numpy as np
import pickle
from monitoring.initialize_monitoring import initialize_monitoring
from config import configs_pipeline, v_dask
from dask import dataframe as dd
from tools.signature import GLOBAL_SIGNATURE

if __name__ == '__main__':

    def start_ml_worker(df, configs_pipeline, clf):
        for config in configs_pipeline:
            print(f'Pipeline runs with config: {config}')
            prediction = run_pipeline(df, config, clf)
            return prediction

    df = dd.read_csv('output_1.csv')
    df['time'] = dd.to_datetime(df['time'])
    df = df.set_index('time')

    with open('ml_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    prediction = start_ml_worker(df, configs_pipeline, clf)
    print(prediction)
