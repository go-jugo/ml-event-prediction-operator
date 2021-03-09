from pipeline import run_pipeline
import pandas as pd
import numpy as np
import random
import pickle
from monitoring.initialize_monitoring import initialize_monitoring
from config import configs_pipeline, v_dask
from local_conf import local_workers, local_threads_per_worker, local_memory_limit, only_base_conf_run
from dask import dataframe as dd
from datetime import datetime
from tools.signature import GLOBAL_SIGNATURE

if __name__ == '__main__':

    def start_ml_worker(df, configs_pipeline, clf):
        if v_dask:
            from dask.distributed import Client
            CLIENT = Client(n_workers=local_workers, threads_per_worker=local_threads_per_worker,
                            memory_limit=local_memory_limit)
            print(CLIENT.scheduler_info()['services'])
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
