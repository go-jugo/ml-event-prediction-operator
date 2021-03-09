from local_conf import server
import importlib.util
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

v_dask = True
debug_mode = False
write_monitoring = False
store_results = False

def create_configs():
    base_config = dict(
        sampling_frequency=['30S'],
        imputations_technique_str=['pad'],
        imputation_technique_num=['pad'],
        ts_fresh_window_length=[3600],
        ts_fresh_window_end=[3600],
        ts_fresh_minimal_features=[True],
        scaler=[StandardScaler()],
        target_col=['module_4_errorcode'],
        target_errorCode=[2],
        balance_ratio = [0.5],
        random_state = [[0]],
        cv=[5],
        oversampling_method = [False],
        ml_algorithm=[RandomForestClassifier()]
        )
    configs_pipeline = [dict(zip(base_config, v)) for v in product(*base_config.values())]
    return configs_pipeline

configs_pipeline = create_configs()

