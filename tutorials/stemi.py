"""Helper functions for STEMI-specific workflow analysis"""
import pandas as pd
import numpy as np
from typing import Callable, Tuple, List
import aplusml

# TODO
def generate_csv(PATH_TO_DL_MODEL, PATH_TO_RF_MODEL, PATH_TO_LR_MODEL, PATH_TO_PATIENT_PROPERTIES) -> pd.DataFrame:
    #
    # Read CSVs
    #
    print("---- Read ----")
    # Read model predictions
    df_dl = pd.read_csv(PATH_TO_DL_MODEL)
    df_rf = pd.read_csv(PATH_TO_RF_MODEL)
    df_lr = pd.read_csv(PATH_TO_LR_MODEL)
    print('Size:', 'df_dl =', df_dl.shape, 'df_rf =', df_rf.shape, 'df_lr =', df_lr.shape)
    # Standardize columns
    df_dl = df_dl.rename(columns={ 'Person_id' : 'id', 'Label' : 'y', 'Pred Proba' : 'y_hat' })[['id', 'y', 'y_hat']]
    df_rf = df_rf.rename(columns={ 'person_id' : 'id', 'label' : 'y', 'prediction' : 'y_hat' })[['id', 'y', 'y_hat']]
    df_lr = df_lr.rename(columns={ 'person_id' : 'id', 'label' : 'y', 'prediction' : 'y_hat' })[['id', 'y', 'y_hat']]
    print("---- De-duplicate ----")
    # Drop duplicates (since CSV files contain multiple folds), keep first predicition
    print("Removed", df_dl['id'].shape[0] - df_dl['id'].nunique(), "repeated rows in df_dl")
    print("Removed", df_rf['id'].shape[0] - df_rf['id'].nunique(), "repeated rows in df_rf")
    print("Removed", df_lr['id'].shape[0] - df_lr['id'].nunique(), "repeated rows in df_lr")
    df_dl = df_dl.drop_duplicates(subset=['id'], keep='first')
    df_rf = df_rf.drop_duplicates(subset=['id'], keep='first')
    df_lr = df_lr.drop_duplicates(subset=['id'], keep='first')
    # Get de-duplicated df sizes
    print('Size:', 'df_dl =', df_dl.shape, 'df_rf =', df_rf.shape, 'df_lr =', df_lr.shape)
    #
    # Merge CSVs
    #
    print("---- Merge ----")
    # Merge df's by patient ID
    print("# of overlapping patients btwn df_dl and df_rf:", df_dl.merge(df_rf, how='inner', on='id', suffixes=('_dl', '_rf')).shape[0])
    print("# of overlapping patients btwn df_dl and df_lr:", df_dl.merge(df_lr, how='inner', on='id', suffixes=('_dl', '_rf')).shape[0])
    print("# of overlapping patients btwn df_lr and df_rf:", df_lr.merge(df_rf, how='inner', on='id', suffixes=('_dl', '_rf')).shape[0])
    df_merged = df_dl.merge(df_rf, how='inner', on='id', suffixes=('_dl', '_rf')) \
                    .merge(df_lr, how='inner', on='id', suffixes=('', '_lr')) \
                    .rename(columns = { 'y' : 'y_lr', 'y_hat' : 'y_hat_lr' })
    # Remove patients who have conflict ground-truth labels (i.e. `y` doesn't match across all models)
    conflicting_y_row_idxs = np.where(~((df_merged['y_dl'] == df_merged['y_rf']) &
                                        (df_merged['y_dl'] == df_merged['y_lr']) & 
                                        (df_merged['y_rf'] == df_merged['y_lr'])))[0]
    print('Removed', len(conflicting_y_row_idxs), "rows in df_merged with conflicting 'y' labels")
    df_merged = df_merged.drop(index = conflicting_y_row_idxs)
    # Standardize `y` column
    df_merged['y'] = df_merged['y_dl'].astype(int)
    df_merged = df_merged.drop(columns = ['y_dl', 'y_rf', 'y_lr',])
    print('Size:', 'df_merged =', df_merged.shape)
    #
    # Add patient properties
    #
    # ABI test
    print("---- ABI Test ----")
    np.random.seed(0)
    df_merged.loc[df_merged['y'] == 1, 'abi_test_pred'] = np.random.normal(0.65, 0.15, np.sum(df_merged['y'] == 1))
    df_merged.loc[df_merged['y'] == 0, 'abi_test_pred'] = np.random.normal(1.09, 0.11, np.sum(df_merged['y'] == 0))
    tp = np.sum(df_merged[df_merged['y'] == 1]['abi_test_pred'] < 0.9)
    fn = np.sum(df_merged[df_merged['y'] == 1]['abi_test_pred'] >= 0.9)
    fp = np.sum(df_merged[df_merged['y'] == 0]['abi_test_pred'] < 0.9)
    tn = np.sum(df_merged[df_merged['y'] == 0]['abi_test_pred'] >= 0.9)
    print("ABI sensitivity @ 0.9:", tp / (tp + fn))
    print("ABI specificity @ 0.9:", tn / (tn + fp))
    # For prioritization to limited resources
    print("---- Set Random Nonce for Resource Priority ----")
    df_merged['random_resource_priority'] = np.random.choice(df_merged.shape[0], df_merged.shape[0], replace=False)
    # Save to CSV
    print("---- Stats ----")
    print("Prevalence of PAD:", np.mean(df_merged['y']))
    for x in [.4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9]:
        print(f"P(ABI < {x}| PAD):", np.mean(df_merged.loc[df_merged['y'] == 1]['abi_test_pred'] < x))
    df_merged.to_csv(PATH_TO_PATIENT_PROPERTIES)
    return df_merged


###################################
#
# Patients
#
###################################
def generate_patient_list(simulation: aplusml.Simulation,
                          mean_admits_per_day: int = 170, 
                          num_days: int = 100) -> List[aplusml.Patient]:
    """Generate list of Patient objects fed into Simulation
        These patients have the model predictions from Maya's original files,
        plus a `start_timestep` (so that their admittance is properly staggered)

    Args:
        mean_admits_per_day (int, optional): Mean of Poisson dist for admits per day. Defaults to 170.
        num_days (int, optional): Total number of days to simulate. Defaults to 100.

    Returns:
        List[aplusml.Patient]: List of Patients
    """    
    # Sample patients randomly
    np.random.seed(0)

    # Simulate number of patients per day
    num_admits_per_day = np.random.poisson(lam=mean_admits_per_day, size=num_days)

    # Simulate patients
    all_patients: List[aplusml.Patient] = []
    for timestep, n_admits in enumerate(num_admits_per_day):
        for x in range(n_admits):
            all_patients.append(aplusml.Patient(
                len(all_patients), # ID
                timestep, # Start timestep
            ))
    all_patients = aplusml.create_patients_for_simulation(simulation, 
                                                          all_patients,
                                                          # Ignore p.id since we want to randomly sample patients from our CSV
                                                          lambda p_id, random_idx, df, col: df.iloc[random_idx][col],
                                                          random_seed = 0)
    return all_patients


###################################
#
# Simulation
#
###################################
def run_test(patients: List[aplusml.Patient],
             labels: List[str], 
             settings: List[dict], 
             MODELS: List[str],
             THRESHOLDS: List[float],
             PATH_TO_YAML: str,
             PATH_TO_PROPERTIES: str,
             is_use_multi_processing: bool = True) -> Tuple[dict, dict]:
    model_2_result = {} # [key] = model name, [value] = df
    for m in MODELS:
        simulation = aplusml.load_simulation(PATH_TO_YAML, PATH_TO_PROPERTIES)
        df_result = aplusml.run_test(simulation, patients,
                                 labels + [f"optimistic"], settings + [{}],
                                 pd.DataFrame(),
                                 lambda sim, patients, label: aplusml.test_diff_thresholds(sim, patients, THRESHOLDS, utility_unit='qaly'),
                                 is_use_multi_processing=is_use_multi_processing)
        model_2_result[m] = df_result.copy()
    # Save Treat All / None / Perfect baselines under optimistic conditions
    baselines = [
        # Treat All - treat everyone as if y_hat == 1
        { 'label' : 'all', '_if' : True },
        # Treat None - treat everyone as if y_hat == 0
        { 'label' : 'none', '_if' : False },
        # Treat Perfect - treat everyone as if y_hat == y
        { 'label' : 'perfect', '_if' : 'y == 1' },
    ]
    baseline_2_result = {}
    for b in baselines:
        label = b['label']
        _if = b['_if']
        # This is INDEPENDENT of any model (MODELS[0] used for simplicity here)
        simulation = aplusml.load_simulation(PATH_TO_YAML, PATH_TO_PROPERTIES)
        simulation.states['model_prediction'].transitions[0]._if = _if
        # This only has one threshold (at 0) since this is independent of the model (and thus the threshold of the model)
        df_result = aplusml.run_test(simulation, patients,
                                labels + [f"optimistic"], settings + [{}],
                                pd.DataFrame(),
                                lambda sim, patients, label: aplusml.test_diff_thresholds(sim, patients, [0], utility_unit='qaly'),
                                is_use_multi_processing=is_use_multi_processing)
        baseline_2_result[label] = df_result.copy()
    return model_2_result, baseline_2_result

###################################
#
# Workflows
#
###################################
def setup_optimistic(simulation: aplusml.Simulation):
    """Settings for "optimistic" case
    """
    # Reset simulation to "optimistic" case
    simulation.variables['model_threshold']['value'] = 0.004065758
    # Capacities
    for resource in ['registration_clerk', 'triage_nurse', 'ecg_tech', 'triage_physician']:
        simulation.variables[resource]['init_amount'] = 1e5
        simulation.variables[resource]['max_amount'] = 1e5
        simulation.variables[resource]['refill_amount'] = 1e5
        simulation.variables[resource]['refill_duration'] = 1

if __name__ == "__main__":
    pass