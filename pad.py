import pandas as pd
import numpy as np
from typing import Callable, Tuple
import sim
import parse
import run

"""

    Helper functions for pad.ipynb

"""

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
    print("---- Random Nonce for Resource Priority ----")
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
def generate_patient_list(simulation: sim.Simulation,
                          mean_admits_per_day: int = 35, 
                          num_days: int = 500) -> list[sim.Patient]:
    """Generate list of Patient objects fed into Simulation
        These patients have the model predictions from Elsie's original files,
        plus a `start_timestep` (so that their admittance is properly staggered)

    Args:
        mean_admits_per_day (int, optional): Mean of Poisson dist for admits per day. Defaults to 35.
        num_days (int, optional): Total number of days to simulate. Defaults to 500.

    Returns:
        list[sim.Patient]: List of Patients
    """    
    # Sample patients randomly
    np.random.seed(0)

    # Simulate number of patients per day
    num_admits_per_day = np.random.poisson(lam=mean_admits_per_day, 
                                            size=num_days)

    # Simulate patients
    all_patients: list[sim.Patient] = []
    for timestep, n_admits in enumerate(num_admits_per_day):
        for x in range(n_admits):
            all_patients.append(sim.Patient(
                len(all_patients), # ID
                timestep, # Start timestep
            ))
    all_patients = sim.create_patients_for_simulation(simulation, 
                                                        all_patients,
                                                        # Ignore p.id since we want to randomly sample patients from our CSV
                                                        lambda p_id, random_idx, df, col: df.iloc[random_idx][col],
                                                        random_seed = 0)
    return all_patients

def setup_simulation_for_model(simulation: sim.Simulation, 
                               model: str = 'dl',
                               is_patient_sort_by_y_hat: bool = True):
    """Ensure that we're using the proper `y_hat` for the model we want to test (as specified in `model`)

    Args:
        simulation (sim.Simulation): Simulation
        model (str, optional): String (dl|lr|rf). Defaults to 'dl'.
        is_patient_sort_by_y_hat (bool, optional): If TRUE, then prioritize patients from HIGHEST y_hat -> LOWEST y_hat
    """    
    simulation.states['model_pred'].transitions[0]._if = f"y_hat_{model} >= model_threshold"
    if is_patient_sort_by_y_hat:
        simulation.metadata['patient_sort_preference_property']['variable'] = f"y_hat_{model}"
        simulation.metadata['patient_sort_preference_property']['is_ascending'] = False
    else:
        simulation.metadata['patient_sort_preference_property']['variable'] = f"random_resource_priority"
        simulation.metadata['patient_sort_preference_property']['is_ascending'] = False

def load_simulation(path_to_yaml: str, path_to_properties: str) -> sim.Simulation:
    """Loads YAML into Simulation object

    Args:
        path_to_yaml (str): Path to YAML file

    Returns:
        sim.Simulation: Simulation object
    """    
    yaml = parse.load_yaml(path_to_yaml)
    simulation = parse.create_simulation_from_yaml(yaml)
    simulation.metadata['path_to_properties'] = path_to_properties
    return simulation

def load_simulation_for_model(path_to_yaml: str,
                              path_to_properties: str,
                              model: str,
                              is_patient_sort_by_y_hat: bool,
                              func_setup_optimistic: Callable = None,
                              random_seed: int = 0) -> sim.Simulation:
    """Loads YAML into Simulation object, sets up Model-specific settings, and sets up optimistic settings

    Args:
        model (str): Name of model (dl|rf|lr)
        is_patient_sort_by_y_hat (bool): if TRUE, then prioritize patients by y_hat
        func_setup_optimistic (Callable, optional): Workflow specific. Defaults to None.
        random_seed (int, optional): Random seed. Defaults to 0.

    Returns:
        sim.Simulation: Simulation object
    """    
    simulation = load_simulation(path_to_yaml, path_to_properties)
    setup_simulation_for_model(simulation, model, is_patient_sort_by_y_hat)
    if func_setup_optimistic:
        func_setup_optimistic(simulation)
    return simulation



###################################
#
# Simulation
#
###################################
def run_test(patients: list[sim.Patient],
             labels: list[str], 
             settings: list[dict], 
             MODELS: list[str],
             THRESHOLDS: list[float],
             PATH_TO_YAML: str,
             PATH_TO_PROPERTIES: str,
             is_patient_sort_by_y_hat: bool = False,
             func_setup_optimistic: Callable = None,
             is_use_multi_processing: bool = True) -> Tuple[dict, dict]:
    model_2_result = {} # [key] = model name, [value] = df
    for m in MODELS:
        simulation = load_simulation_for_model(PATH_TO_YAML, 
                                               PATH_TO_PROPERTIES, 
                                               m, 
                                               is_patient_sort_by_y_hat=is_patient_sort_by_y_hat, 
                                               func_setup_optimistic=func_setup_optimistic)
        df_result = run.run_test(simulation, patients,
                                 labels + [f"optimistic"], settings + [{}],
                                 pd.DataFrame(),
                                 lambda sim, patients, label: run.test_diff_thresholds(sim, patients, THRESHOLDS, utility_unit='qaly'),
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
        simulation = load_simulation_for_model(PATH_TO_YAML, PATH_TO_PROPERTIES, MODELS[0], is_patient_sort_by_y_hat=False, func_setup_optimistic=func_setup_optimistic)
        simulation.states['model_pred'].transitions[0]._if = _if
        # This only has one threshold (at 0) since this is independent of the model (and thus the threshold of the model)
        df_result = run.run_test(simulation, patients,
                                labels + [f"optimistic"], settings + [{}],
                                pd.DataFrame(),
                                lambda sim, patients, label: run.test_diff_thresholds(sim, patients, [0], utility_unit='qaly'),
                                is_use_multi_processing=is_use_multi_processing)
        baseline_2_result[label] = df_result.copy()
    return model_2_result, baseline_2_result


###################################
#
# Plotting
#
###################################
def plot_helper(model_2_result: dict, 
                baseline_2_result: dict,
                threshold: float = None) -> pd.DataFrame:
    """Convert `model_2_result` and `baseline_2_result` dicts into unified dfs
    If `threshold` is not specified, choose `threshold` that achieves max `mean_utility`

    Args:
        model_2_result (dict): Output from `run_test`
        baseline_2_result (dict): Output from `run_test`
        threshold (float, optional): If not specified, choose `threshold` that achieves max `mean_utility`. Defaults to None.

    Returns:
        pd.DataFrame: 4 columns -- label, y, y_sem, model
    """    
    plot_avg_utilities = []
    for m, df in model_2_result.items():
        for label in df['label'].unique():
            if threshold:
                y = df[(df['label'] == label) & (df['threshold'] == threshold)]['mean_utility'].values[0]
                y_sem = df[(df['label'] == label) & (df['threshold'] == threshold)]['sem_utility'].values[0]
            else:
                max_idx = df[df['label'] == label]['mean_utility'].argmax()
                y = df[df['label'] == label].iloc[max_idx]['mean_utility']
                y_sem = df[df['label'] == label].iloc[max_idx]['sem_utility']
            plot_avg_utilities.append({
                'label' : label,
                'y' : y,
                'y_sem' : y_sem,
                'model' : m,
            })
    for m, df in baseline_2_result.items():
        plot_avg_utilities += [{
                'label' : row['label'],
                'y' : row['mean_utility'],
                'y_sem' : 0,
                'model' : m,
            } for idx, row in df.iterrows() ]
    df_avg_utilities = pd.DataFrame(plot_avg_utilities)
    return df_avg_utilities