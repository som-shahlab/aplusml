"""Wrapper functions around `sim.py` to help run/track multiple simulations"""
import os
import collections
from typing import Callable
import copy
import pandas as pd
import numpy as np
from mpire import WorkerPool
import aplusml.sim as sim

def test_diff_thresholds(simulation: sim.Simulation, 
                         all_patients: list[sim.Patient], 
                         thresholds: list[float],
                         utility_unit: str = '',
                         positive_outcome_state_ids: list[str] = [ 'positive_end_state', ],
                         **kwargs) -> pd.DataFrame:
    """Runs the simulation against each possible value for `model_threshold[value]` in the `thresholds` list

    Args:
        simulation (sim.Simulation): Simulation object
        all_patients (list[sim.Patient]): List of Patient objects
        thresholds (list[float]): List of model thresholds to binarize probabilistic predictions. Given t \in thresholds, we assume any prediction >= t is POSITIVE
        utility_unit (str, optional): Name of unit of utility that we want to sum over (e.g. 'qaly', 'usd'). Defaults to ''.
        positive_outcome_state_ids (list[str], optional): IDs of States that correspond to TP or FP outcomes, i.e. treatment was administered. Defaults to ['positive_end_state'].

    Returns:
        pd.DataFrame: 5 column DataFrame, each row is a diff threshold.
                        Columns: threshold, mean_utility, std_utility, sem_utility, mean_work_per_timestep
    """    
    rows = []
    assert 'model_threshold' in simulation.variables, "ERROR - The key 'model_threshold' must exist in 'simulation.variables' but is currently missing"
    for x in thresholds:
        simulation.variables['model_threshold']['value'] = x
        simulation.run(all_patients, **kwargs)
        utilities = [ p.get_sum_utilities(simulation)[utility_unit] for p in all_patients ]
        mean_work_per_timestep = len([p for p in all_patients if p.history[-1].state_id in positive_outcome_state_ids ]) / (simulation.current_timestep + 1)
        rows.append({
            'threshold' : x,
            'mean_utility' : np.mean(utilities),
            'std_utility' : np.std(utilities),
            'sem_utility' : np.std(utilities) / np.sqrt(len(all_patients)),
            'mean_work_per_timestep' : mean_work_per_timestep,
        })
    df = pd.DataFrame(rows)
    # Best model threshold
    max_threshold = df['threshold'].iloc[df['mean_utility'].argmax()]
    simulation.variables['model_threshold']['value'] = max_threshold
    # Set patients to correspond to best utility
    simulation.run(all_patients)
    return df

def _run_test(simulation: sim.Simulation,
                all_patients: list[sim.Patient],
                func_run_test: Callable,
                func_match_patient_to_property_column: Callable,
                is_refresh_patients: bool,
                l: str,
                k2v: dict,
                is_log: bool = False) -> pd.DataFrame:
    """Helper function used in `run_test()` to enable parallel processing of different runs - arguments have identical meanings
    
    Returns:
        pd.DataFrame: Has [1 + (# of cols returned by `test_diff_thresholds`)] columns, each row is a (threshold, label)
                        Columns: threshold, mean_utility, std_utility, sem_utility, mean_work_per_timestep, label
    """    
    if is_log:
        print(f"Run: {l}")
    simulation: sim.Simulation = copy.deepcopy(simulation)
    for key, val in k2v.items():
        simulation.variables[key] = val
    if is_refresh_patients:
        all_patients = sim.create_patients_for_simulation(simulation, all_patients, func_match_patient_to_property_column, random_seed = 0)
    if func_run_test:
        _df: pd.DataFrame = func_run_test(simulation, all_patients, l)
        _df['label'] = l
    else:
        simulation.run(all_patients)
        _df = collections.defaultdict(float)
        for p in all_patients:
            _u: dict = p.get_sum_utilities(simulation)
            for key, val in _u.items():
                _df[key] += val
        _df['label'] = l
        _df = pd.DataFrame([_df])
    return _df

def run_test(simulation: sim.Simulation, 
             all_patients: list[sim.Patient],
             labels: list, 
             keys2values: list[dict[dict]],
             df: pd.DataFrame = None, 
             func_run_test: Callable = None,
             func_match_patient_to_property_column: Callable = None,
             is_refresh_patients: bool = False,
             is_use_multi_processing: bool = False) -> pd.DataFrame:
    """Runs all settings of the simulation described in `keys2values` against all model cutoff thresholds in `thresholds`

    Args:
        simulation (sim.Simulation): Simulation object
        all_patients (list[sim.Patient]): List of Patient objects
        df (pd.DataFrame): Starting DataFrame (might contain results from previous runs or baselines)
        thresholds (list[float]): List of model thresholds to test
        labels (list): Name for each setting
        func_run_test (Callable): Function that actually runs the simulation -- typically set to `test_diff_thresholds`
        func_match_patient_to_property_column (Callable): Used in `create_patients_for_simulation`
        keys2values (list[dict[dict]]): Variables in simulation to overwrite for each setting
        is_refresh_patients (bool): If TRUE, then re-create patients after running each setting (useful for patient-specific properties)
        is_use_multi_processing (bool): If TRUE, then use MultiProcessing for faster speed

    Returns:
        pd.DataFrame: Has [1 + (# of cols returned by `test_diff_thresholds`)] columns, each row is a (threshold, label)
                        Columns: threshold, mean_utility, std_utility, sem_utility, mean_work_per_timestep, label
    """
    df = df.copy() if df is not None else pd.DataFrame()

    if is_use_multi_processing:
        n_jobs = os.cpu_count() - 1
        print('Processes:', n_jobs)
        with WorkerPool(n_jobs, use_dill=True) as pool:
            results = pool.map(_run_test, [(copy.deepcopy(simulation),
                                                all_patients,
                                                func_run_test,
                                                func_match_patient_to_property_column,
                                                is_refresh_patients,
                                                l, k2v) for l, k2v in zip(labels, keys2values) ])
            df = pd.concat(results + [df])
    else:
        for l, k2v in zip(labels, keys2values):
            _df = _run_test(copy.deepcopy(simulation),
                            all_patients,
                            func_run_test,
                            is_refresh_patients,
                            func_match_patient_to_property_column,
                            l, k2v)
            df = pd.concat([df, _df])
    return df

if __name__ == "__main__":
    pass