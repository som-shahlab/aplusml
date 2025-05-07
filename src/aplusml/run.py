"""
Wrapper functions around `sim.py` to help run/track multiple simulations
"""
import os
import collections
from typing import Callable, Optional
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
    """Tests different model threshold values to find optimal cutoff point for binary predictions.

    For each threshold value, runs the simulation and calculates utility metrics. The simulation must 
    contain a ``model_threshold`` variable that is used to binarize probabilistic predictions.
    After testing all thresholds, sets the simulation and patients to use the threshold that 
    maximizes mean utility.

    Args:
        simulation (sim.Simulation): Simulation object containing workflow definition
        all_patients (list[sim.Patient]): List of patients to simulate through workflow
        thresholds (list[float]): List of threshold values to test (between 0 and 1)
        utility_unit (str, optional): Name of utility unit to optimize (e.g. ``'qaly'``, ``'usd'``). Defaults to ``''``.
        positive_outcome_state_ids (list[str], optional): State IDs that represent positive outcomes 
            (treatment administered). Used to calculate work per timestep. Defaults to ``['positive_end_state']``.
        **kwargs: Additional arguments passed to ``simulation.run()``

    Returns:
        pd.DataFrame: Results for each threshold tested with columns
        
            - ``threshold``: Threshold value tested
            - ``mean_utility``: Mean utility achieved across all patients
            - ``std_utility``: Standard deviation of utilities
            - ``sem_utility``: Standard error of mean utility
            - ``mean_work_per_timestep``: Average number of positive outcomes per timestep

    Raises:
        AssertionError: If ``model_threshold`` variable is not defined in ``simulation.variables``
    """    
    rows = []
    assert 'model_threshold' in simulation.variables, "ERROR - The key 'model_threshold' must exist in 'simulation.variables' but is currently missing"
    for x in thresholds:
        simulation.variables['model_threshold']['value'] = x
        all_patients = simulation.run(all_patients, **kwargs)
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
    all_patients = simulation.run(all_patients)
    return df

def _run_test(simulation: sim.Simulation,
                all_patients: list[sim.Patient],
                func_run_test: Optional[Callable],
                func_match_patient_to_property_column: Callable,
                is_refresh_patients: bool,
                l: str,
                k2v: dict,
                is_log: bool = False) -> pd.DataFrame:
    """Helper function that runs a single simulation test with specified parameters.

    This function is designed to be used by ``run_test()`` for both serial and parallel processing.
    
    It handles:
    
        1. Creating a deep copy of the simulation to avoid state conflicts
        2. Updating simulation variables based on test settings
        3. Optionally refreshing patient properties
        4. Running either a custom test function or basic simulation
        5. Collecting and formatting results

    Args:
        simulation (sim.Simulation): Base simulation object to copy and modify
        all_patients (list[sim.Patient]): List of patients to simulate
        func_run_test (Optional[Callable]): Custom function to run simulation test, typically test_diff_thresholds.
            If ``None``, runs basic simulation and sums utilities.
        func_match_patient_to_property_column (Callable): Function to match patients to properties in CSV.
            Takes ``(patient_id, random_idx, df, column)`` as arguments.
        is_refresh_patients (bool): If ``True``, recreates patient objects with new properties
        l (str): Label for this test run
        k2v (dict): Dictionary mapping variable names to new values for this test
        is_log (bool, optional): If ``True``, prints run progress. Defaults to ``False``.

    Returns:
        pd.DataFrame: Results dataframe. If using ``func_run_test``, matches that function's output
            with added ``label`` column. Otherwise, contains summed utilities and ``label``.
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
        all_patients = simulation.run(all_patients)
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
    """Runs multiple simulation tests with different variable settings.

    This is the main entry point for running simulation experiments. It supports:

        1. Testing multiple configurations in parallel or serial
        2. Custom test functions (e.g. threshold testing)
        3. Patient property refreshing between runs
        4. Appending results to existing dataframes
        5. Multiprocessing for improved performance

    The function pairs each label with its corresponding variable settings from ``keys2values``
    and runs the simulation with those settings. Results from all runs are combined into
    a single dataframe.

    Args:
        simulation (sim.Simulation): Base simulation object
        all_patients (list[sim.Patient]): List of patients to simulate
        labels (list): Names for each test configuration
        keys2values (list[dict[dict]]): List of variable settings to test. Each dict maps
            variable names to new values/settings for that test run.
        df (pd.DataFrame, optional): Existing results to append to. Defaults to None.
        func_run_test (Callable, optional): Custom function to run each test. Typically
            test_diff_thresholds. If ``None``, runs basic simulation. Defaults to ``None``.
        func_match_patient_to_property_column (Callable, optional): Function to match patients
            to properties in CSV. Required if refreshing patients. Defaults to ``None``.
        is_refresh_patients (bool, optional): If ``True``, recreates patients with new properties
            between runs. Defaults to ``False``.
        is_use_multi_processing (bool, optional): If ``True``, runs tests in parallel using
            available CPU cores. Defaults to ``False``.

    Returns:
        pd.DataFrame: Combined results from all test runs. Format depends on ``func_run_test``,
            but always includes a ``label`` column identifying the test configuration.

    References:
        For usage examples, see:
        - :doc:`/usage/tutorial_pad`
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