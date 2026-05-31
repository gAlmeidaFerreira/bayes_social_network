import os
import itertools
import time
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from configs.simulation_config import SimulationConfig
from core.environment import Environment

def run_single_experiment(run_id: int, params: dict, output_dir: Path) -> pd.DataFrame:
    """
    Worker function to execute a single simulation instance. 
    It captures the results, tags them with the specific hyperparameters used, 
    and saves the individual snapshot archive.
    """
    
    # Extract tuple and convert to numpy array for opinions_reliability
    if "opinions_reliability" in params:
        params["opinions_reliability"] = np.array(params["opinions_reliability"], dtype = float)
    
    config = SimulationConfig(**params)
    
    env = Environment(config)
    env.reset()
    
    for _ in range(config.max_steps):
        env.step()
        
    # Retrive the time-series dataframe
    df = env.tracker.get_dataframe()
    
    # Tag the dataframe with hypermarameters coordinates for later filtering
    df.insert(0, "run_id", run_id)
    for param_name, param_val in params.items():
        # If the parameter is an array, save it as a string representation in the CSV
        if isinstance(param_val, np.ndarray):
            df[param_name] = str(param_val.tolist())
        else:
            df[param_name] = str(param_val)
        
    # Save heavy structural snapshot archive independently
    npz_path = output_dir / f"alpha_snapshots_run_{run_id}.npz"
    np.savez_compressed(npz_path, **env.tracker.alpha_snapshots)
    
    return df

def run_parameter_sweep():
    """Generates the hyperparameter grid and orchestrates parallel execution."""
    print("Initializing Parameter Sweep...")
    
    # 1. Define the parameter grid
    sweep_grid = {
        "opinions_reliability": [(0.8,0.2), (0.6, 0.4), (0.5, 0.5)],
        "beta_soc": [1.0, 3.0, 5.0],
        "tau_soc": [0.5, 1.0, 1.5],
        "lambda_decay": [0.90, 0.95],
        "max_steps": [5000],
        "N": [100],
        "m": [3],
        "K": [2]
    }
    
    # Generate all matehmatical combinations of the defined parameters
    keys = list(sweep_grid.keys())
    combinations = list(itertools.product(*[sweep_grid[k] for k in keys]))
    
    # Convert combinations back into a list of kwargs dictionaries
    experiment_configs = [dict(zip(keys, combo)) for combo in combinations]
    total_runs = len(experiment_configs)
    
    output_dir = Path("data/sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Total experiment combinations to run: {total_runs}")
    start_time = time.time()
    
    all_results = []
    
    # 2. Parallel Execution
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_single_experiment, run_id, exp_params, output_dir): run_id
            for run_id, exp_params in enumerate(experiment_configs)
        }
        
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                result_df = future.result()
                all_results.append(result_df)
                print(f"Completed run {run_id + 1}/{total_runs}")
            except Exception as e:
                print(f"Run {run_id} failed with exception: {e}")
    
    # 3. Aggregation and final export
    if all_results:
        master_df = pd.concat(all_results, ignore_index=True)
        master_csv_path = output_dir / "master_sweep_results.csv"
        master_df.to_csv(master_csv_path, index=False)
        print(f"\nAll scalar data consolidated and saved to: {master_csv_path}")
        
    print(f"Sweep completed entirely in {time.time() - start_time:.2f} seconds.")
    
if __name__ == "__main__":
    run_parameter_sweep()
    
