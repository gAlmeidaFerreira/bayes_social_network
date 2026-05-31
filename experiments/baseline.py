import os
import time
import numpy as np
from pathlib import Path
from configs.simulation_config import SimulationConfig
from core.environment import Environment

def run_baseline():
    """
    Executes a single instance of the opinion dynamics simulation using 
    the baseline configuration. Exports scalar telemetry to CSV and 
    caches raw structural snapshots to a compressed NumPy archive.
    """
    print("Initializing Baseline Simulation")
    
    # 1. Initizalize configuration with explicit tracking intervals
    config = SimulationConfig(
        N=100,
        K=2,
        max_steps=10000
    )
    
    config.record_alphas_every = 200
    
    # Setting up environment
    env = Environment(config)
    env.reset()
    
    # Execution Loop
    start_time = time.time()
    for step in range(config.max_steps):
        env.step()
        
        # Print progress every 10%
        if step > 0 and step % (config.max_steps // 10) == 0:
            progress = (step / config.max_steps) * 100
            print(f"Simulation Progress: {progress: .0f}%")
            
    execution_time = time.time() - start_time
    print(f"Simulation completed in {execution_time: .2f} seconds")
    
    # Export Pipeline
    output_dir = Path("data/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export fast track scallar metrics
    df_results = env.tracker.get_dataframe()
    csv_path = output_dir / "baseline_scalars.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Scalar metrics saved to: {csv_path}")
    
    # Export slow track multi-dimansional structural snapshots
    npz_path = output_dir / "baseline_alpha_snapshots.npz"
    np.savez_compressed(npz_path, **env.tracker.alpha_snapshots)
    print(f"Structural matrices saved to: {npz_path}")
    
if __name__ == "__main__":
    run_baseline() 