"""
scripts/benchmark_benign_physics.py

Simulates 1 hour of the benign channel under first principles.
Compares the output variables against expected experimental regimes.
"""

import sys
import numpy as np

# Ensure core module can be imported
sys.path.append('.')

from core.channel import simulate_normal_channel
from core.config import PHYSICS_CONFIG

def main():
    print("=== Benign Physical Simulator Benchmark ===\n")
    print("Simulating 3600 seconds (1 hour)...")
    
    data = simulate_normal_channel(n_seconds=3600, seed=101)
    
    qber = data['qber']
    bell_S = data['bell_S']
    coincidences = data['coincidence_rate']
    loss_dB = data['channel_loss_dB']
    
    print("\n--- Mean Observables ---")
    print(f"Channel Loss : {np.mean(loss_dB):.2f} dB")
    print(f"Coincidences : {np.mean(coincidences):.0f} cps (Expected ~ 5k-20k cps for a 1-5km link)")
    print(f"QBER         : {np.mean(qber)*100:.2f} % (Expected ~ 1.0 - 4.0 % based on Erven et al. 2012)")
    print(f"Bell S       : {np.mean(bell_S):.3f} (Expected ~ 2.600 - 2.750 for highly entangled states)")
    
    print("\n--- Peak Perturbations (Worst Fades) ---")
    p99_loss = np.percentile(loss_dB, 99)
    mask_deep_fade = loss_dB > p99_loss
    
    print(f"Deepest 1% Fades (Mean Loss = {np.mean(loss_dB[mask_deep_fade]):.2f} dB):")
    print(f"  Coincidences drops to : {np.mean(coincidences[mask_deep_fade]):.0f} cps")
    print(f"  QBER spikes to        : {np.mean(qber[mask_deep_fade])*100:.2f} %")
    print(f"  Bell S degrades to    : {np.mean(bell_S[mask_deep_fade]):.3f}")
    
    print("\n--- Physical Validation ---")
    print("Notice that QBER and Bell S dynamically degrade during deep fades WITHOUT")
    print("any heuristic parameters coupling them. This happens strictly because the")
    print("signal coincidence rate drops toward the constant accidental (dark count) floor.")

if __name__ == '__main__':
    main()
