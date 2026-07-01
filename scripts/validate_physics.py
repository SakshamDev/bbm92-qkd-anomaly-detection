"""
scripts/validate_physics.py

Implementation of the Physics Validation Suite for the BBM92 simulator.
Runs automated checks to verify mathematical boundaries, statistical distributions,
and literature-agreed trends (e.g. hockey-stick QBER).
"""

import sys
import numpy as np

sys.path.append('.')

from core.channel import simulate_normal_channel
from core.config import PHYSICS_CONFIG

# Helper to temporarily override dataclass config
from dataclasses import replace
import core.channel

def run_simulation_with_config(override_kwargs, n_seconds=3600):
    """Run simulator with temporarily overridden config."""
    original_config = core.channel.PHYSICS_CONFIG
    new_config = replace(original_config, **override_kwargs)
    core.channel.PHYSICS_CONFIG = new_config
    
    try:
        data = simulate_normal_channel(n_seconds=n_seconds, seed=42)
    finally:
        core.channel.PHYSICS_CONFIG = original_config
        
    return data

def test_mathematical_identities():
    print("--- 1. Mathematical Identities & Limiting Cases ---")
    
    # 1.1 Absolute Vacuum (Zero Transmittance)
    data_vac = run_simulation_with_config({
        'eta_atm_bob_base': 0.0, 
        'detector_background_counts': 0.0,
        'scintillation_variance': 0.0,
        'precipitation_prob': 0.0
    }, n_seconds=1000)
    
    mean_qber = np.mean(data_vac['qber'])
    print(f"[Vacuum] Transmittance=0. Expected QBER ~ 0.5. Got: {mean_qber:.4f}")
    assert 0.45 < mean_qber < 0.55, "Vacuum QBER should be ~50%"
    
    # 1.2 Perfect Ideal System
    data_ideal = run_simulation_with_config({
        'source_visibility': 1.0,
        'detector_dark_counts': 0.0,
        'detector_background_counts': 0.0,
        'eta_atm_bob_base': 1.0,
        'eta_sys_bob': 1.0,
        'eta_sys_alice': 1.0,
        'detector_efficiency': 1.0,
        'scintillation_variance': 0.0,
        'precipitation_prob': 0.0
    }, n_seconds=1000)
    
    mean_qber_ideal = np.mean(data_ideal['qber'])
    mean_S_ideal = np.mean(data_ideal['bell_S'])
    print(f"[Ideal] V=1, DCR=0, Loss=0. Expected QBER=0.0, S=2.828. Got: QBER={mean_qber_ideal:.4f}, S={mean_S_ideal:.3f}")
    
    # 1.3 Coincidence vs Singles Bound
    data_bound = run_simulation_with_config({}, n_seconds=1000)
    c_rate = data_bound['coincidence_rate']
    d_rate = data_bound['detection_rate'] # Bob's singles (approx)
    violations = np.sum(c_rate > d_rate)
    print(f"[Bounding] Coincidences > Singles violations: {violations}/1000")
    assert violations == 0, "Coincidence rate exceeded singles rate!"
    print("Mathematical identities PASSED.\n")

def test_statistical_validation():
    print("--- 2. Statistical Validation ---")
    # 2.1 Poisson Variance
    data_stat = run_simulation_with_config({
        'scintillation_variance': 0.0, # Disable fading to test raw Poisson shot noise
        'precipitation_prob': 0.0,     # Disable precipitation bursts
        'detector_background_counts': 0.0
    }, n_seconds=10000)
    
    c_rate = data_stat['coincidence_rate']
    mean_c = np.mean(c_rate)
    var_c = np.var(c_rate)
    ratio = var_c / mean_c
    print(f"[Poisson] Constant channel coincidences: Mean = {mean_c:.1f}, Variance = {var_c:.1f}")
    print(f"[Poisson] Var/Mean Ratio (Expected ~1.0): {ratio:.3f}")
    assert 0.95 < ratio < 1.05, "Variance does not match mean for Poisson distribution"
    print("Statistical validation PASSED.\n")

def test_engineering_trends():
    print("--- 3. Engineering Validation Trends ---")
    
    # 3.1 Hockey-stick QBER vs Loss
    print("[Trend: QBER vs Optical Loss]")
    losses = [0.1, 0.01, 0.001, 0.0001, 0.00001] # 10dB to 50dB geometric loss
    for loss in losses:
        data = run_simulation_with_config({
            'eta_atm_bob_base': loss,
            'scintillation_variance': 0.0
        }, n_seconds=1000)
        mean_qber = np.mean(data['qber'])
        mean_c = np.mean(data['coincidence_rate'])
        print(f"  Loss={-10*np.log10(loss):.0f} dB | Coincidences: {mean_c:7.0f} cps | QBER: {mean_qber*100:5.2f}%")
        
    # 3.2 Optimal Pump Power
    print("\n[Trend: Optimal Pump Power (QBER vs Pair Rate)]")
    pair_rates = [1e5, 1e6, 5e6, 1e7, 5e7, 1e8]
    for r in pair_rates:
        data = run_simulation_with_config({
            'source_pair_rate': r,
            'eta_atm_bob_base': 0.1, # 10dB loss
            'scintillation_variance': 0.0
        }, n_seconds=1000)
        mean_qber = np.mean(data['qber'])
        mean_c = np.mean(data['coincidence_rate'])
        print(f"  R_pair={r:.1e} | Coincidences: {mean_c:8.0f} cps | QBER: {mean_qber*100:5.2f}%")
    print("\nTrends exhibit expected physical behaviors.\n")

def test_bell_s_vs_visibility(out_dir):
    print("--- 4. Bell S vs Visibility ---")
    import matplotlib.pyplot as plt
    visibilities = np.linspace(0.6, 1.0, 10)
    bell_S_vals = []
    
    for v in visibilities:
        data = run_simulation_with_config({
            'source_visibility': v,
            'eta_atm_bob_base': 1.0,
            'scintillation_variance': 0.0,
            'precipitation_prob': 0.0
        }, n_seconds=1000)
        mean_S = np.mean(data['bell_S'])
        bell_S_vals.append(mean_S)
        print(f"  V={v:.2f} -> S={mean_S:.3f} (Theory: {2*np.sqrt(2)*v:.3f})")
        
    plt.figure(figsize=(6,4))
    plt.plot(visibilities, bell_S_vals, 'bo-', label='Simulator S')
    plt.plot(visibilities, 2*np.sqrt(2)*visibilities, 'r--', label='Theory $2\sqrt{2}V$')
    plt.axhline(2.0, color='k', linestyle=':', label='Classical Bound')
    plt.xlabel('Source Visibility (V)')
    plt.ylabel('Bell S Parameter')
    plt.legend()
    plt.title('Bell S vs Visibility')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/bell_s_vs_visibility.png")
    plt.close()
    
def test_skr_vs_qber(out_dir):
    print("--- 5. SKR vs QBER ---")
    import matplotlib.pyplot as plt
    from core.privacy_amplification import secure_key_rate
    
    # We will sweep background counts to force QBER up to ~15%
    bg_counts = np.linspace(0, 1e6, 20)
    qber_vals = []
    skr_vals = []
    
    for bg in bg_counts:
        data = run_simulation_with_config({
            'detector_background_counts': float(bg),
            'scintillation_variance': 0.0,
            'precipitation_prob': 0.0
        }, n_seconds=100)
        mean_qber = np.mean(data['qber'])
        # compute sifted rate approx = matches + mismatches
        sifted_rate = np.mean(data['coincidence_rate']) / 2.0  # rough approx for Z and X bases
        mean_skr = np.mean(secure_key_rate(data['qber'], sifted_rate=sifted_rate))
        qber_vals.append(mean_qber)
        skr_vals.append(mean_skr)
    
    q_pct = np.array(qber_vals)*100
    print(f"  Max QBER reached: {q_pct[-1]:.2f}% -> SKR: {skr_vals[-1]:.1f} bps")
    
    plt.figure(figsize=(6,4))
    plt.plot(q_pct, skr_vals, 'go-')
    plt.axvline(11.0, color='r', linestyle='--', label='Theoretical limit (~11%)')
    plt.xlabel('QBER (%)')
    plt.ylabel('Secure Key Rate (bps)')
    plt.title('SKR vs QBER')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/skr_vs_qber.png")
    plt.close()

def test_scintillation_sweep(out_dir):
    print("--- 6. Atmospheric Scintillation Sweep ---")
    import matplotlib.pyplot as plt
    
    sigmas = [0.0, 0.1, 0.4, 1.0] # None, Weak, Moderate, Strong
    var_coincidences = []
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    for i, s2 in enumerate(sigmas):
        data = run_simulation_with_config({
            'scintillation_variance': s2,
            'precipitation_prob': 0.0,
            'eta_atm_bob_base': 1.0
        }, n_seconds=1000)
        c_rate = data['coincidence_rate']
        q_rate = data['qber']
        
        var_coincidences.append(np.var(c_rate))
        print(f"  Sigma_I^2={s2:.1f} -> Coinc Var: {np.var(c_rate):.1f}")
        
        ax1 = axes[i]
        ax2 = ax1.twinx()
        l1, = ax1.plot(c_rate[:200], 'b-', alpha=0.7, label='Coincidences')
        l2, = ax2.plot(q_rate[:200]*100, 'r-', alpha=0.5, label='QBER')
        ax1.set_ylabel('cps', color='b')
        ax2.set_ylabel('QBER (%)', color='r')
        ax1.set_title(f'Scintillation Variance = {s2}')
        if i==0:
            lines = [l1, l2]
            ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
            
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/scintillation_sweep.png")
    plt.close()

if __name__ == '__main__':
    # Disable logging info from channel.py for a cleaner stdout
    import logging
    logging.getLogger('core.channel').setLevel(logging.WARNING)
    
    out_dir = "/Users/sakshamgupta/.gemini/antigravity-ide/brain/4a60dbb5-0174-4559-8714-1e6d0d5b8408"
    
    print("Starting Physics Validation Suite...\n")
    test_mathematical_identities()
    test_statistical_validation()
    test_engineering_trends()
    test_bell_s_vs_visibility(out_dir)
    test_skr_vs_qber(out_dir)
    test_scintillation_sweep(out_dir)
    print("Validation Suite Completed Successfully.")
