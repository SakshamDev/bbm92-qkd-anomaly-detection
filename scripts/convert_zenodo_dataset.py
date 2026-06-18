import scipy.io as sio
import pandas as pd
import numpy as np
import os

def convert():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    zenodo_dir = os.path.join(base_dir, 'data', 'zenodo')
    output_path = os.path.join(base_dir, 'data', 'zenodo_ent_telemetry.parquet')

    print("Loading MATLAB files...")
    # Load data
    qber_mat = sio.loadmat(os.path.join(zenodo_dir, 'QBER_ent_Helmos_Matera.mat'))['QBER']
    skr_mat = sio.loadmat(os.path.join(zenodo_dir, 'SKR_ent_Helmos_Matera.mat'))['SKR']
    loss_mat = sio.loadmat(os.path.join(zenodo_dir, 'LINKLOSS_HELMOS.mat'))['LinkLoss1']

    # Column 0 seems to be the baseline scenario
    qber = qber_mat[:, 0]
    skr = skr_mat[:, 0]
    loss = loss_mat[:, 0]

    # Create dataframe
    df = pd.DataFrame({
        'time_index': np.arange(len(qber)) * 10,  # 10 second sampling rate
        'qber': qber,
        'skr': skr,
        'link_loss': loss
    })

    # The satellite is out of view when QBER is 0.5 (50%). Let's keep only the active pass.
    # Alternatively, we can keep all data and let the anomaly detection see the 'out of view' 
    # as 100% loss. But it's better to just extract the actual link passes.
    # We will flag "in_view" as true if QBER < 0.5
    df['in_view'] = df['qber'] < 0.5
    
    # Optional: Filter to only in-view samples for our testing
    df_valid = df[df['in_view']].copy()
    
    # Save to parquet
    df_valid.to_parquet(output_path, index=False)
    print(f"Successfully exported {len(df_valid)} valid telemetry records to {output_path}")

if __name__ == '__main__':
    convert()
