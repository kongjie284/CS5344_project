import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

def create_master_smote(train_path, output_path, target_n, dataset_name, 
                        bootstrap_fraction=0.40, alpha_max=0.45, normal_quantile=0.01):
    print(f"--- Processing {dataset_name} ---")
    train = pd.read_csv(train_path)
    label_col = 'is_anomaly'
    
    feature_cols = [c for c in train.columns if c != label_col]
    
    normal = train[train[label_col] == 0].copy().reset_index(drop=True)
    anomalous = train[train[label_col] == 1].copy().reset_index(drop=True)
    
    print(f"Normal samples: {len(normal)}, Anomalous samples: {len(anomalous)}")
    
    categorical_columns = [
        c for c in feature_cols
        if pd.api.types.is_object_dtype(normal[c]) or pd.api.types.is_string_dtype(normal[c]) or isinstance(normal[c].dtype, pd.CategoricalDtype)
    ]
    numeric_columns = [c for c in feature_cols if c not in categorical_columns]
    
    mins = normal[numeric_columns].min()
    maxs = normal[numeric_columns].max()
    
    grouped_normal = normal.groupby(categorical_columns)
    grouped_anom = anomalous.groupby(categorical_columns)
    
    stds = normal[numeric_columns].std(ddof=0)
    stds[stds == 0] = 1.0
    
    group_sizes = grouped_normal.size()
    # empirical weighting
    group_probs = group_sizes / group_sizes.sum()
    
    # We will compute quotas exactly to ensure exactly target_n rows
    raw_quotas = group_probs * target_n
    quotas = np.floor(raw_quotas).astype(int)
    remainder = target_n - quotas.sum()
    
    if remainder > 0:
        order = np.argsort(-(raw_quotas - quotas))
        quotas.iloc[order[:remainder]] += 1
        
    synthetic_dfs = []
    rng = np.random.default_rng(42)
    
    # Identify integer columns
    integer_columns = []
    for col in numeric_columns:
        values = normal[col].dropna().values
        if np.all(np.isclose(values, np.round(values))):
            integer_columns.append((col, normal[col].dtype))
    
    for g, quota in zip(group_sizes.index, quotas):
        if quota == 0:
            continue
            
        group_data = grouped_normal.get_group(g)
        norm_group = (group_data[numeric_columns] - mins) / stds
        
        neighbor = None
        margin = 0.0
        
        if g in grouped_anom.groups:
            anom_group = grouped_anom.get_group(g)
            if not anom_group.empty:
                norm_anom = (anom_group[numeric_columns] - mins) / stds
                neighbor = NearestNeighbors(n_neighbors=1, metric="manhattan", n_jobs=-1)
                neighbor.fit(norm_anom)
                
                dist_normal = neighbor.kneighbors(norm_group, return_distance=True)[0][:, 0]
                margin = float(np.quantile(dist_normal, normal_quantile))
                # Never exclude all normal samples
                margin = min(margin, float(np.nextafter(dist_normal.max(), -np.inf)))
                # Don't let margin be zero if there is space
                margin = max(margin, 1e-6)
                
        # Generate samples for this group
        bootstrap_n = int(round(quota * bootstrap_fraction))
        interpolation_n = quota - bootstrap_n
        
        group_synthetic = []
        
        # 1. Bootstrap
        if bootstrap_n > 0:
            boot_idx = rng.choice(len(group_data), size=bootstrap_n, replace=True)
            boot_samples = group_data.iloc[boot_idx].copy()
            group_synthetic.append(boot_samples[feature_cols])
            
        # 2. Interpolation
        accepted_interp = []
        accepted_count = 0
        while accepted_count < interpolation_n:
            remaining = interpolation_n - accepted_count
            batch_size = max(256, int(remaining * 1.5))
            
            anchors_idx = rng.choice(len(group_data), size=batch_size, replace=True)
            if len(group_data) == 1:
                donors_idx = anchors_idx
            else:
                donors_idx = rng.choice(len(group_data), size=batch_size, replace=True)
                same = donors_idx == anchors_idx
                while same.any():
                    donors_idx[same] = rng.choice(len(group_data), size=int(same.sum()), replace=True)
                    same = donors_idx == anchors_idx
                    
            alpha = rng.uniform(0, alpha_max, size=(batch_size, 1))
            left = group_data.iloc[anchors_idx][numeric_columns].values
            right = group_data.iloc[donors_idx][numeric_columns].values
            
            new_num = left + alpha * (right - left)
            new_num = np.clip(new_num, mins.values, maxs.values)
            
            cand_df = pd.DataFrame(new_num, columns=numeric_columns)
            
            for col, dtype in integer_columns:
                cand_df[col] = np.rint(cand_df[col]).astype(dtype)
                
            norm_cand = (cand_df - mins) / stds
            
            if neighbor is not None:
                dist_cand = neighbor.kneighbors(norm_cand, return_distance=True)[0][:, 0]
                valid = dist_cand >= margin
            else:
                valid = np.ones(batch_size, dtype=bool)
                
            valid_rows = cand_df.loc[valid].copy()
            take = min(remaining, len(valid_rows))
            
            if take > 0:
                # Add categorical columns
                if isinstance(g, tuple):
                    for i, col in enumerate(categorical_columns):
                        valid_rows[col] = g[i]
                else:
                    valid_rows[categorical_columns[0]] = g
                    
                accepted_interp.append(valid_rows.iloc[:take][feature_cols])
                accepted_count += take
                
        if accepted_interp:
            group_synthetic.append(pd.concat(accepted_interp, ignore_index=True))
            
        synthetic_dfs.append(pd.concat(group_synthetic, ignore_index=True))
        
    final_df = pd.concat(synthetic_dfs, ignore_index=True)
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_df.insert(0, "id", np.arange(target_n))
    final_df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with shape {final_df.shape}\n")

if __name__ == '__main__':
    # NSL-KDD
    # Use config that gave great results for NSL-KDD
    create_master_smote(
        train_path='../train-NSL-KDD.csv', 
        output_path='submission-NSL-KDD.csv', 
        target_n=40000, 
        dataset_name='NSL-KDD',
        bootstrap_fraction=0.40,
        alpha_max=0.45,
        normal_quantile=0.01
    )
    
    # UNSW-NB15
    create_master_smote(
        train_path='../train-UNSW-NB15.csv', 
        output_path='submission-UNSW-NB15.csv', 
        target_n=70000, 
        dataset_name='UNSW-NB15',
        bootstrap_fraction=0.40,
        alpha_max=0.45,
        normal_quantile=0.01
    )
