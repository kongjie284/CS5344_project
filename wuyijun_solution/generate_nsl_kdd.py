import pandas as pd
import numpy as np
import os
from sklearn.neighbors import NearestNeighbors

def create_safe_smote(train_path, output_path, target_n, dataset_name):
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
    
    stds = normal[numeric_columns].std(ddof=0)
    stds[stds == 0] = 1.0
    
    norm_anom_num = (anomalous[numeric_columns] - mins) / stds
    
    # Fit NN for anomalies
    nn_anom = NearestNeighbors(n_neighbors=1, metric='manhattan', n_jobs=-1)
    nn_anom.fit(norm_anom_num)
    
    group_sizes = grouped_normal.size()
    group_probs = group_sizes / group_sizes.sum()
    
    synthetic_dfs = []
    generated_count = 0
    rng = np.random.default_rng(42)
    
    while generated_count < target_n:
        batch_size = min(target_n - generated_count + 1000, 15000)
        chosen_groups = rng.choice(group_sizes.index, size=batch_size, p=group_probs)
        
        new_numeric_rows = []
        new_cat_rows = []
        
        for g in chosen_groups:
            group_data = grouped_normal.get_group(g)
            
            if len(group_data) == 1:
                anchor = group_data.iloc[0]
                new_numeric_rows.append(anchor[numeric_columns].values)
                # handle the case where g is a single value vs tuple
                cat_vals = anchor[categorical_columns].values
                new_cat_rows.append(cat_vals)
                continue
                
            idx1, idx2 = rng.integers(0, len(group_data), size=2)
            anchor = group_data.iloc[idx1]
            other = group_data.iloc[idx2]
            alpha = rng.uniform(0, 1)
            
            new_num = anchor[numeric_columns].values + alpha * (other[numeric_columns].values - anchor[numeric_columns].values)
            new_numeric_rows.append(new_num)
            new_cat_rows.append(anchor[categorical_columns].values)
            
        new_numeric_np = np.array(new_numeric_rows)
        new_numeric_np = np.clip(new_numeric_np, mins.values, maxs.values)
        
        norm_new_num = (new_numeric_np - mins.values) / stds.values
        dist_to_anom, _ = nn_anom.kneighbors(norm_new_num)
        
        valid_mask = dist_to_anom[:, 0] > 0.05
        
        valid_numeric = new_numeric_np[valid_mask]
        valid_cat = np.array(new_cat_rows, dtype=object)[valid_mask]
        
        if len(valid_numeric) > 0:
            df_num = pd.DataFrame(valid_numeric, columns=numeric_columns)
            df_cat = pd.DataFrame(valid_cat, columns=categorical_columns)
            batch_df = pd.concat([df_num, df_cat], axis=1)[feature_cols]
            
            needed = target_n - generated_count
            if len(batch_df) > needed:
                batch_df = batch_df.iloc[:needed]
                
            synthetic_dfs.append(batch_df)
            generated_count += len(batch_df)
            print(f"Generated {generated_count} / {target_n}...")
            
    final_df = pd.concat(synthetic_dfs, ignore_index=True)
    
    for col in numeric_columns:
        values = normal[col].dropna().values
        if np.all(np.isclose(values, np.round(values))):
            final_df[col] = np.rint(final_df[col].astype(float)).clip(mins[col], maxs[col]).astype(normal[col].dtype)
            
    final_df.insert(0, "id", np.arange(target_n))
    final_df.to_csv(output_path, index=False)
    print(f"Saved {output_path} with shape {final_df.shape}\n")

if __name__ == '__main__':
    create_safe_smote(
        train_path='../train-NSL-KDD.csv', 
        output_path='submission-NSL-KDD.csv', 
        target_n=40000, 
        dataset_name='NSL-KDD'
    )
