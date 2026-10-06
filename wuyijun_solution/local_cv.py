import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
import time

def evaluate_unsw():
    train_path = '../train-UNSW-NB15.csv'
    df = pd.read_csv(train_path)
    label_col = 'is_anomaly'
    
    # 80/20 split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_col])
    
    # Normal data for generation
    normal_train = train_df[train_df[label_col] == 0].copy().reset_index(drop=True)
    
    # Validation data
    X_val = val_df.drop(columns=[label_col])
    y_val = val_df[label_col].values
    
    feature_cols = [c for c in df.columns if c != label_col]
    
    categorical_columns = [
        c for c in feature_cols
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    
    print(f"Train normal size: {len(normal_train)}, Val size: {len(X_val)} (Anomalies: {y_val.sum()})")
    
    # We will test a generator function
    def generate_synthetic(normal_data, target_n, bootstrap_frac=0.8, alpha_max=0.1):
        rng = np.random.default_rng(42)
        numeric_cols = [c for c in feature_cols if c not in categorical_columns]
        
        mins = normal_data[numeric_cols].min()
        maxs = normal_data[numeric_cols].max()
        
        grouped = normal_data.groupby(categorical_columns)
        group_sizes = grouped.size()
        group_probs = group_sizes / group_sizes.sum()
        
        synthetic_rows = []
        batch_size = target_n
        
        chosen_groups = rng.choice(group_sizes.index, size=batch_size, p=group_probs)
        
        new_num_list = []
        new_cat_list = []
        
        for g in chosen_groups:
            group_data = grouped.get_group(g)
            
            if len(group_data) == 1 or rng.uniform() < bootstrap_frac:
                idx = rng.integers(0, len(group_data))
                row = group_data.iloc[idx]
                new_num_list.append(row[numeric_cols].values)
                new_cat_list.append(row[categorical_columns].values)
            else:
                idx1, idx2 = rng.integers(0, len(group_data), size=2)
                anchor = group_data.iloc[idx1]
                other = group_data.iloc[idx2]
                alpha = rng.uniform(0, alpha_max)
                new_num = anchor[numeric_cols].values + alpha * (other[numeric_cols].values - anchor[numeric_cols].values)
                new_num_list.append(new_num)
                new_cat_list.append(anchor[categorical_columns].values)
                
        new_num_np = np.clip(np.array(new_num_list), mins.values, maxs.values)
        
        df_num = pd.DataFrame(new_num_np, columns=numeric_cols)
        df_cat = pd.DataFrame(new_cat_list, columns=categorical_columns)
        
        syn_df = pd.concat([df_num, df_cat], axis=1)[feature_cols]
        
        # Int fix
        for col in numeric_cols:
            if np.all(np.isclose(normal_data[col].dropna().values, np.round(normal_data[col].dropna().values))):
                syn_df[col] = np.rint(syn_df[col].astype(float)).clip(mins[col], maxs[col]).astype(normal_data[col].dtype)
                
        return syn_df
        
    print("Generating synthetic data...")
    syn_df = generate_synthetic(normal_train, 70000, bootstrap_frac=0.9, alpha_max=0.05)
    
    # Preprocessing: OHE for categories
    print("Encoding features for evaluation...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    encoder.fit(df[categorical_columns])
    
    def process_features(data_df):
        cat_encoded = encoder.transform(data_df[categorical_columns])
        num_data = data_df.drop(columns=categorical_columns).values
        return np.hstack((num_data, cat_encoded))
        
    X_syn_proc = process_features(syn_df)
    X_val_proc = process_features(X_val)
    
    print("Training ECOD...")
    ecod = ECOD()
    ecod.fit(X_syn_proc)
    y_val_scores_ecod = ecod.decision_function(X_val_proc)
    auprc_ecod = average_precision_score(y_val, y_val_scores_ecod)
    print(f"ECOD AUPRC: {auprc_ecod:.5f}")
    
    print("Training IForest (10 seeds)...")
    iforest_scores = np.zeros_like(y_val_scores_ecod)
    for seed in range(10):
        clf = IForest(random_state=seed, n_estimators=500)
        clf.fit(X_syn_proc)
        iforest_scores += clf.decision_function(X_val_proc)
    
    iforest_scores /= 10
    auprc_iforest = average_precision_score(y_val, iforest_scores)
    print(f"IForest AUPRC: {auprc_iforest:.5f}")
    
    print(f"Average AUPRC: {(auprc_ecod + auprc_iforest) / 2:.5f}")

if __name__ == "__main__":
    evaluate_unsw()
