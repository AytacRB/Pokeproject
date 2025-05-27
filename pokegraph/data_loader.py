import pandas as pd
import ast

"""
Loads and processes dataframe.
Args:
    path: String reference to input data
    hr: If True Limits sample to high-ranking battles
    cutoff: specifies high-rank cutoff

Returns:
    df: processed tabular data
"""

def load_and_preprocess_data(path, hr=False, cutoff=1400):
     # Preprocess csv file to ensure proper loading
    df = pd.read_csv(path)
    df['players'] = df['players'].apply(ast.literal_eval)
    df['p1'] = df['p1'].apply(ast.literal_eval)
    df['p2'] = df['p2'].apply(ast.literal_eval)
    df = df.drop_duplicates(subset=['id', 'uploadtime'])
    # Optional low-rank filter

    if hr:
        df = df[df['rating'] >= cutoff]

    return df