from difflib import SequenceMatcher
import numpy as np
import pandas as pd


def remove_similar_rows(column_as_df, df, threshold=0.9):
    # Compute similarity scores for each pair of rows
    similarity_scores = {}
    for i, row in column_as_df.iterrows():
        for j, other_row in column_as_df.iterrows():
            if i >= j:
                continue
            score = SequenceMatcher(None, row, other_row).ratio()
            if score >= threshold:
                similarity_scores[(i, j)] = score
    
    # Identify rows to remove
    rows_to_remove = []
    for (i, j), score in similarity_scores.items():
        if i not in rows_to_remove and j not in rows_to_remove:
            rows_to_remove.append(j if df.index[i] < df.index[j] else i)
    
    # Remove rows and return modified DataFrame
    return df.drop(rows_to_remove)
