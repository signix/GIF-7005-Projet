import torch
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List


def evaluate(model, loader, adj_matrix, device):
    model.eval()
    sum_sq_arr = 0.0
    sum_sq_dep = 0.0
    total_samples = 0
    adj_device = adj_matrix.to(device).float()
    with torch.no_grad():
        for x_batch, arr_batch, dep_batch in loader:
            x_batch = x_batch.to(device)         # (B, M, N, d)
            arr_batch = arr_batch.to(device)     # (B, N, horizon)
            dep_batch = dep_batch.to(device)
            B = x_batch.shape[0]
            pred_arr, pred_dep = model(x_batch, adj_device)
            # accumulate squared errors
            sum_sq_arr += ((pred_arr - arr_batch) ** 2).sum().item()
            sum_sq_dep += ((pred_dep - dep_batch) ** 2).sum().item()
            total_samples += B * arr_batch.shape[1] * arr_batch.shape[2]  # B * N * horizon
    rmse_arr = math.sqrt(sum_sq_arr / total_samples)
    rmse_dep = math.sqrt(sum_sq_dep / total_samples)
    return rmse_arr, rmse_dep


def build_trip_adjacency(trips_csv: str, station_codes: List[str], chunksize: int = 200000) -> torch.Tensor:
    """
    Build adjacency matrix from trip data (origin->destination counts).
    
    Args:
        trips_csv: path to CSV with start/end station columns
        station_codes: ordered list of station codes to match
        chunksize: chunk size for reading large CSV files
    
    Returns:
        adjacency matrix (N, N) row-normalized and symmetrized
    """
    code_to_idx = {c: i for i, c in enumerate(station_codes)}
    N = len(station_codes)
    counts = defaultdict(float)

    # Detect column names
    it = pd.read_csv(trips_csv, nrows=0)
    cols = set(it.columns.tolist())
    possible_start = ['start_station_code', 'start_station_id', 'start_station']
    possible_end = ['end_station_code', 'end_station_id', 'end_station']
    start_col = next((c for c in possible_start if c in cols), None)
    end_col = next((c for c in possible_end if c in cols), None)
    if start_col is None or end_col is None:
        raise RuntimeError(f"Could not find start/end station columns in {trips_csv}")

    # Read in chunks and aggregate counts
    for chunk in pd.read_csv(trips_csv, usecols=[start_col, end_col], chunksize=chunksize):
        s = chunk[start_col].astype(str)
        e = chunk[end_col].astype(str)
        for a, b in zip(s, e):
            if a in code_to_idx and b in code_to_idx:
                counts[(code_to_idx[a], code_to_idx[b])] += 1.0

    # Build and normalize adjacency matrix
    adj_matrix = torch.zeros((N, N), dtype=torch.float32)
    for (i, j), v in counts.items():
        adj_matrix[i, j] = v

    # Row-normalize
    rowsum = adj_matrix.sum(dim=1, keepdim=True)
    rowsum[rowsum == 0] = 1.0
    adj_matrix = adj_matrix / rowsum
    
    # Symmetrize (average)
    A_sym = (adj_matrix + adj_matrix.t()) / 2.0
    
    # Re-normalize
    rowsum = A_sym.sum(dim=1, keepdim=True)
    rowsum[rowsum == 0] = 1.0
    A_sym = A_sym / rowsum
    return A_sym