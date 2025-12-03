import numpy as np
import torch
from torch.utils.data import Dataset


# TODO Use torch insted of numpy :(
class TimeStationDataset(Dataset):
    def __init__(self, data_raw, arr_all, dep_all, nb_time_steps, horizon=1, feat_idxs=None, normalize=True):
        """
        data_raw: np.array (T_all, N, F_raw) -- features (may include historical-derived features but not target at t)
        arr_all: np.array (T_all, N) -- arrivals time series (labels)
        dep_all: np.array (T_all, N) -- departures time series (labels)
        nb_time_steps: history length (timesteps)
        horizon: number of future steps to predict
        feat_idxs: list of indices in data_raw to use as input features; if None use all channels
        """
        self.nb_time_steps_back = nb_time_steps
        self.horizon = horizon
        if feat_idxs is None:
            X = data_raw.astype(np.float32)  # (T_all, N, F_raw)
        else:
            X = data_raw[..., list(feat_idxs)].astype(np.float32)  # (T_all, N, d)
        T_all, N, d = X.shape

        # Normalization per-station-feature (try if this helps)
        if normalize:
            self.mean = X.mean(axis=0, keepdims=True)  # (1,N,d)
            self.std = X.std(axis=0, keepdims=True)
            self.std[self.std == 0] = 1.0
            X = (X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        self.data = X  # (T_all, N, d)
        self.arr = arr_all.astype(np.float32)  # (T_all, N)
        self.dep = dep_all.astype(np.float32)

        # Build valid indices where we can take M history and horizon target
        self.indices = []
        for t in range(nb_time_steps, T_all - horizon + 1):
            self.indices.append(t)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x_hist = self.data[t - self.nb_time_steps_back : t]           # (nb_time_steps, N, d)
        # targets for horizon (t .. t+horizon-1)
        arr_target = self.arr[t : t + self.horizon]   # (horizon, N)
        dep_target = self.dep[t : t + self.horizon]   # (horizon, N)
        # transpose to (N, horizon)
        arr_target = arr_target.T.copy()  # (N, horizon)
        dep_target = dep_target.T.copy()  # (N, horizon)

        # convert to torch, and reorder to model's expected shape (B will be stacked by DataLoader)
        x = torch.from_numpy(x_hist).float()         # (nb_time_steps, N, d)
        return x, torch.from_numpy(arr_target).float(), torch.from_numpy(dep_target).float()