import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split
from time_station_dataset import TimeStationDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from utils import evaluate, build_trip_adjacency

from st_gnn import ST_GNN
 
# Device selection early so other code can use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# df = pd.read_csv("Data/merged2018v2.csv", parse_dates=['start_date', 'end_date'])
df = pd.read_csv("Data/OD_2018-04.csv", parse_dates=['start_date', 'end_date'])

df['start_date'] = pd.to_datetime(df['start_date'])
# print(df.get(0))
df_grouped_start = df.groupby([ pd.Grouper(key="start_date", freq="15min"),
                 pd.Grouper('start_station_code')
                 ]).size().reset_index(name="departures")
df_grouped_start.columns = ["time", "station", "departures"]

# print(df_grouped_start.sort_values(by=['departures']))


df_grouped_end = df.groupby([ pd.Grouper(key="end_date", freq="15min"),
                 pd.Grouper('start_station_code')
                 ]).size().reset_index(name="arrivals")
df_grouped_end.columns = ["time", "station", "arrivals"]

# We aggregate arrivals and departures per station per time slot (currently 15min))
# For each station code we have a time series of arrivals and departures
# Then we can build a feature matrix X where each node (station)  per time slot (currently only using adjacency Matrix)

# We have a Matrix with shape time x stations x feature tensor (departures, arrivals, adjacency info for now so not used as global)

print("Building data tensor")
# unified time and station indices
all_times = sorted(set(df_grouped_start["time"]).union(df_grouped_end["time"]))
all_stations = sorted(set(df_grouped_start["station"]).union(df_grouped_end["station"]))
time_to_idx = {t: i for i, t in enumerate(all_times)}
station_to_idx = {s: i for i, s in enumerate(all_stations)}

nb_times = len(all_times)
nb_stations = len(all_stations)
print(f"Number of time slots: {nb_times}, number of stations: {nb_stations}")
# nb_features = 4 (from temporal day of week and hour as cyclic conversion need cos and sin) 
# # TODO SIG Add dimensions if we use other features than time, current station and stations adjacency
print("Building temporal features")
temporal_feats = np.zeros((nb_times, 4), dtype=np.float32)
# Use cos and sin to project on circle as dow and hour is cyclic (suggested by Claude)
for i, ts in enumerate(all_times):
    hour = ts.hour
    dow = ts.dayofweek
    temporal_feats[i, 0] = np.sin(2 * np.pi * hour / 24)
    temporal_feats[i, 1] = np.cos(2 * np.pi * hour / 24)
    temporal_feats[i, 2] = np.sin(2 * np.pi * dow / 7)
    temporal_feats[i, 3] = np.cos(2 * np.pi * dow / 7)

# Replicate to all stations: (nb_times, 4) -> (nb_times, nb_stations, 4)
data = np.repeat(temporal_feats[:, np.newaxis, :], nb_stations, axis=1)

# Initialize label arrays (departures, arrivals)
departs_all = np.zeros((nb_times, nb_stations), dtype=np.float32)
arrivals_all = np.zeros((nb_times, nb_stations), dtype=np.float32)


print("Filling departures")
# fill departures
for _, row in df_grouped_start.iterrows():
    ti = time_to_idx[row["time"]]
    si = station_to_idx[row["station"]]
    departs_all[ti, si] = float(row["departures"])



print("Filling arrivals")
# fill arrivals
for _, row in df_grouped_end.iterrows():
    ti = time_to_idx[row["time"]]
    si = station_to_idx[row["station"]]
    arrivals_all[ti, si] = float(row["arrivals"])


# tensor for downstream models (nb_times x nb_stations x features)

print("Creating dataset")
ds_all_time_data = TimeStationDataset(data, arrivals_all, departs_all, nb_time_steps=12, horizon=1, feat_idxs=None, normalize=True)


# Replace load of station adjacency from csv as the matrix is not the same size
# TODO Would be nice to get it from csv and filter
print("Building adjacency matrix from trip data")
# Convert station codes to strings to match CSV format
station_codes_str = [str(s) for s in all_stations]
adjacency_matrix = build_trip_adjacency("Data/OD_2018-04.csv", station_codes_str)
print(f"Adjacency matrix shape: {adjacency_matrix.shape}")


print("Build model")

# Set parameters (to adjust later)
model = ST_GNN(in_features=4, hidden=64, max_distance_hop=2,
                max_history_slots=12, horizon=1) # in_feat=4 (temporal features)

model.to(device)


####### Create DataLoader
print("Build Data loader")

n_total = len(ds_all_time_data)
train_frac = 0.8
n_train = int(train_frac * n_total)
train_idx = list(range(0, n_train))            # train on earliest time windows
test_idx  = list(range(n_train, n_total))      # test on later windows

train_set = Subset(ds_all_time_data, train_idx)
test_set  = Subset(ds_all_time_data, test_idx)


# Not usre about batch_size
batch_size=64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False)
epochs = 10
best_val = float('inf')
ckpt_path = 'best_model.pt'

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss() # Mean Squared error (squarred L2 norm)

print("Start training")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for data_batch, arrivals_batch, departures_batch in train_loader:
        print( f"Starting new batch for epoch {epoch:03d}")
        data_batch = data_batch.to(device)
        arrivals_batch = arrivals_batch.to(device)
        departures_batch = departures_batch.to(device)
        adj_device = adjacency_matrix.to(device).float()

        optimizer.zero_grad()
        pred_arr, pred_dep = model(data_batch, adj_device)
        loss = loss_fn(pred_arr, arrivals_batch) + loss_fn(pred_dep, departures_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    train_loss = running_loss / len(train_loader)
    val_rmse_arr, val_rmse_dep = evaluate(model, test_loader, adjacency_matrix, device)
    val_rmse = (val_rmse_arr + val_rmse_dep) / 2.0

    print(f"Epoch {epoch:03d}  Train loss: {train_loss:.4f}  Val RMSE arr:{val_rmse_arr:.4f} dep:{val_rmse_dep:.4f}")
    
    # check if this is the best model (by average RMSE)
    if val_rmse < best_val:
        best_val = val_rmse
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_rmse': val_rmse}, ckpt_path)

print("Training finished. Best avg RMSE:", best_val)

# could use best_model.pt
SAVED_MODEL_PATH = './tranied_model.pth' 
torch.save(model.state_dict(), SAVED_MODEL_PATH)

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()
test_rmse_arr, test_rmse_dep = evaluate(model, test_loader, adjacency_matrix, device)
print(f"Final test RMSE — arrivals: {test_rmse_arr:.4f}, departures: {test_rmse_dep:.4f}")



print("Training complete")







# Build adjacency matrix 

# For each time slot add arrivals aand departure per range : 0-3km, 3-6km, 6+


# M Graph named Gt = situtation at time T
# Gt = ( Xt Feature Matrix at time T for each node and Adjacency Matrix W = Euclidian distane between nodes)
# Each nodes =>  Xt = difference between entries and exits at time T
#                W


# (one entry per time step => one entry per station with entry variation))

# 𝑋(𝑒)𝑡= 𝐷(𝑒)−1 𝑊(𝑒) 𝑋𝑡  ? If yes we should calculate hop between nodes



# Temporal between step = 𝑌𝑣𝑡=𝑅𝑒𝐿𝑈( Σ𝑍𝑘𝑡−1𝑘=1Ẏ𝑣𝑘 ) => Zk = trainable parameter

# Ẏ𝑣𝑘 =𝑅𝑒𝐿𝑈(𝑍𝑠𝑡(𝑋𝑣𝑡 || Ẋ𝑣𝑡 || 𝑌𝑣𝑡)) => (Zst) learnable parameter






# On perd l'information de départ de l'une vers une autre
# Option Ajouter les départs dans le time slot précédent ?