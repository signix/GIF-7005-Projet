"""

Model summary :
- Spatial aggregation :
  X^(e)_t = D(e)^{-1} W^(e) X_t => Ẋ_vt = sum_e Z_e( X^(e)_t )
- Temporal aggregation 
  Y_vt = ReLU( sum_{k=1 to t-1} T_k( Ẏ_vk ) )
- Spatio-temporal step
  Ẏ_vt = ReLU( Z_st( concat( X_vt || Ẋ_vt || Y_vt ) ) )
- Final step
   F_v = Z_F( concat_t Ẏ_vt )
   => MLP.

"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
from collections import defaultdict


def compute_hop_matrices(adj: torch.Tensor, E: int) -> List[torch.Tensor]:
    """Compute binary matrices W^(e) where W^(e)[i,j]=1 iff shortest path distance(i,j)==e.

    adj: (N,N) adjacency (weights >0 considered edges). Undirected assumed.
    Returns list of length E of float tensors shape (E, N, N).
    """
    assert adj.dim() == 2 and adj.shape[0] == adj.shape[1]
    N = adj.shape[0]
    A = (adj > 0).to(torch.uint8)

    # We'll run BFS from each source node (simple and clear, ok for moderate N)
    hops = torch.zeros((E, N, N), dtype=torch.float32)

    for src in range(N):
        visited = {src}
        frontier = {src}
        for e in range(1, E + 1):
            next_frontier = set()
            for u in frontier:
                nbrs = torch.nonzero(A[u]).squeeze(1).tolist()
                for v in nbrs:
                    if v not in visited:
                        next_frontier.add(v)
                        visited.add(v)
            for v in next_frontier:
                hops[e - 1, src, v] = 1.0
            frontier = next_frontier

    # Row-normalize each W^(e): D(e)^{-1} W^(e)
    for e in range(E):
        row_sum = hops[e].sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        hops[e] = hops[e] / row_sum

    return [hops[e] for e in range(E)]

# ChatGPT Suggestion. Do we need it ? I don't think so
# def build_trip_adjacency(trips_csv: str, station_codes: List[str], chunksize: int = 200000) -> torch.Tensor:
#     """
#     Adjacency based on trips between stations from trips CSV.
#     """
#     code_to_idx = {c: i for i, c in enumerate(station_codes)}
#     N = len(station_codes)
#     counts = defaultdict(float)

#     usecols = None
#     # Common column names to try
#     possible_start = ['start_station_code', 'start_station_id', 'start_station']
#     possible_end = ['end_station_code', 'end_station_id', 'end_station']

#     # We'll read first chunk to detect columns
#     it = pd.read_csv(trips_csv, nrows=0)
#     cols = set(it.columns.tolist())
#     start_col = next((c for c in possible_start if c in cols), None)
#     end_col = next((c for c in possible_end if c in cols), None)
#     if start_col is None or end_col is None:
#         raise RuntimeError(f"Could not find start/end station columns in {trips_csv}")

#     for chunk in pd.read_csv(trips_csv, usecols=[start_col, end_col], chunksize=chunksize):
#         # coerce to str for matching codes
#         s = chunk[start_col].astype(str)
#         e = chunk[end_col].astype(str)
#         for a, b in zip(s, e):
#             if a in code_to_idx and b in code_to_idx:
#                 counts[(code_to_idx[a], code_to_idx[b])] += 1.0

#     adjacenccy_matrix = torch.zeros((N, N), dtype=torch.float32)
#     for (i, j), v in counts.items():
#         adjacenccy_matrix[i, j] = v

#     # row-normalize (transition probabilities)
#     rowsum = adjacenccy_matrix.sum(dim=1, keepdim=True)
#     rowsum[rowsum == 0] = 1.0
#     adjacenccy_matrix = adjacenccy_matrix / rowsum
#     # symmetrize to make undirected adjacency (average)
#     A_sym = (adjacenccy_matrix + adjacenccy_matrix.t()) / 2.0
#     # re-normalize
#     rowsum = A_sym.sum(dim=1, keepdim=True)
#     rowsum[rowsum == 0] = 1.0
#     A_sym = A_sym / rowsum
#     return A_sym


class ST_GNN(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, max_distance_hop: int = 2,
                 max_history_slots: int = 12, horizon: int = 1):
        """SST-GNN model.

        Args:
            in_features: number of features per time/station (currently none)
            hidden: hidden dimension for spatial/temporal embeddings
            max_distance_hop: maximum hop distance to aggregate spatially (to limit the number of hops)
            max_history_slots: number of historical timestamps to use (e.g., 12 for 3 hour at 15-min intervals)
            horizon: number of future steps to predict (default 1, 4 would give next hour with 15min interval)
        """
        super(ST_GNN, self).__init__()

        ############ Build layers ############
        self.in_features = in_features
        self.hidden = hidden
        self.max_distance_hop = max_distance_hop
        self.max_history_slots = max_history_slots
        self.horizon = horizon
        # self.adjacency_matrix = adjacency_matrix

        ############ Build layers ############
        ############ Group data to reduce size by keeping only a few hopes and previous time slots ############
        # TODO Keep same month on other years to capture seasonal patterns Not now as we only use 2018

        # Filter by distance 
        # Spatial hop transforms Z^(e): maps input features d -> hidden
        # These will be applied to the travel adjacency matrix `G` (primary spatial matrix).
        self.spatial_linears = nn.ModuleList([nn.Linear(in_features, hidden, bias=False)
                                               for _ in range(max_distance_hop)])

        # Temporal lag transforms T_k (for k-th lag): maps hidden -> hidden (no bias as in paper)
        # We create M temporal transforms (used up to t-1 lags)
        self.temporal_linears = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(max_history_slots)])

        # Spatio-temporal fusion Z_st: concat(X_vt (d), Ẋ_vt (hidden), Y_vt (hidden)) -> hidden
        self.st_fusion = nn.Linear(in_features + hidden + hidden, hidden)

        # Final aggregator Z_F: combine concatenated spatio-temporal embeddings across M timesteps -> hidden
        self.zf = nn.Linear(hidden * max_history_slots, hidden)

        # Prediction MLP (two-layer) to predict horizon values per node.
        # We predict two labels per horizon: arrivals and departures.
        self.predictor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden, max(hidden // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden // 2, 8), 2 * horizon),
        )


        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        precomputed_hops: Optional[List[torch.Tensor]] = None,
    ):
        """Forward pass.

        Args:
            x: (B, T, N, d) input sequence (historical), T >= M (we use the last M timestamps)
            adjacency_matrix: (N,N) travel-based adjacency matrix (spatial matrix).
            precomputed_hops: optional list of length E of (N,N) normalized hop matrices for `G`.

        Returns:
            (arrivals, departures): tuple of tensors each shaped (B, N, horizon)
        """
        sample_size, nb_time_steps, nb_of_stations, nb_features = x.shape # TODO Use external features if any

        # Work with the last M timestamps
        assert nb_time_steps >= self.max_history_slots, "Input nb_time_steps must be >= max_history_slots"
        x_hist = x[:, -self.max_history_slots :, :, :] 

        # Prepare hop matrices for adjacency matrix (required)
        if precomputed_hops is None:
            hop_mats = compute_hop_matrices(adjacency_matrix, self.max_distance_hop)  # list length E of (N,N)
        else:
            hop_mats = precomputed_hops

        # Move hop mats to same device and dtype
        hop_mats = [hm.to(x.device).to(x.dtype) for hm in hop_mats]

        # We'll compute spatio-temporal embeddings sequentially for t=0..M-1
        Y_prev = []  # list of Ẏ_t tensors (B, N, hidden)

        for t in range(self.max_history_slots):
            X_t = x_hist[:, t, :, :]  # (B, N, d)

            # Spatial aggregation across hops for adjacency `adj`
            spatial_sum = 0.0
            for e in range(self.max_distance_hop):
                W_e = hop_mats[e]  # (N,N) normalized
                W_e_exp = W_e.unsqueeze(0).expand(sample_size, -1, -1)  # (B,N,N)
                # X_t is (B,N,d) -> bmm -> (B,N,d)
                X_e = torch.bmm(W_e_exp, X_t)
                # Apply linear Z_e: elementwise across nodes
                Z_e = self.spatial_linears[e](X_e)  # (B,N,hidden)
                spatial_sum = spatial_sum + Z_e if not isinstance(spatial_sum, float) else Z_e

            X_hat = spatial_sum  # Ẋ_t (B,N,hidden)

            # Temporal aggregation: sum of transformed previous Ẏ (lags)
            temporal_sum = torch.zeros((sample_size, nb_of_stations, self.hidden), device=x.device, dtype=x.dtype)
            for k in range(1, min(t + 1, self.max_history_slots) + 0):
                # use lag k => transform Y_prev[-k]
                if k <= len(Y_prev):
                    Y_lag = Y_prev[-k]  # (B,N,hidden)
                    T_k = self.temporal_linears[k - 1](Y_lag)
                    temporal_sum = temporal_sum + T_k

            Y_t = self.activation(temporal_sum)  # (B,N,hidden)

            # Spatio-temporal embedding Ẏ_t = ReLU( Z_st( concat(X_t || X_hat || Y_t) ) )
            concat_in = torch.cat([X_t, X_hat, Y_t], dim=-1)  # (B,N, d+hidden+hidden)
            Yst = self.st_fusion(concat_in)  # (B,N,hidden)
            Yst = self.activation(Yst)

            Y_prev.append(Yst)

        # Concatenate embeddings over M timestamps: (B,N, hidden*M)
        Y_cat = torch.cat(Y_prev, dim=-1)

        # Final embedding Z_F
        F = self.zf(Y_cat)  # (B,N,hidden)
        F = self.activation(F)

        # Prediction per node -> outputs 2*horizon channels (arrivals, departures)
        pred = self.predictor(F)  # (B,N, 2*horizon)
        pred = pred.view(sample_size, nb_of_stations, self.horizon, 2)  # (B,N,horizon,2)
        arrivals = pred[..., 0].contiguous()  # (B,N,horizon)
        departures = pred[..., 1].contiguous()  # (B,N,horizon)

        return arrivals, departures


if __name__ == "__main__":
    # Smoke test: small synthetic example
    B = 2
    T = 12
    N = 8
    d = 2
    hidden = 32
    E = 2
    M = 12
    horizon = 1

    model = SST_GNN(in_features=d, hidden=hidden, max_distance_hop=E, max_history_slots=M, horizon=horizon)
    model.eval()

    x = torch.randn(B, T, N, d)

    # create simple undirected adjacency (random geometric-like)
    coords = torch.randn(N, 2)
    D = torch.cdist(coords, coords)
    A = (D < 1.0).float()

    with torch.no_grad():
        arrivals, departures = model(x, A)
    print("input", x.shape)
    print("adj/G", A.shape)
    print("arrivals", arrivals.shape)
    print("departures", departures.shape)
