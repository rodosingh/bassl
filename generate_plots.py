import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Load data
with open("/ssd_scratch/cvit/rodosingh/data/24/movienet/pred_metadata.pkl", "rb") as f:
   pred_meta = pickle.load(f)

pred = np.load("/ssd_scratch/cvit/rodosingh/data/24/movienet/pred.npy")

seasons = [f"S%02d"%i for i in range(2, 9)]
episodes = [f"E%02d"%i for i in range(1, 25)]

# Plot
os.makedirs("/ssd_scratch/cvit/rodosingh/data/24/movienet/plots", exist_ok=True)
cnt = 0
for i in seasons:
    for j in episodes:
        ep_name = i + j
        len_ep = len([k for k in pred_meta['vid'] if k == ep_name])
        plt.figure(figsize=(10,5), dpi=300)
        plt.plot(np.arange(len_ep), pred[cnt:cnt+len_ep])
        plt.savefig(f"/ssd_scratch/cvit/rodosingh/data/24/movienet/plots/{ep_name}.png")
        cnt += len_ep