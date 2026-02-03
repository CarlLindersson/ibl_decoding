#%% Import necessary libraries
from one.api import ONE
from brainbox.io.one import SpikeSortingLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import importlib
import sys
import os
if 'src.functions' in sys.modules:
    importlib.reload(sys.modules['src.functions'])
from src.functions import compute_firing_rates, plot_example_neuron

os.chdir(r'c:\Users\carll\OneDrive\Skrivbord\Oxford\Tutoring\ibl_decoding')

#%% Set up ONE instance
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password='international')

#%% Find sessions with recordings from motor areas
region = "MOp"          # Allen CCF acronym (e.g., MOp, MOs)
need_ds = "spikes.times.npy"  # optional: only keep insertions that have this dataset

pids = one.search_insertions(atlas_acronym=region,
                            datasets=need_ds,
                            project="brainwide")   # project optional

# convert each pid to (eid, probe_label)
eid_probe = [one.pid2eid(pid) for pid in pids]
eids = [eid for eid, probe in eid_probe]

#pid = 'da8dfec1-d265-44e8-84ce-6ae9c109b8bd'
#id = one.pid2eid(pid)[0]

#print(f"{len(pids)} insertions, {len(set(eids))} sessions")
#print(eid_probe[:5]) # example (eid, probe)

#%% Select session and participant  

# Get the session experiment identification (EID) from PID
eids, probe= eid_probe[1]
eid = str(eids) # example eid
pid = one.eid2pid(eid)
# Example EID with data from motor area
print(f'Selected PID: {pid}, EID: {eid}')

#%% Get behavioral data from the session
 
# Load trials to get response times
trials = one.load_object(eid, 'trials')

#%%  Filter out NaN values (e.g., no-go trials)
# Build ONE mask for "go" trials (keep alignment across all fields)
go_mask = ~np.isnan(trials['response_times']) # removes no-go

# Apply the SAME mask to everything trial-aligned
response_times = trials['response_times'][go_mask]
choices = trials['choice'][go_mask]

# Keep original trial indices too (very useful for debugging / merging later)
trial_ids = np.flatnonzero(go_mask)

#response_times = response_times[~np.isnan(response_times)]
#print(f"Valid response_times after filtering NaNs: {len(response_times)}")

#%% Load spike sorting data from specific session (eid) and probe (probe00) and spike_sorter (pykilosort)
sl = SpikeSortingLoader(eid=eid, pname='probe00', one=one)
spikes, clusters, channels = sl.load_spike_sorting(spike_sorter='pykilosort')
clusters = sl.merge_clusters(spikes, clusters, channels)  # adds brain area acronyms/atlas IDs if available
print(clusters.keys())

# Filter clusters to only those with spikes in any 200ms pre-choice window
win = [0.2, 0.0]  # 200ms window from -400ms to -200ms before response
mask = np.zeros(len(spikes['times']), dtype=bool)
for rt in response_times:
    mask |= (spikes['times'] >= rt - win[0]) & (spikes['times'] < rt - win[1]) 
active_cluster_ids = np.unique(spikes['clusters'][mask]) #
print(f"Active clusters (with spikes in pre-choice windows): {len(active_cluster_ids)}")

#%% Plot PETHs and raster for example clusters (200ms pre-choice)
#plot_example_neuron(spikes, clusters, response_times, active_cluster_ids[2:3])

#%% Calculate firing rates in -200ms to response time window for all active clusters
firing_rates = compute_firing_rates(spikes, response_times, active_cluster_ids, win)
print(f"Firing rates shape: {firing_rates.shape}")  # Should be (num_response_times, num_active_clusters)

#%% Create a DataFrame and save to Excel 

# Create DataFrame with UUIDs as columns
#columns = [clusters['uuids'].iloc[idx] for idx in active_cluster_ids]
#df = pd.DataFrame(firing_rates, columns=columns, index=range(len(response_times)))

# Create MultiIndex columns with (UUID, brain region)
cid_to_idx = {cid: i for i, cid in enumerate(clusters["cluster_id"])}
uuids = [clusters["uuids"][cid_to_idx[cid]] for cid in active_cluster_ids]
acronyms = [clusters["acronym"][cid_to_idx[cid]] for cid in active_cluster_ids]
neuron_cols = pd.MultiIndex.from_arrays([uuids, acronyms], names=["uuid", "acronym"])
df = pd.DataFrame(firing_rates, columns=neuron_cols, index=range(len(response_times)))

# Add aligned trial info
df.insert(0, 'trial_id', trial_ids)   # original trial index in the session
df.insert(1, 'choice', choices)       # -1 left, +1 right (0 excluded)

# Save to Excel
df.to_excel(r'data/firing_rates.xlsx')
print("Firing rates saved to firing_rates.xlsx")