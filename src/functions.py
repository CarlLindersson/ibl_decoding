
import numpy as np
import matplotlib.pyplot as plt
from brainbox.singlecell import calculate_peths
import re
from iblatlas.regions import BrainRegions

def compute_firing_rates(spikes, response_times, active_cluster_ids, win=[0.4, 0.2]):
    """
    Compute mean firing rates for each cluster in the 200ms window before each response time.

    Parameters
    ----------
    spikes : dict
        Dictionary containing 'times' and 'clusters' arrays.
    response_times : np.ndarray
        Array of response times.
    active_cluster_ids : np.ndarray
        Array of active cluster IDs.

    Returns
    -------
    firing_rates : np.ndarray
        2D array of shape (num_response_times, num_active_clusters) with mean firing rates.
    """

    # sort response times
    rt_order = np.argsort(response_times)
    rt = response_times[rt_order]

    # sort spikes by time
    spk_order = np.argsort(spikes['times'])
    st = spikes['times'][spk_order]
    sc = spikes['clusters'][spk_order]

    active_cluster_ids = np.sort(np.asarray(active_cluster_ids))

    # next response time for each spike (can be len(rt) if spike is after last rt)
    trial_idx = np.searchsorted(rt, st, side="left")

    # IMPORTANT: drop spikes after the last response time BEFORE indexing rt[trial_idx]
    valid_trial = trial_idx < len(rt)
    trial_idx_v = trial_idx[valid_trial]
    st_v = st[valid_trial]
    sc_v = sc[valid_trial]

    # now safe to index rt[trial_idx_v]
    valid_time = (st_v >= (rt[trial_idx_v] - win[0])) & (st_v < (rt[trial_idx_v] - win[1]))  # Adjusted for -400 to -200 ms

    trial_idx_v = trial_idx_v[valid_time]
    cids = sc_v[valid_time]

    # map cluster IDs -> columns
    cols = np.searchsorted(active_cluster_ids, cids)
    ok = (cols < len(active_cluster_ids)) & (active_cluster_ids[cols] == cids)

    trial_idx_v = trial_idx_v[ok]
    cols = cols[ok]

    # accumulate spike counts
    counts = np.zeros((len(rt), len(active_cluster_ids)), dtype=np.int32)
    np.add.at(counts, (trial_idx_v, cols), 1)

    firing_rates_sorted = counts / 0.2  # Adjusted for the new window size

    # unsort to original response_times order
    firing_rates = np.empty_like(firing_rates_sorted, dtype=np.float32)
    firing_rates[rt_order, :] = firing_rates_sorted

    return firing_rates

# plot PETHs and rasters for example neuron (200ms pre-choice)
def plot_example_neuron(spikes, clusters, response_times, active_cluster_ids):
    """Plot PETHs and raster for example neurons/clusters (200ms pre-choice)."""

    example_cluster_ids = active_cluster_ids  # These are indices

    # set up figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Top: PETH
    # Compute PETHs for 200ms pre-choice (aligned to response_times)
    peths, _ = calculate_peths(spikes['times'], spikes['clusters'], example_cluster_ids, 
                            align_times=response_times,
                            pre_time=0.2, post_time=0.0, bin_size=0.01, smoothing=0.025)
    for idx, cluster_id in enumerate(example_cluster_ids):
        axs[0].plot(peths['tscale'], peths['means'][idx, :], label=f'Cluster {clusters["uuids"].iloc[cluster_id]}')
    axs[0].set_ylabel('Firing Rate (Hz)')
    axs[0].set_title('Peri-Event Time Histograms: 200ms Pre-Choice')
    axs[0].legend()

    # Bottom: Raster for the first example cluster
    cluster_id = example_cluster_ids[0]
    for i, rt in enumerate(response_times):
        spike_times_rel = spikes['times'][(spikes['times'] >= rt - 0.2) & (spikes['times'] < rt) & (spikes['clusters'] == cluster_id)] - rt
        axs[1].plot(spike_times_rel, [i] * len(spike_times_rel), '|', color='black')
    axs[1].set_xlim(-0.2, 0)
    axs[1].set_ylim(-0.5, len(response_times) - 0.5)
    axs[1].set_xlabel('Time relative to choice (s)')
    axs[1].set_ylabel('Trial')
    axs[1].set_title(f'Raster Plot for Cluster {clusters["uuids"].iloc[cluster_id]}')

    plt.tight_layout()
    plt.show()

def leaf_descendants(parent_acronyms, only_present=None, pattern=None):
    """
    Get all leaf descendant brain region acronyms of given parent acronyms.   
    """
    
    br = BrainRegions()
    parent_ids = br.acronym2id(parent_acronyms, mapping="Allen")     # acronym -> atlas ids
    desc_ids = br.descendants(parent_ids).id                         # all descendants 

    # keep only leaf nodes (no children)
    leaf_ids = np.intersect1d(desc_ids, br.leaves().id)              # filter to leaves
    acr = np.unique(br.id2acronym(leaf_ids, mapping="Allen"))        # ids -> acronyms 

    if only_present is not None:
        only_present = set(only_present)
        acr = np.array([a for a in acr if a in only_present])

    if pattern is not None:
        rgx = re.compile(pattern)
        acr = np.array([a for a in acr if rgx.search(a)])

    return sorted(acr.tolist())

# Get lowest class count of each area_subdivisions for balancing
def get_lowest_class_count(acronyms):
    """
    Get the lowest class count (0 or 1) for the neurons in the given area_subdivisions.
    """
    neuron_cols_area = [c for c in df.columns if (c[0] not in meta_lvl0) and (c[1] in acronyms)]
    X_area = df[neuron_cols_area].to_numpy(dtype=float)
    
    # Skip if no neurons found for this area
    if X_area.shape[1] == 0:
        return 0

    # Count classes
    class_counts = np.bincount(y)
    return min(class_counts)  