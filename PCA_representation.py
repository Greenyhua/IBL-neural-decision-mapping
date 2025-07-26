# pca_representation.py

import numpy as np
import pandas as pd
from one.api import ONE
from sklearn.decomposition import PCA
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
import matplotlib.pyplot as plt

one = ONE()
ba = AllenAtlas()
eid = '4720c98a-a305-4fba-affb-bbfa00a724a4'
trials = one.load_object(eid, 'trials')
choice = np.array(trials['choice'])

probe_names = one.list_collections(eid)
probe_names = [x.split('/')[1] for x in probe_names if 'pykilosort' in x]

region_name = []
corr_coef = []
n_neuron = []
latency_70 = []

for pname in probe_names:
    ssl = SpikeSortingLoader(eid=eid, pname=pname, one=one, atlas=ba)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    for acr in np.unique(clusters['acronym']):
        idx = clusters['acronym'] == acr
        if idx.sum() < 4: continue
        # time bins -0.5 to 0s, 10 bins
        time_bins = np.linspace(-0.5, 0, 10)
        dist_time = []
        for t0 in time_bins:
            frs = []
            for i in range(len(trials['choice'])):
                t_start = trials['firstMovement_times'][i] + t0
                t_end = t_start + 0.05
                if np.isnan(t_start) or np.isnan(t_end):
                    frs.append(np.full(idx.sum(), np.nan))
                    continue
                mask = (spikes['times'] >= t_start) & (spikes['times'] < t_end)
                trial_fr = []
                for cid in clusters['cluster_id'][idx]:
                    c_mask = mask & (spikes['clusters'] == cid)
                    trial_fr.append(np.sum(c_mask)/0.05)
                if len(trial_fr) == idx.sum():
                    frs.append(trial_fr)
                else:
                    frs.append(np.full(idx.sum(), np.nan))
            frs = np.array(frs)
            keep = ~np.isnan(frs).any(axis=1)
            if keep.sum() < 8: 
                dist_time.append(np.nan)
                continue
            X = frs[keep]
            y = choice[keep]
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X)
            left = X2[y==0]
            right = X2[y==1]
            if len(left)==0 or len(right)==0:
                dist_time.append(np.nan)
            else:
                dist_time.append(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))
        dist_time = np.array(dist_time)
        if np.isnan(dist_time).all(): continue
        # corrcoef
        t_idx = ~np.isnan(dist_time)
        if t_idx.sum()<3: continue
        corr = np.corrcoef(time_bins[t_idx], dist_time[t_idx])[0,1]
        # MAX 70%
        dist_max = np.nanmax(dist_time)
        above_70 = np.where(dist_time >= 0.7*dist_max)[0]
        latency = time_bins[above_70[0]] if len(above_70)>0 else np.nan
        region_name.append(acr)
        corr_coef.append(corr)
        n_neuron.append(idx.sum())
        latency_70.append(latency)

df = pd.DataFrame({'region': region_name, 'corr_coef': corr_coef, 'n_neuron': n_neuron, 'latency_70': latency_70})

# Euclidean distance
plt.figure(figsize=(8,6))
plt.hist(df['corr_coef'].dropna(), bins=20, color='skyblue')
plt.xlabel('Correlation coefficient (Euclidean distance vs time)')
plt.title('Distribution of correlation coefficients by region')
plt.tight_layout()
plt.savefig('pca_corrcoef_distribution.png')
plt.show()

# scatter-outlier
plt.figure(figsize=(7,6))
plt.scatter(df['n_neuron'], df['corr_coef'], alpha=0.7)
plt.xlabel('Number of neurons')
plt.ylabel('Correlation coefficient')
plt.title('Corr coef vs number of neurons')
plt.tight_layout()
plt.savefig('pca_corrcoef_vs_n.png')
plt.show()