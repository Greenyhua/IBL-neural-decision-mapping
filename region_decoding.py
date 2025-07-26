# region_decoding.py

import numpy as np
import pandas as pd
from one.api import ONE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas

one = ONE()
ba = AllenAtlas()
eid = '4720c98a-a305-4fba-affb-bbfa00a724a4'
trials = one.load_object(eid, 'trials')
choice = np.array(trials['choice'])

# region mapping, incomplete
acronym_to_group = {
    'VISp': 'Isocortex', 'MOp': 'Isocortex', 'PL': 'Isocortex', 
    'CP': 'Cerebral nuclei', 'CA3': 'Hippocampal formation',
    'TTd': 'Olfactory areas', 'PARN': 'Hindbrain', 'SCm': 'Midbrain'
}

probe_names = one.list_collections(eid)
probe_names = [x.split('/')[1] for x in probe_names if 'pykilosort' in x]

region_acc = []
region_name = []
region_group = []

for pname in probe_names:
    ssl = SpikeSortingLoader(eid=eid, pname=pname, one=one, atlas=ba)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    for acr in np.unique(clusters['acronym']):
        idx = clusters['acronym'] == acr
        if idx.sum() < 2: continue
        frs = []
        for i in range(len(trials['choice'])):
            t0 = trials['firstMovement_times'][i] - 0.5 if not np.isnan(trials['firstMovement_times'][i]) else np.nan
            t1 = trials['firstMovement_times'][i]
            if np.isnan(t0) or np.isnan(t1):
                frs.append(np.nan)
                continue
            mask = (spikes['times'] >= t0) & (spikes['times'] < t1)
            trial_fr = []
            for cid in clusters['cluster_id'][idx]:
                c_mask = mask & (spikes['clusters'] == cid)
                trial_fr.append(np.sum(c_mask)/0.5)
            if len(trial_fr) == 0:
                frs.append(np.nan)
            else:
                frs.append(np.mean(trial_fr))
        frs = np.array(frs)
        y = choice
        keep = ~np.isnan(frs) & ~np.isnan(y)
        if keep.sum() < 12: continue
        X = frs[keep].reshape(-1,1)
        y = y[keep]
        try:
            X_train, X_test = X[:len(X)//2], X[len(X)//2:]
            y_train, y_test = y[:len(y)//2], y[len(y)//2:]
            clf = LogisticRegression(max_iter=500).fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
        except Exception as e:
            acc = np.nan
        region_acc.append(acc)
        region_name.append(acr)
        region_group.append(acronym_to_group.get(acr, 'Other'))

df = pd.DataFrame({'region': region_name, 'group': region_group, 'decoding_acc': region_acc})

# barplot
plt.figure(figsize=(10, 8))
sns.barplot(data=df.sort_values('decoding_acc', ascending=False).head(20), 
            x='decoding_acc', y='region', hue='group', dodge=False)
plt.xlabel('Decoding Accuracy')
plt.title('Top 20 Regions by Decoding Accuracy')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('decoding_accuracy_by_region.png')
plt.show()