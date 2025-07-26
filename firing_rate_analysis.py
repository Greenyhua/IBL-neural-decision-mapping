# firing_rate_analysis.py

import numpy as np
import pandas as pd
from one.api import ONE
import matplotlib.pyplot as plt
import seaborn as sns

from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas

one = ONE()
ba = AllenAtlas()
eid = '4720c98a-a305-4fba-affb-bbfa00a724a4' # change as needed
trials = one.load_object(eid, 'trials')

# Just testing (not sure if all sessions have good spike data)
probe_names = one.list_collections(eid)
probe_names = [x.split('/')[1] for x in probe_names if 'pykilosort' in x]

all_frs = []
all_regions = []
all_groups = []
all_choice = []

# region mapping, incomplete, just for demo
acronym_to_group = {
    'VISp': 'Isocortex', 'MOp': 'Isocortex', 'PL': 'Isocortex', 
    'CP': 'Cerebral nuclei', 'CA3': 'Hippocampal formation',
    'TTd': 'Olfactory areas', 'PARN': 'Hindbrain', 'SCm': 'Midbrain'
}

for pname in probe_names:
    print('Analyzing probe:', pname)
    ssl = SpikeSortingLoader(eid=eid, pname=pname, one=one, atlas=ba)
    spikes, clusters, channels = ssl.load_spike_sorting()
    clusters = ssl.merge_clusters(spikes, clusters, channels)
    for acr in np.unique(clusters['acronym']):
        idx = clusters['acronym'] == acr
        if idx.sum() < 2: continue
        # per trial, per choice
        for choice_val in [0, 1]:
            frs = []
            for i in range(len(trials['choice'])):
                if trials['choice'][i] != choice_val: continue
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
            if len(frs) > 0:
                all_frs.append(np.nanmean(frs))
                all_regions.append(acr)
                all_groups.append(acronym_to_group.get(acr, 'Other'))
                all_choice.append('Left' if choice_val==0 else 'Right')

df = pd.DataFrame({'region': all_regions, 'group': all_groups, 'firing_rate': all_frs, 'choice': all_choice})

# group-bar
plt.figure(figsize=(10, 8))
sns.barplot(data=df, x='firing_rate', y='region', hue='group', dodge=False)
plt.xlabel('Mean firing rate (Hz)')
plt.title('Mean Pre-movement Firing Rate by Region (split by group)')
plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('firing_rate_by_region.png')
plt.show()

# choice-bar
plt.figure(figsize=(10, 8))
sns.barplot(data=df, x='firing_rate', y='region', hue='choice')
plt.xlabel('Mean firing rate (Hz)')
plt.title('Pre-movement Firing Rate by Region (Left vs Right)')
plt.tight_layout()
plt.savefig('firing_rate_by_region_choice.png')
plt.show()