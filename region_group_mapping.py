# region_group_mapping.py
import pandas as pd

region2group = {
    'VISp': 'Isocortex',
    'MOp': 'Isocortex',
    'PL': 'Isocortex',
    'CP': 'Cerebral nuclei',
    'CA3': 'Hippocampal formation',
    'TTd': 'Olfactory areas',
    'PARN': 'Hindbrain',
    'SCm': 'Midbrain'
}
df = pd.read_csv('region_decoding.csv')
df['group'] = df['region'].map(region2group)
df.to_csv('region_decoding_grouped.csv', index=False)