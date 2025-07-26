# plot_summary.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('region_decoding_grouped.csv')

plt.figure(figsize=(10, 8))
sns.barplot(data=df.sort_values('decoding_acc', ascending=False).head(20), 
            x='decoding_acc', y='region', hue='group', dodge=False)
plt.xlabel('Decoding Accuracy')
plt.title('Top 20 Regions by Decoding Accuracy')
plt.tight_layout()
plt.savefig('decoding_accuracy_by_region_group.png')
plt.show()

# group-wise summary bar
group_stats = df.groupby('group')['decoding_acc'].mean().reset_index()
plt.figure(figsize=(7,5))
sns.barplot(data=group_stats, x='group', y='decoding_acc')
plt.ylabel('Mean Decoding Accuracy')
plt.title('Mean Decoding Accuracy by Group')
plt.tight_layout()
plt.savefig('decoding_accuracy_by_group.png')
plt.show()