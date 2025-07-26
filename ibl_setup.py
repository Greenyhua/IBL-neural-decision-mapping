# ibl_setup.py

from one.api import ONE

# just follow the tutorial
one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international')


subject = 'KS023'
sess = one.search(subject=subject)
print("Total sessions:", len(sess))

# Print some session IDs for reference
for i, eid in enumerate(sess[:5]):
    print(f"Session {i}: {eid}")
