# IBL-neural-decision-mapping
A project for the Neuromatch Academy Computational Neuroscience course. We use machine learning methods (logistic regression, PCA) to decode mouse decisions from brain-wide neural activity recorded in the IBL (International Brain Laboratory) dataset. 
# Mouse Decision-Making Brain-Wide Analysis 🐭🧠

Hi!  
This repo is our code collection for our Neuromatch project _Mapping Decision-Related Neural Dynamics Across the Mouse Brain Using Machine Learning Approaches_, using IBL data.  
Most scripts are not perfect.  
You can always email me at qingshijiark800@gmail.com if there's problem!  
**This whole thing is a work in progress, but I hope it helps someone!**

---

## Project Summary

We tried to figure out which mouse brain regions can predict left/right decisions *before* movement, using International Brain Lab (IBL) open-access data.  
We did per-region decoding, firing rate stats, PCA, and some rough region grouping.  


---

## File Guide

- **ibl_setup.py**  
  Just basic connection to IBL server, session listing, etc.  
  (remember to change the session ID)

- **firing_rate_analysis.py**  
  For each brain region: calculates mean firing rate in pre-movement window.  


- **region_decoding.py**  
  Logistic regression decoding (left vs right choice) for each region.  
  Our main analysis script.  

- **pca_representation.py**  
  Tries PCA for each region and computes how much neural population trajectories diverge for left/right.  
  (Not sure if the PCA is always meaningful, but results look interesting.)

- **region_group_mapping.py**  
  Just some quick mapping of region acronyms to big brain groups (isocortex, hippocampus, etc).  

- **plot_summary.py**  
  Plots top regions by decoding accuracy.  
  
---

## Requirements

- Python 3.8+ 
- `ibllib`, `one-api`, `brainbox`, `iblatlas` (see [IBL docs](https://int-brain-lab.github.io/ONE/))
- `numpy`, `matplotlib`, `sklearn`, `pandas`
- If you run into missing packages, just pip install as needed

---

## Usage

All scripts are meant to be run independently, you can just  
```bash
python region_decoding.py
