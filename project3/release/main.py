#!/usr/bin/env python3
# Changes to this file will not be used during grading

import glob
import os
import pickle

import project3 as p3
import utils as utils

import pandas as pd

PROJ_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJ_DIR, 'models')

if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

# -------------------------------------------------------------------------------
# Part 1.1
# -------------------------------------------------------------------------------

toy_data = pd.read_csv(os.path.join(PROJ_DIR, 'toy_data.csv')).as_matrix()

# for k in range(1, 6):
#     mu, cluster_assignments = p3.k_means(toy_data, k)
#     utils.plot_kmeans_clusters(toy_data, k, mu, cluster_assignments)

# -------------------------------------------------------------------------------
# Part 1.2
# -------------------------------------------------------------------------------

GMM_K_MIN_MAX = (1, 5)
utils.fit_k(p3.GMM, toy_data, *GMM_K_MIN_MAX,MODELS_DIR, verbose=False,
														 d=toy_data.shape[1])

# -------------------------------------------------------------------------------
# Part 1.3
# -------------------------------------------------------------------------------

utils.test_em_gmm(toy_data)

# -------------------------------------------------------------------------------
# Part 1.4
# -------------------------------------------------------------------------------

# snaps = glob.glob(os.path.join(MODELS_DIR, 'gmm_*.pkl'))
# snaps.sort(key=utils.get_k)
# for snap in snaps:
#     with open(snap, 'rb') as f_snap:
#         model = pickle.load(f_snap)
#         utils.plot_em_clusters(toy_data, model)

# -------------------------------------------------------------------------------
# Part 2.4
# -------------------------------------------------------------------------------

utils.test_em_cmm()

# -------------------------------------------------------------------------------
# Part 2.5
# # -------------------------------------------------------------------------------

field_cats = utils.load_categories(os.path.join(PROJ_DIR, 'categories.txt'))
data = pd.read_csv(os.path.join(PROJ_DIR, 'census_data.csv.gz'))
ds = data.apply(pd.Series.nunique)

# CMM_K_MIN_MAX = (2, 20)
# utils.fit_k(p3.CMM, data, *CMM_K_MIN_MAX, MODELS_DIR, verbose=False, ds=ds)

# -------------------------------------------------------------------------------
# Part 2.6b
# -------------------------------------------------------------------------------

# snaps = glob.glob(os.path.join(MODELS_DIR, 'cmm_*.pkl'))
# snaps.sort(key=utils.get_k)
# ks, bics, lls = [], [], []
# for snap in snaps:
#     with open(snap, 'rb') as f_snap:
#         model = pickle.load(f_snap)
#     ks.append(utils.get_k(snap))
#     lls.append(model.max_ll)
#     bics.append(model.bic)
# utils.plot_ll_bic(ks, lls, bics)

# -----------------------------s--------------------------------------------------
# Part 2.7
# -------------------------------------------------------------------------------

K_SHOW = 7 # best K and then some other K
with open(os.path.join(MODELS_DIR, 'cmm_k%d.pkl' % K_SHOW), 'rb') as f_model:
    model = pickle.load(f_model)
utils.print_census_clusters(model, data.columns, field_cats)
