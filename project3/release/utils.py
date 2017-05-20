# Changes to this file will not be used during grading

import pickle
import re
import os

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import project3 as p3


def plot_kmeans_clusters(data, k, mu, cluster_assign):
    print(mu)
    color_map = {0: "r", 1: "b", 2: "g", 3: "y", 4: "m", 5: "c"}
    colors = [color_map[x % 5] for x in cluster_assign]
    if k > 2:
        v = Voronoi(mu)
        voronoi_plot_2d(v)
    plt.title("K-means clustering with k= " + str(k))
    ax = plt.gcf().gca()
    ax.set_xlim((-15, 5))
    ax.set_ylim((-15, 5))
    plt.scatter(data[:, 0], data[:, 1], color=colors)
    print(mu[:, 0],mu[:, 1])
    plt.scatter(mu[:, 0], mu[:, 1], color="k")
    plt.show()


def fit_k(Model, data, k_min, k_max, snap_path, verbose, **model_opts):
    for k in range(k_min, k_max + 1):
        while True:
            print('Fitting k = %d' % k, end=': ', flush=True)
            model = Model(k, **model_opts)
            if model.fit(data, verbose=False):
                break
            print('bad init; trying again...')
        model_type = Model.__name__.lower()
        msnap_path = os.path.join(snap_path, '%s_k%s.pkl' % (model_type, k))
        with open(msnap_path, 'wb') as f_snap:
            pickle.dump(model, f_snap)


def plot_em_clusters(data, model):
    r = 0.25
    color = ["r", "b", "g", "y", "m", "c"]

    k = model.k
    mu = model.params['mu']
    pi = model.params['pi']
    var = model.params['sigsq']
    p_z = model.params['p_z']

    n, d = np.shape(data)
    per = p_z/(1.0*np.tile(np.reshape(np.sum(p_z, axis=1), (n, 1)), (1, k)))
    fig = plt.figure()
    ax = plt.gcf().gca()
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))

    for i in range(len(data)):
        angle = 0
        for j in range(k):
            c = (data[i, 0], data[i, 1])
            cir = pat.Arc(c, r, r, 0, angle,
                          angle + per[i, j]*360, edgecolor=color[j])
            ax.add_patch(cir)
            angle += per[i, j]*360

    for j in range(k):
        c = (mu[j, 0], mu[j, 1])
        sigma = np.sqrt(var[j])
        circle = plt.Circle(c, sigma, color=color[j], fill=False)
        ax.add_artist(circle)
        text = plt.text(*c,
                        "mu=(" + str("%.2f" % c[0]) + "," +
                        str("%.2f" % c[1])+")" +
                        ",stdv=" + str("%.2f" % np.sqrt(var[j])))
        ax.add_artist(text)
    plt.axis('equal')
    plt.show()


def test_em_gmm(data):
    k = 3
    model = p3.GMM(k, data.shape[1])
    model.params = {
        'mu': data[range(1, k+1)],
        'sigsq': np.ones(k),
        'pi': np.ones(k)/k,
    }

    ll, pz = model.e_step(data)
    new_params = model.m_step(data, pz)

    np.testing.assert_almost_equal(ll, -3355.0166, decimal=3)
    np.testing.assert_almost_equal(new_params['mu'].sum(), -29.5513, decimal=3)
    np.testing.assert_almost_equal(new_params['sigsq'].sum(), 8.8038, decimal=3)
    np.testing.assert_almost_equal(new_params['pi'].sum(), 1, decimal=3)

    print('Tests passed!')


def get_k(p):
    return int(re.search('_k(\d+)\.pkl', p).group(1))


def load_categories(catfile):
    with open(catfile) as f_cat:
        field_cats = {}
        cur_field = None
        for l in f_cat:
            l = l.rstrip()
            if not l:
                continue
            elif not l.startswith(' '):
                field, field_desc = l.split(' - ')
                cur_field = field
                field_cats[cur_field] = []
            else:
                field_cats[cur_field].append(l.split(': ')[1])
    return field_cats


def plot_ll_bic(ks, lls, bics):
    plt.figure()
    plt.plot(ks, lls, label='ll')
    plt.plot(ks, bics, label='BIC')
    plt.title('Maximized LL and BIC vs K')
    plt.xlabel('K')
    plt.ylabel('(penalized) LL')
    plt.legend()
    plt.show()


def print_census_clusters(model, fields, categories):
    assert isinstance(model, p3.CMM)
    assert len(model.alpha) == len(fields)
    assert len(fields) <= len(categories)

    max_cats = np.zeros((model.k, len(fields)))
    for i, a in enumerate(model.alpha):
        max_cats[:, i] = a.argmax(1)

    for k in range(model.k):
        cluster_mcs = max_cats[k].astype(int)
        cnames = [categories[f][cluster_mcs[i]] for i, f in enumerate(fields)]
        fc_strs = '\n  '.join('%s: %s' % fc for fc in zip(fields, cnames))
        print('Cluster %s:\n  %s\n' % (k + 1, fc_strs))


def test_em_cmm():
    np.random.seed(42)

    ds = [4, 2, 3]
    data = np.zeros((12, len(ds)))
    for i, d in enumerate(ds):
        data[:, i] = np.random.randint(d, size=len(data))
    mask = np.random.binomial(1, p=0.75, size=data.shape).astype(float)
    mask[mask == 0] = np.nan
    data = pd.DataFrame(data * mask)

    model = p3.CMM(4, ds=ds)

    ll, p_z = model.e_step(data)
    np.testing.assert_almost_equal(ll, -38.39676, decimal=3)
    np.testing.assert_almost_equal(p_z.var(), 0.07213, decimal=3)

    new_params = model.m_step(data, p_z)
    np.testing.assert_almost_equal(new_params['pi'].var(), 0.018766, decimal=3)
    np.testing.assert_almost_equal(new_params['alpha'][1].var(), 0.09872,
                                                                     decimal=3)
