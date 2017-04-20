import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from autoencoder import build_autoencoder
from myutils import make_logger
from itertools import product
import argparse

logger = make_logger('clustering')
PATH = '/home/aeulloacerna/data/copd/preprocessing/'


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true',
                        default=False,
                        help='Compute runs from scratch')
    return parser


def plot(dfIN, X2, folder_path='/tmp/'):
    df = dfIN.copy()
    df['PT_SURVIVAL'] = df.PT_DEATH_DT.dt.year - df.LAB_DT
    df['N_COMOB'] = df.select_dtypes(include=['bool']).sum(axis=1)
    sns.set(rc={'axes.facecolor': 'black',
                'axes.grid': False,
                'legend.fontsize': 14,
                'legend.framealpha': 1,
                'legend.markerscale': 5})

    tf = pd.DataFrame(X2, columns=['t1', 't2'], index=df.index)
    # Continuous factors
    factors = ['PT_SURVIVAL', 'PT_AGE', 'N_COMOB']
    factors.extend(df.select_dtypes(include=['float64']
                                    .columns.tolist()))
    df = pd.concat((df, tf), axis=1)
    for factor in factors:
        df[factor] = pd.qcut(df[factor], 4)
        plt.figure(figsize=(10, 10))
        sns.lmplot(data=df, x='t1', y='t2',
                   hue=factor, fit_reg=False,
                   palette=sns.color_palette(
                       "winter", n_colors=df[factor].nunique()),
                   markers='.',
                   scatter_kws={'alpha': 0.8, 's': 20})
        plt.savefig('{}/{}.png'
                    .format(folder_path, factor.replace("/", "")))

    # Categorical factors
    cols = ['PT_SEX', 'PT_HX_SMOKE', 'STUDY_NM', 'PT_STATUS']
    cols.extend(df.select_dtypes(include=['bool']).columns.tolist())
    for factor in cols:
        plt.figure(figsize=(10, 10))
        if factor == 'PT_SEX':
            hue_order = ['Female', 'Unknown', 'Male']
        elif factor == 'PT_STATUS':
            hue_order = ['ALIVE', 'Unknown', 'DECEASED']
        else:
            hue_order = df[factor].unique()
        sns.lmplot(data=df, x='t1', y='t2',
                   hue=factor, fit_reg=False,
                   hue_order=hue_order,
                   markers='.',
                   palette=sns.color_palette(
                       "RdBu", n_colors=df[factor].nunique()),
                   scatter_kws={'alpha': 0.8, 's': 20})
        plt.savefig('{}/{}.png'.format(folder_path, factor))


def run_pca(X, **kwargs):

    logger.info('Standarize')
    if X.dtype == 'bool':
        logger.info('Data is bool type, skipping.')
        Xs = X
    else:
        # Standarize
        ptp = X.ptp(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1
        sd[ptp == 1] = 1
        mu = X.mean(axis=0)
        mu[sd == 0] = 0
        mu[ptp == 1] = 0
        Xs = (X - mu) / sd
        logger.info('Done.')
    # PCA
    logger.info('Running PCA with 90% retention')
    pca_args = {'n_components': 0.9, 'whiten': True}
    pca_args.update(kwargs)
    pc = PCA(**pca_args)
    Xp = pc.fit_transform(Xs)
    logger.info('Done. Kept {} components'.format(pc.components_.shape[0]))
    return Xp


def run_ae(X, **kwargs):
    logger.info('Standarize')
    if X.dtype == 'bool':
        loss = 'binary_crossentropy'
        Xs = X
    else:
        # Standarize
        ptp = X.ptp(axis=0)
        ptp[ptp == 0] = 1
        Xs = (X - X.min(axis=0)) / ptp
        loss = 'mse'
        logger.info('Done. AE Loss: {}'.format(loss))

    # AE
    ae_args = {'layers_dim': [200, 50], 'inits': ['normal', 'normal'],
               'activations': ['relu', 'sigmoid'], 'l2': 0,
               'optimizer': 'adagrad'}
    ae_args.update(kwargs)
    logger.info('Training Autoencoder')
    ae, encoder = build_autoencoder(Xs.shape[1], **ae_args)
    logger.info(ae.summary())
    ae.fit(Xs, Xs, batch_size=128, epochs=20, shuffle=False,
           validation_split=0.1)
    Xp = encoder.predict(Xs)
    logger.info('Done.')
    return Xp


def run_tsne(Xp, **kwargs):
    # TSNE
    logger.info('Running TSNE on {} samples with params: {}'.format(
        Xp.shape[0], kwargs))
    Xt = TSNE(**kwargs).fit_transform(Xp)
    logger.info('Done')
    return Xt


def run_experiment(X, df, group,
                   feature_reduction='pca',
                   fr_params={},
                   name='temp',
                   tsne_params={}):
    """
    Data storage
    name
    - feature_reduction
    - tsne
    -- images

    """
    project_path = PATH + name + '/'
    plot_path = project_path + '/plots/' + feature_reduction + '/'

    if not os.path.exists(project_path):
        os.makedirs(project_path)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    X2_path = project_path + '{}_tsne.npy'.format(feature_reduction)

    # Feature reduction
    if feature_reduction == 'pca':
        Xp = run_pca(X, **fr_params)
    elif feature_reduction == 'ae':
        Xp = run_ae(X, **fr_params)
    else:
        logger.error('Feature reduction {} not implemented'
                     .format(feature_reduction))
    temp = project_path + '{}.npy'.format(feature_reduction)
    logger.info('Saving {}'.format(temp))
    np.save(temp, Xp)
    # Sample set selection
    if group is not None:
        Xp = Xp[group, :]
    # TSNE
    X2 = run_tsne(Xp, **tsne_params)

    logger.info('Saving {}'.format(X2_path))
    np.save(temp, X2)

    plot(df[group], X2, folder_path=plot_path)


if __name__ == "__main__":
    df = pd.read_pickle(PATH + 'miced.pkl')
    parser = make_argparser().parse_args()

    X1 = df.select_dtypes(include=['float64']).values
    X2 = df.select_dtypes(include=['bool']).values
    Xs = {'measurements': X1,
          'comorbidities': X2,
          'meas_comob': np.concatenate((X1, X2), axis=1)}
    groups = {'deceased': (df['PT_STATUS'] == 'DECEASED').values,
              'diabetes': df['DIABETES_INDEX_DT'].values,
              'cough': df['CHRONICCOUGH_INDEX_DT'].values}
    frs = ['pca', 'ae']

    for (X, group, fr) in product(*[Xs.keys(), groups.keys(), frs]):
        logger.info('EXPERIMENT on {}, {}, with {}'.format(X, group, fr))
        exp_params = {'X': Xs[X], 'df': df,
                      'group': groups[group],
                      'name': X + '_' + group,
                      'feature_reduction': fr,
                      'tsne_params': {'perplexity': 100}}
        run_experiment(**exp_params)
