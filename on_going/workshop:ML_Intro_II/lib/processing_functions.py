import re

import matplotlib.pyplot as plt
import pandas as pd


def convert_to_pandas(dataset):
    """Convert a scikit-learn dataset into a features DataFrame and target
    Series. Column naming in the feature DataFrame will be according
    to the supplied feature_names, or accoring to the default 'feature_1',
    'feature_2', ..., 'feature_n' naming.

    Parameters
    ----------
    dataset : dict
        scikit-learn dataset.

    Returns
    -------
    (feature_df,target_s) : (DataFrame,Series)
        Containing both the target and feature data.
    """
    # create feature DataFrame
    feature_names = (dataset.feature_names if 'feature_names' in dataset.keys()
                     else None)
    if (feature_names is None and (not isinstance(dataset.data, list) and
                                   not isinstance(dataset.data, tuple))):
        n_feats = dataset.data.shape[1]
        n_zfill = len(str(n_feats-1))
        columns = list(map(lambda num: 'feature_' +
                           str(num).zfill(n_zfill), range(n_feats)))
    elif feature_names is not None:
        columns = list(map(clean_name, feature_names))
    else:
        columns = ['feature_01']
    feature_df = pd.DataFrame(dataset.data, columns=pd.Index(columns,
                                                             name='features'))

    # create target Series
    target_names = (dataset.target_names if 'target_names' in dataset.keys()
                    else None)
    target_data = list(map(lambda i: target_names[i], dataset.target))\
        if target_names is not None else dataset.target
    target_s = pd.Series(target_data, name='target')

    return feature_df, target_s


def clean_name(name):
    """Clean a name string by filtering out non-alphanumeric and non-
    whitespace characters and replacing the whitespaces with underscores.

    Parameters
    ----------
    name : string

    Return
    ------
    name_sub_2 : string
        Cleaned name
    """
    name_sub1 = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    name_sub2 = re.sub(r'\s', '_', name_sub1)
    return name_sub2


def show_digit(image, label=None, color='green', ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(image, cmap='binary', interpolation='nearest')
    ax.set_axis_off()
    if label is not None:
        ax.text(0, 0, str(label), transform=ax.transAxes, color=color,
                fontsize=16)


def display_digits(X, y, y_pred=None, n_max=20):
    n = n_max if len(X) > n_max else len(X)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X[:n, :])
    if not isinstance(y_pred, pd.Series) and y_pred is not None:
        y_pred = pd.Series(y_pred)
    ncol = 10
    nrow = (n-1)//ncol+1
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        if i < n:
            image = X.iloc[i].values.reshape(8, 8)
            label = y.iloc[i] if y_pred is None else y_pred.iloc[i]
            color = 'green' if y.iloc[i] == label else 'red'
            show_digit(image, label, color, ax=ax)
        else:
            ax.axis('off')
    return fig
