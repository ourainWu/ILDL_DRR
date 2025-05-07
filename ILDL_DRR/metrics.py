from scipy import stats
from sklearn.metrics.pairwise import *

eps = np.finfo(np.float64).eps

def chebyshev(y, y_pred):
    diff_abs = np.abs(y - y_pred)
    cheby = np.max(diff_abs, 1)

    return cheby.mean()


def clark(y, y_pred):
    y = np.clip(y, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)
    sum_2 = np.power(y + y_pred, 2)
    diff_2 = np.power(y - y_pred, 2)
    clark = np.sqrt(np.sum(diff_2 / sum_2, 1))

    return clark.mean()


def canberra(y, y_pred):
    y = np.clip(y, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)
    sum_2 = y + y_pred
    diff_abs = np.abs(y - y_pred)
    can = np.sum(diff_abs / sum_2, 1)

    return can.mean()


def kl_divergence(y, y_pred):
    y = np.clip(y, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)
    kl = np.sum(y * (np.log(y) - np.log(y_pred)), 1)

    return kl.mean()


def cosine(y, y_pred):
    return 1 - paired_cosine_distances(y, y_pred).mean()


def intersection(y, y_pred):
    return 1 - 0.5 * np.sum(np.abs(y - y_pred), 1).mean()


def spearman(y, y_pred):
    sum = 0.
    for i in range(y.shape[0]):
        s, _ = stats.spearmanr(y[i], y_pred[i])
        sum += s
    sum /= y.shape[0]
    return sum


def kendall(y, y_pred):
    sum = 0.
    for i in range(y.shape[0]):
        s, _ = stats.kendalltau(y[i], y_pred[i])
        sum += s
    sum /= y.shape[0]
    return sum
