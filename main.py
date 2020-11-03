import numpy as np

from sklearn.utils import check_random_state
from itertools import product
from os.path import join

from explearner import *


# TODO write ModelBasedDataset with mixins for trees, linear models, and
# non-linear model + LIME or SENN


DATASETS = {
    # Datasets with ground-truth explanations
    'debug':
        DebugDataset,
    'line':
        LineDataset,
    'sine':
        SineDataset,
    'colors-0-relevance':
        lambda *args, **kwargs:
            ColorsDataset(*args, rule=0, kind='relevance', **kwargs),
    'colors-1-relevance':
        lambda *args, **kwargs:
            ColorsDataset(*args, rule=1, kind='relevance', **kwargs),
    'colors-0-polarity':
        lambda *args, **kwargs:
            ColorsDataset(*args, rule=0, kind='polarity', **kwargs),
    'colors-1-polarity':
        lambda *args, **kwargs:
            ColorsDataset(*args, rule=1, kind='polarity', **kwargs),

    # Datasets with explanations extracted from a model
    'banknote':
        BanknoteAuth,
    'breast':
        BreastCancer,
    'wine':
        WineQuality,
}


def evaluate_iter(dataset, gp, i, ts):
    # XXX we distinguish between query arm and predicted arm so that random
    # selection and UCB can be compared fairly

    # Predict and compute the regret of the best arm
    zbest, ybest = gp.predict_arm(dataset, dataset.X[i])
    pred_regret = dataset.regret(i, zbest, ybest)

    # Compute the average regret over the test contexts
    test_regrets = []
    for j in ts:
        zhat, yhat = gp.predict_arm(dataset, dataset.X[j])
        test_regrets.append(dataset.regret(j, zhat, yhat))
    test_regret = np.mean(test_regrets)

    print(f'iter: best reg={pred_regret:5.3f} test reg={test_regret:5.3f}  ctx {i}  true=({dataset.Z[i]}, {dataset.y[i]}) pred=({zbest}, {ybest})')

    return pred_regret, test_regret


def evaluate_fold(dataset, tr, ts, args, rng=None):
    """Run EXPLEARN on a given kn-tr-ts fold.

    Arguments
    ---------
    dataset : explearner.Dataset
        The dataset.
    tr : list of int
        Indices of contexts sampled during training.
    ts : list of int
        Indices of contexts used for measuring generalization-across-contexts.
    args : Arguments
        The command-line arguments.
    rng : None or int or RandomState, defaults to None
        The RNG.
    """
    rng = check_random_state(rng)
    Xsize = dataset.X.shape[0]
    delta = 0.8

    gp = CGPUCB(kernel=dataset.kernel,
                strategy=args.strategy,
                random_state=rng)

    # Observe the reward of some random context-arm pairs
    observed_X, observed_Z, observed_y, observed_f = [], [], [], []
    if args.passive:
        for i, arm in product(range(len(dataset.X)), dataset.arms):
            observed_X.append(dataset.X[i])
            observed_Z.append(arm[0])
            observed_y.append(arm[1])
            observed_f.append(dataset.reward(i, arm[0], arm[1], noise=args.noise))
        n_iters = 1
    else:
        n_known = (args.p_known if args.p_known > 1 else
                   max(1, np.ceil(len(tr) * args.p_known)))
        for _ in range(int(n_known)):
            i = rng.choice(tr)
            arm = dataset.arms[rng.choice(len(dataset.arms))]
            observed_X.append(dataset.X[i])
            observed_Z.append(arm[0])
            observed_y.append(arm[1])
            observed_f.append(dataset.reward(i, arm[0], arm[1], noise=args.noise))
        n_iters = args.n_iters

    print(f'running fold:  #arms={len(dataset.arms)} - #kn={len(observed_f)} #tr={len(tr)} #ts={len(ts)}')

    trace = [evaluate_iter(dataset, gp, i, ts)]
    for t in range(n_iters):

        # Fit the GP on the observed data
        gp.fit(np.array(observed_X),
               np.array(observed_Z),
               np.array(observed_y),
               np.array(observed_f))

        # XXX DEBUG
        if args.passive:
            from pprint import pprint
            pprint(list(zip(observed_X, observed_Z, observed_y, observed_f)))
            for i, arm in product(range(len(dataset.X)), dataset.arms):
                x = dataset.X[i]
                z = arm[0]
                y = arm[1]
                tempx = x.reshape((1, -1))
                tempz = z.reshape((1, -1))
                tempy = np.array([y])
                print(f'reward @ ({x}, {z}, {y}): true={dataset.reward(i, z, y):7.4f} pred={gp.predict(tempx, tempz, tempy)[0]:7.4f}')
            quit()

        # Select a context
        i = rng.choice(tr)

        # Select a query arm and observe the reward
        # XXX beta = 2*B**2 + 300*gamma*np.log(t / delta)**3
        beta = 2 * np.log(Xsize * ((t + 1) ** 2) * (np.pi ** 2) / (6 * delta))
        xhat = dataset.X[i]
        zhat, yhat = gp.select_arm(dataset, xhat, beta=1)
        fhat = dataset.reward(i, zhat, yhat, noise=args.noise)

        # Update the set of observations
        observed_X.append(xhat)
        observed_Z.append(zhat)
        observed_y.append(yhat)
        observed_f.append(fhat)

        trace.append(evaluate_iter(dataset, gp, i, ts))

    return trace


def _get_basename(args):
    fields = [
        (None, args.dataset),
        ('C', args.combiner),
        ('N', args.noise),
        ('S', args.strategy),
        ('k', args.n_splits),
        ('K', args.p_known),
        ('T', args.n_iters),
        (None, args.seed),
    ]

    basename = '__'.join([name + '=' + str(value) if name else str(value)
                         for name, value in fields])
    return basename


def main():
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)

    group = parser.add_argument_group('Data')
    group.add_argument('dataset', choices=sorted(DATASETS.keys()),
                       help='dataset to be used')
    group.add_argument('-C', '--combiner', type=str, default='prod',
                       help='How to combine the various kernels.')
    group.add_argument('-N', '--noise', type=float, default=0,
                       help='Std. dev. of reward noise')
    group.add_argument('-s', '--seed', type=int, default=0,
                       help='RNG seed')

    group = parser.add_argument_group('Evaluation')
    group.add_argument('-S', '--strategy', type=str, default='random',
                       help='Query selection strategy')
    group.add_argument('-k', '--n-splits', type=int, default=5,
                       help='Number of cross-validation folds')
    group.add_argument('-K', '--p-known', type=float, default=0.01,
                       help='Proportion of seed context-arm rewards')
    group.add_argument('-T', '--n-iters', type=int, default=100,
                       help='Maximum number of learning iterations')
    group.add_argument('--passive', action='store_true',
                       help='Print passive performance, then quit')

    args = parser.parse_args()

    #np.seterr(all='warn') # XXX the RBF kernel underflows often
    np.set_printoptions(precision=3, linewidth=80)

    np.random.seed(args.seed) # XXX just in case
    rng = np.random.RandomState(args.seed)

    dataset = DATASETS[args.dataset](combiner=args.combiner, rng=rng)
    traces = [evaluate_fold(dataset, tr, ts, args, rng=rng)
            for tr, ts in list(dataset.split(args.n_splits))]

    path = _get_basename(args) + '__trace.pickle'
    dump(join('results', path), {
        'args': args,
        'traces': traces
    })


if __name__ == '__main__':
    main()


