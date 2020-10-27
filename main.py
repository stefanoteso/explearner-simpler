import numpy as np

from sklearn.utils import check_random_state
from os.path import join

from explearner import *


# TODO write ModelBasedDataset with mixins for trees, linear models, and
# non-linear model + LIME or SENN


DATASETS = {
    # Datasets with ground-truth explanations
    'line':
        LineDataset,
    'sine':
        SineDataset,
    'colors-0':
        lambda *args, **kwargs: ColorsDataset(*args, rule=0, **kwargs),
    'colors-1':
        lambda *args, **kwargs: ColorsDataset(*args, rule=1, **kwargs),

    # Datasets with explanations extracted from a model
    'bank':
        BanknoteAuth,
    'adult-lm':
        lambda *args, **kwargs: AdultDataset(*args, clf='lm', **kwargs),
    'adult-dt':
        lambda *args, **kwargs: AdultDataset(*args, clf='dt', **kwargs),
}


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
    n_known = (args.p_known if args.p_known > 1 else
               max(1, np.ceil(len(tr) * args.p_known)))
    observed_X, observed_Z, observed_y, observed_f = [], [], [], []
    for _ in range(int(n_known)):
        i = rng.choice(tr)
        arm = dataset.arms[rng.choice(len(dataset.arms))]
        observed_X.append(dataset.X[i])
        observed_Z.append(arm[0])
        observed_y.append(arm[1])
        observed_f.append(dataset.reward(i, arm[0], arm[1], noise=args.noise))

    print(f'running fold:  #kn={len(observed_f)} #tr={len(tr)} #ts={len(ts)}')

    trace = []
    for t in range(args.n_iters):

        # Fit the GP on the observed data
        gp.fit(np.array(observed_X),
               np.array(observed_Z),
               np.array(observed_y),
               np.array(observed_f))

        # Select a context
        i = rng.choice(tr)

        # TODO plot reward function over time for train contexts

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

        # Predict and compute the regret
        # XXX I am distinguishing between query and prediction so that random
        # selection and UCB can be compared fairly
        zbest, ybest = gp.predict_arm(dataset, dataset.X[i])
        regret = dataset.regret(i, zbest, ybest)

        # Compute the average regret over the test contexts
        test_regrets = []
        for j in ts:
            zhat, yhat = gp.predict_arm(dataset, dataset.X[j])
            test_regrets.append(dataset.regret(j, zhat, yhat))
        avg_test_regret = np.mean(test_regrets)

        print(f'iter {t:2d}:  {regret:5.3f} {avg_test_regret:5.3f}  ctx {i}  y: {dataset.y[i]} vs {ybest}  z: {dataset.Z[i]} vs {zbest}')
        trace.append((regret, avg_test_regret))

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


