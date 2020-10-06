import numpy as np

from sklearn.utils import check_random_state
from sklearn.model_selection import KFold
from os.path import join

from explearner import *


# TODO convert to sklearn's GPs + perhaps gpflow Gram matrices
# TODO write ModelBasedDataset with mixins for trees, linear models, and
# non-linear model + LIME or SENN


DATASETS = {
    # Datasets with ground-truth explanations
    'sine':
        SineDataset,
    'colors-0':
        lambda *args, **kwargs: ColorsDataset(*args, rule=0, **kwargs),
    'colors-1':
        lambda *args, **kwargs: ColorsDataset(*args, rule=1, **kwargs),

    # Datasets with explanations extracted from a model
    'adult-lm':
        lambda *args, **kwargs: AdultDataset(*args, clf='lm', **kwargs),
    'adult-dt':
        lambda *args, **kwargs: AdultDataset(*args, clf='dt', **kwargs),
}


def evaluate_fold(dataset, kn, tr, ts, args, rng=None):
    rng = check_random_state(rng)

    gp = CGPUCB(dataset, strategy=args.strategy, rng=rng)

    trace = []
    fhat = np.zeros_like(dataset.f)
    for t in range(args.n_iters):

        # Fit the GP on the observed data
        gp.X = gp.concat(dataset.X[kn], dataset.Z[kn], dataset.y[kn].reshape(-1, 1))
        gp.Y = fhat[kn]

        # Select query
        i = rng.choice(tr)
        kn, tr = move_indices(kn, tr, [i])

        # Select an arm and observe the reward
        zhat, yhat = gp.select_arm(dataset, dataset.X[i])
        fhat[i] = dataset.reward(i, zhat, yhat, noise=args.noise)

        # Predict and compute the regret
        # XXX I am distinguishing between query and prediction so that random
        # selection and UCB can be compared fairly
        zpred, ypred = gp.predict(dataset, dataset.X[i])
        regret = dataset.f[i] - dataset.reward(i, zpred, ypred, noise=0)
        print(f'iter {t:2d}:  regret={regret:5.3f} true={dataset.y[i]} pred={ypred}')
        trace.append(regret)

    return trace


def evaluate(dataset, args, rng=None):
    rng = check_random_state(rng)

    traces = []
    split = KFold(n_splits=args.n_splits, shuffle=True, random_state=rng)
    for k, (tr, ts) in enumerate(split.split(dataset.X)):
        n_known = max(1, int(np.ceil(len(tr) * args.p_known)))
        tr = rng.permutation(tr)
        kn, tr = tr[:n_known], tr[n_known:]
        traces.append(evaluate_fold(dataset, kn, tr, ts, args, rng=rng))

    return traces


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
                       help='Proportion of initial labelled examples')
    group.add_argument('-T', '--n-iters', type=int, default=100,
                       help='Maximum number of learning iterations')

    args = parser.parse_args()

    np.seterr(all='raise')
    np.set_printoptions(precision=3, linewidth=80)

    np.random.seed(args.seed) # XXX just in case
    tf.random.set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    dataset = DATASETS[args.dataset](combiner=args.combiner, rng=rng)
    traces = evaluate(dataset, args, rng=rng)

    path = _get_basename(args) + '__trace.pickle'
    dump(join('results', path), {
        'args': args,
        'traces': traces
    })


if __name__ == '__main__':
    main()
