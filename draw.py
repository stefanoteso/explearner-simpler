import numpy as np
import matplotlib.pyplot as plt

from explearner import load


def get_style(args, trace_args):
    label = f'EXPLEARN {trace_args.strategy} {trace_args.combiner}'

    color = {
        'ucb': '#ff0000',
        'random': '#00007f',
    }[trace_args.strategy]

    linestyle = {
        'prod': '-',
        'sum': '.',
    }[trace_args.combiner]

    return label, color, linestyle


def draw(args, traces, traces_args):
    n_pickles, n_folds, n_iters, n_measures = traces.shape

    cumavg = lambda y: 1 / np.arange(1, len(y) + 1) * np.cumsum(y)
    PLOTS = [
        ('Regret', 0, None),
        ('Cum. Regret', 0, cumavg),
        ('Test Regret', 1, None),
    ]

    for j, (y_label, index, transform) in enumerate(PLOTS):
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

        ax.set_xlabel('Iterations')
        ax.set_ylabel(y_label)

        for p in range(n_pickles):
            perf = traces[p, :, :, index]

            x = np.arange(n_iters)
            y = np.mean(perf, axis=0)
            if transform:
                y = transform(y)
            yerr = np.std(perf, axis=0) / np.sqrt(n_folds)

            label, color, linestyle = get_style(args, trace_args[p])
            ax.plot(x, y, linewidth=2,
                    label=label, color=color, linestyle=linestyle)
            ax.fill_between(x, y - yerr, y + yerr,
                            alpha=0.35, linewidth=0, color=color)

        ax.legend(loc='upper right', fontsize=8, shadow=False)
        fig.savefig(args.basename + '__{}.png'.format(j),
                    bbox_inches='tight',
                    pad_inches=0)
        del fig


if __name__ == '__main__':
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    args = parser.parse_args()

    traces, trace_args = [], []
    for path in args.pickles:
        data = load(path)
        traces.append(data['traces'])
        trace_args.append(data['args'])
    traces = np.array(traces)

    draw(args, traces, trace_args)
