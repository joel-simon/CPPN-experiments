import matplotlib.pyplot as plt
import numpy as np

def plot_fitnesses(data, labels, name):
    f, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True) # 2 patterns

    means = data.mean(axis=2)
    stds = data.std(axis=2)

    axes[0].set_title('Random Pattern')
    axes[1].set_title('Checkerboard Pattern')

    X = range(data.shape[3])
    for li, l in enumerate(labels):
        axes[0].errorbar(X, means[li][0], stds[li][0], label=l)
        axes[1].errorbar(X, means[li][1], stds[li][1], label=l)

    axes[0].set_xlabel('Generations')
    axes[0].set_ylabel('Average fitness.')
    axes[0].legend(loc="best")

def plot_best(F, O, i=0):
    print('Best Random Fitness:', F[i, 0, :, -1].max())
    print('Best Grid Fitness:', F[i, 1, :, -1].max())
    best_random = O[i, 0, np.argmax(F[i, 0].max(axis=1))]
    best_grid = O[i, 1, np.argmax(F[i, 1].max(axis=1))]
    plot_patterns(best_random, best_grid)

def plot_patterns(*args):
    f, axes = plt.subplots(1, len(args), figsize=(2*len(args), 2))
    for i, (x, a) in enumerate(zip(args, axes)):
        a.set_title(str(i))
        a.imshow(x, cmap='gray_r', interpolation='nearest')
