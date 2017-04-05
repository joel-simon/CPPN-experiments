from __future__ import print_function
import numpy as np
import neat
from neat import population, config, nn
from classification import balanced_accuracy

def diff(a, b):
    return balanced_accuracy(a, b)

def express_cppn(net, w, h):
    grid = np.zeros((w, h), dtype=bool)

    for x in range(w):

        for y in range(h):
            # Map x, y to [-1, 1]
            _x = -1.0 + 2.0 * x / (w - 1)
            _y = -1.0 + 2.0 * y / (h - 1)

            out = net.serial_activate([_x, _y])

            grid[x, y] = (out[0] > .5)

    return grid

def express_recurrent_cppn(net, w, h, max_steps, yield_steps=False):
    grid = np.zeros((w, h), dtype=bool)
    next_grid = np.zeros((w, h), dtype=bool)
    inputs = np.zeros(7)

    for s in range(max_steps):
        for x in range(w):
            for y in range(h):

                inputs.fill(0)

                inputs[0] = grid[x, y] * 2 - 1
                inputs[1] = -1.0 + 2.0 * x / (w - 1)
                inputs[2] = -1.0 + 2.0 * y / (h - 1)

                # Map each neighbor to (-1 | 1)
                if x > 0:
                    inputs[3] = (grid[x-1, y]) * 2 - 1

                if x < w-1:
                    inputs[4] = (grid[x+1, y]) * 2 - 1

                if y > 0:
                    inputs[5] = (grid[x, y-1]) * 2 - 1

                if y < h-1:
                    inputs[6] = (grid[x, y+1]) * 2 - 1

                out = net.serial_activate(inputs)

                next_grid[x, y] = (out[0] > 0.5)

        grid, next_grid = next_grid, grid
        next_grid.fill(False)

        if yield_steps:
            yield grid.copy()

    if not yield_steps:
        return grid

def evolve(config, eval_func, pattern, generations):
    """
        eval_func = (net, w, h) -> bool_map of shape [w, h]
    """
    pop = population.Population(config)
    w, h = pattern.shape

    # Use closure to pass evaluate function.
    def evaluate_all(genomes):
        for g in genomes:
            net = nn.create_feed_forward_phenotype(g)
            out = eval_func(net, w, h)
            g.fitness = diff(pattern, out)

    pop.run(evaluate_all, generations)

    winner = pop.statistics.best_genome()
    net = nn.create_feed_forward_phenotype(winner)
    out = eval_func(net, w, h)

    return pop.statistics, out
    # return winner.fitness, out

def evolve_cppn(config, pattern, generations):
    return evolve(config, express_cppn, pattern, generations)

def evolve_recurrent_cppn(config, pattern, max_steps, generations):

    def eval_func (net, w, h):
        return express_recurrent_cppn(net, w, h, max_steps)

    return evolve(config, eval_func, pattern, generations)


if __name__ == '__main__':
    import numpy as np
    from neat.config import Config # Python neat library
    from neat import nn
    import patterns
    target = patterns.checkerboard(16, 16, 4)
    config = Config('config.txt')
    config.pop_size = 50
    config.input_nodes = 7
    evolve_recurrent_cppn(config, target, 2, 50)


