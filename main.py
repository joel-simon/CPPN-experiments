from __future__ import print_function
import numpy as np
import pickle
from neat.config import Config
from patterns import patterns
from cppn import evolve_cppn, evolve_recurrent_cppn

NUM_GENS = 300
NUM_REPEATS = 5

config = Config('config.txt')

recurrent_config = Config('config.txt')
recurrent_config.input_nodes = 7

# Currently we are only considering the effect of max_step size.
paremeters = {
    'max_steps': range(1, 6, 1)
}

# Store the results for every run of every pattern with every parameter.
n = len(paremeters['max_steps'])

results = np.zeros((n, len(patterns), NUM_REPEATS))

for s, steps in enumerate(paremeters['max_steps']):

    for p, pattern in enumerate(patterns):

        for r in range(NUM_REPEATS):

            if steps == 1:

                f, out = evolve_cppn(config, pattern, NUM_GENS)

            else:

                f, out = evolve_recurrent_cppn(recurrent_config, pattern, steps, NUM_GENS)

            print(steps, p, r, f)
            results[s, p, r] = f

print(results)

pickle.dump(results, open('results.p', 'wb'))

for i in range(results.shape[0]):
    print(i, results[i][0].mean())
