import matplotlib.pyplot as plt
import numpy as np

def plot(data, dimensions, name):
    plt.figure(figsize=(12,6))
    means = data.mean(axis=2).T
    stds = data.std(axis=2).T
    plt.ylim([.5, 1.1])
    plt.plot(dimensions, means[0], '-bo', label='checkerboard')
    plt.errorbar(dimensions, means[0], stds[0], fmt='-bo',)

    plt.plot(dimensions, means[1], '-go', label='random')
    plt.errorbar(dimensions, means[1], stds[1], fmt='-go')

    plt.xlabel('Grid axis dimension')
    plt.ylabel('Average fitness.')

    plt.xticks(dimensions)
    plt.legend()
    plt.title(name)

# names = ['checkerboard', 'R', 'smile', 'random']

# data = pickle.load(open('cppn_results.p', 'rb'))

# means = data.mean(axis=2).T
# stds = data.std(axis=2).T

# fig, ax = plt.subplots()

# index = np.arange(data.shape[0])
# bar_width = 0.2

# opacity = 0.4
# error_config = {'ecolor': '0.3'}

# plt.bar(index-2*bar_width, means[0], bar_width,
#                  alpha=opacity,
#                  color='b',
#                  yerr=stds[0],
#                  error_kw=error_config,
#                  label='checkerboard')

# plt.bar(index-bar_width, means[1], bar_width,
#                  alpha=opacity,
#                  color='g',
#                  yerr=stds[1],
#                  error_kw=error_config,
#                  label='R')

# plt.bar(index, means[2], bar_width,
#                  alpha=opacity,
#                  color='r',
#                  yerr=stds[2],
#                  error_kw=error_config,
#                  label='smile')

# plt.bar(index + bar_width, means[3], bar_width,
#                  alpha=opacity,
#                  color='y',
#                  yerr=stds[3],
#                  error_kw=error_config,
#                  label='random')

# plt.xlabel('Num Steps')
# plt.ylabel('Scores')
# plt.title('Scores by num_steps and pattern')
# plt.ylim([0,1.1])
# plt.xticks(index + bar_width / 2, map(str, range(data.shape[0])))
# plt.legend()
# plt.tight_layout()
# plt.show()
