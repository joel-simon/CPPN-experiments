import pickle

import matplotlib.pyplot as plt
import numpy as np

data = pickle.load(open('results.p', 'rb'))

means = data.mean(axis=2).T
stds = data.std(axis=2).T

fig, ax = plt.subplots()

index = np.arange(data.shape[0])
bar_width = 0.25

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index-bar_width, means[0], bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=stds[0],
                 error_kw=error_config,
                 label='frame')

rects2 = plt.bar(index, means[1], bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=stds[1],
                 error_kw=error_config,
                 label='smile')

rects2 = plt.bar(index + bar_width, means[2], bar_width,
                 alpha=opacity,
                 color='g',
                 yerr=stds[2],
                 error_kw=error_config,
                 label='random')

plt.xlabel('Num Steps')
plt.ylabel('Scores')
plt.title('Scores by num_steps and pattern')
plt.ylim([0,1.1])
plt.xticks(index + bar_width / 2, map(str, range(1,6)))
plt.legend()
plt.tight_layout()
plt.show()
