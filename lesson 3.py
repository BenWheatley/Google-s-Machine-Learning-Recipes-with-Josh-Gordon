#!/usr/python

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

greyhounds_height = 28 + 4 * np.random.randn(greyhounds)
labs_height = 24 + 4 * np.random.randn(labs)

plt.hist([greyhounds_height, labs_height], stacked=True, color=['r', 'b'])
plt.show()
