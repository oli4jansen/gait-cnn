import argparse
import matplotlib.pyplot as plt
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument('losses_file', type=str, default='losses.json')
args = parser.parse_args()
data = json.load(open(args.losses_file))

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

print()

losses = []
for epoch in data:
    losses = losses + list(data[epoch].values())


plt.plot(list(losses))
plt.plot(movingaverage(list(losses), 100))

plt.xlabel('iteration')
plt.ylabel('loss')
plt.grid(True)
plt.show()
