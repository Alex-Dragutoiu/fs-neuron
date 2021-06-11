import matplotlib.pyplot as plt
import json
from numpy import exp

def visualize(v, spikes, legend=False):
    fig, axes = plt.subplots(1, 2)
    i = 0
    for n in v:
        flatten_n = [item for sublist in n for item in sublist]
        label = "neuron_{}".format(i)
        axes[0].plot(n, label=label)
        i += 1

    axes[0].set_title('Membrane Potential')
    axes[0].set_xlabel('ms')
    axes[0].set_ylabel('mV')
    axes[0].grid(axis='x')
    if legend is True:
        axes[0].legend(shadow=True, fancybox=True)

    indices = []

    for s in range(len(spikes)):
        arr = []
        for l in spikes[s]:
            arr.append(s)
        
        axes[1].scatter(spikes[s], arr, s=5, c='b')
        indices.append(arr)

    axes[1].grid(axis='y', linestyle='dotted')
    axes[1].set_title('Spikes')
    axes[1].set_xlabel('ms')
    axes[1].set_ylabel('Neuron ID')

    plt.show()

# inspired from https://machinelearningmastery.com/softmax-activation-function-with-python/
def softmax(vector):
	e = exp(vector)
	return e / e.sum()

# inspired from https://www.geeksforgeeks.org/implement-sigmoid-function-using-numpy/
def sigmoid(x):
    return 1 / (1 + exp(-x))

def print_to_file(args, file, mode='w'):
    f = open(file, 'w')
    for arg in args:
        f.write(str(arg) + '\n')
    f.close()

# inspired from https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
def flatten(arr):
    return [item for sublist in arr for item in sublist]
