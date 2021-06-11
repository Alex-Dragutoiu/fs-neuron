try:
    import pyNN.spiNNaker as sim
except Exception:
    import spynnaker8 as sim
import numpy as np
import matplotlib.pyplot as plt
from python_models8.neuron.builds.my_full_neuron import FSNeuron

from fsnn.snn import load_model
from fsnn.utils import visualize, softmax
from keras.datasets import mnist

from pyNN.utility import Timer

timer = Timer()
timer.start()

# setup the simulation
sim.setup(timestep=1.0)

simSetupTime = timer.diff() 

# sim.set_number_of_neurons_per_core(FSNeuron, 150)    

fsnn = load_model(filename="mnist_snn.json")

# Set the run time of the execution
runtime = fsnn.get_necessary_runtime()

fsnn.record()

buildCPUTime = timer.diff()

# run the simulation 
sim.run(runtime)

simCPUTime = timer.diff()

fsnn.get_data()
v      = fsnn.get_v() 
spikes = fsnn.get_spikes()
  
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(test_y)

(pred, score) = fsnn.evaluate(test_y[:1000], activation='softmax')
print(pred)

print("Accuracy: {}".format(score))
fsnn.summary()
print("Simulation setup time: {}".format(buildCPUTime))
print("build time: {}".format(buildCPUTime))
print("simulation time: {}".format(simCPUTime))
print("===================================")

sim.end()