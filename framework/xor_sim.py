try:
    import pyNN.spiNNaker as sim
except Exception:
    import spynnaker8 as sim

from fsnn.snn import load_model
from fsnn.utils import visualize
from python_models8.neuron.builds.my_full_neuron import FSNeuron

from pyNN.utility import Timer

timer = Timer()
timer.start()

# setup the simulation
sim.setup(timestep=1.0)

simSetupTime = timer.diff() 

sim.set_number_of_neurons_per_core(FSNeuron, 150)    

# deserialize the snn from JSON and create the object
fsnn = load_model(filename='xor_snn.json')

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

fsnn.summary()

(pred, score) = fsnn.evaluate([0.06128418, 0.9892666, 0.989153,0.00989598], activation='sigmoid')

print("Predictions: {}".format(pred))
print("Accuracy: {}".format(score))
print("runtime: {}".format(runtime))
print("Simulation setup time: {}".format(buildCPUTime))
print("build time: {}".format(buildCPUTime))
print("simulation time: {}".format(simCPUTime))
print("===================================")

visualize(v[3:10], spikes[:18]) # plot results 

sim.end()
