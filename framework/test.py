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
node_id = sim.setup(timestep=1.0)
sim.set_number_of_neurons_per_core(FSNeuron, 100)    

# fsnn = load_model(filename='test_case2.json')
fsnn = load_model(filename='test_case_4.json')
# fsnn = load_model(filename='network_test_case1.json')

# Set the run time of the execution
runtime = fsnn.get_necessary_runtime(full_net=False)

fsnn.record()

buildCPUTime = timer.diff()

# run the simulation 
sim.run(runtime)

simCPUTime = timer.diff()

# extracts the all data from machine
fsnn.get_data()
spikes = fsnn.get_spikes()
v      = fsnn.get_v() 

(pred, score) = fsnn.evaluate([1.0, 2.0, 3.0, 1.0], activation='softmax')

print("Accuracy: {}".format(score))
print(pred)

fsnn.summary()
# print("Network Output: {}".format(fsnn.to_number(spikes[-1])))
print("===================================")
print("build time: {}".format(buildCPUTime))
print("simulation time: {}".format(simCPUTime))
print("===================================")

visualize(v, spikes, legend=True)

sim.end()
