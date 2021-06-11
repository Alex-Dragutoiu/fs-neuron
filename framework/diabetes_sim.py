# try:
#     import pyNN.spiNNaker as sim
# except Exception:
#     import spynnaker8 as sim

# from fsnn.snn import load_model
# from fsnn.utils import visualize
# from python_models8.neuron.builds.my_full_neuron import FSNeuron

# from pyNN.utility import Timer
# from numpy import loadtxt

# timer = Timer()
# timer.start()

# # setup the simulation
# sim.setup(timestep=1.0)

# simSetupTime = timer.diff()   

# # deserialize the snn from JSON and create the object
# fsnn = load_model(filename='diabetes_snn.json')

# # Set the run time of the execution 
# runtime = fsnn.get_necessary_runtime()

# fsnn.record()

# buildCPUTime = timer.diff()

# # run the simulation 
# sim.run(runtime)

# simCPUTime = timer.diff()

# fsnn.get_data()
# v      = fsnn.get_v()
# spikes = fsnn.get_spikes()

# fsnn.summary()

# Y = [1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.,
#      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.,
#      1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
#      0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
#      1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 1.,
#      0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
#      1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0.,
#      0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0.]

# (pred, score) = fsnn.evaluate(Y, activation='sigmoid')
# print("Predictions: {}".format(pred))
# print("Accuracy: {}".format(score))
# print("runtime: {}".format(runtime))
# print("Simulation setup time: {}".format(buildCPUTime))
# print("build time: {}".format(buildCPUTime))
# print("simulation time: {}".format(simCPUTime))
# print("===================================")

# visualize(v[18:19], spikes[:26]) # plot results 

# sim.end()


# plot inputs and outputs
import matplotlib.pyplot as plt
import numpy as np 
# rectified linear function
def rectified(x):
	return max(0.0, x)
 
# define a series of inputs
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [rectified(x) for x in series_in]
# line plot of raw inputs to rectified outputs

sigmoid = lambda x: 1 / (1 + np.exp(-x))
x=np.linspace(-10,10,10)
y=np.linspace(-10,10,100)


plt.xlabel('x')
plt.ylabel('y')
plt.title('Sigmoid')
plt.grid(linestyle='dotted')
plt.plot(y, sigmoid(y), color='purple', label='linspace(-10,10,100)')
# plt.plot(series_in, series_out, color='purple')
plt.show()


# suptitle('Sigmoid')

# text(4,0.8,r'$\sigma(x)=\frac{1}{1+e^{-x}}$',fontsize=15)

# legend(loc='lower right')

# gca().xaxis.set_major_locator(MultipleLocator(1))
# gca().yaxis.set_major_locator(MultipleLocator(0.1))

# legend(bbox_to_anchor=(0.5, -0.2), loc='center', borderaxespad=0)
