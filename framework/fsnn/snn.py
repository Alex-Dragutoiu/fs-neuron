import json

try:
    import pyNN.spiNNaker as sim
except Exception:
    import spynnaker8 as sim

import numpy as np
from python_models8.neuron.builds.my_full_neuron import FSNeuron

from fsnn.utils import sigmoid, softmax, flatten

from sklearn.metrics import accuracy_score, mean_absolute_error

class FSNN:
    __slots__ = [
        "__k", "__mul", "__topology", "__spikes", "__v", "__populations", "__projections", "__input_depth" 
    ]

    def __init__(self, k, alpha, topology, populations, projections):
        self.__k = k
        self.__mul = alpha / 2**k
        self.__spikes = []
        self.__v = []

        self.__topology = topology
        self.__populations = []
        self.__projections = []

        # determines how many networks are processed in the same simulation  
        self.__input_depth = len(populations[0]['type']['values'])
        for pop in populations:
            label = pop['label']
            if pop['type']['name'] == 'spike_source':                
                spikes = []
                offset = 0.0

                for i in pop['type']['values']:
                    spikes.extend(self.__to_spike_train(float(i), offset=offset))
                    offset += (k + 1) * 2

                self.__spikes.append(spikes)
                self.__populations.append(sim.Population(1, sim.SpikeSourceArray(spike_times=spikes), label=label))
            else:  
                temp_pop = sim.Population(1, FSNeuron(bias=float(pop['type']['bias']), k=k, mul=self.__mul), label=label)
                self.__populations.append(temp_pop)

        # set the projections
        for proj in projections:
            src = proj['src']
            dst = proj['dst']
            connector = sim.OneToOneConnector()
        
            if 0 <= src < self.__topology[0]:
                delay = 1.0
            else: 
                delay = self.__k + 1

            self.__projections.append(sim.Projection(self.__populations[src], self.__populations[dst], 
                                      connector,
                                      synapse_type=sim.StaticSynapse(weight=float(proj['weight']), delay=delay),
                                      receptor_type=proj['type']))
           
    def get_output_layer(self):
        fro = sum(self.__topology) - self.__topology[-1]
        to = len(self.__populations)

        return self.__populations[fro:to]

    def get_input_layer(self):  
        fro = 0
        to = self.__topology[0] - 1   

        return self.__populations[fro:to]

    def get_hidden_layers(self):
        fro = self.__topology[0] - 1
        to = sum(self.__topology) - self.__topology[-1]

        return self.__populations[fro:to]

    def predict(self, input_index, activation):
        prediction = []

        fro = sum(self.__topology[1:]) - self.__topology[-1]
        to = len(self.__v)
    
        for v_output in self.__v[fro:to]:
            index =  (self.__k + 1) * (2 * (len(self.__topology[1:]) + input_index) - 1) 
            prediction.append(v_output[index][0])
        result = []
        if activation == 'sigmoid':
            result.append(sigmoid(prediction[0])) 
        elif activation == 'softmax':
            result.append(softmax(prediction))
        else:
            raise ValueError("Unknown activation provided!")
        return result 

    def evaluate(self, target, activation, metrics=True):    
        pred = [] 
        for i in range(self.get_input_depth()):
            if activation == 'sigmoid':
                pred.append(round(self.predict(i, activation=activation)[0]))
            else:
                pred.append(np.argmax(self.predict(i, activation=activation)))
        
        if activation == 'sigmoid':
            score = 100 - mean_absolute_error(target, pred) * 100.0
        else:
            score = accuracy_score(target, pred) * 100.0
        
        return (pred, score)    

    def get_topology(self):
        return self.__topology

    def get_populations(self):
        return self.__populations

    def get_population(self, index):
        return self.__populations[index]

    def get_projections(self): 
        return self.__projections

    def get_k(self):
        return self.__k

    def get_mul(self):
        return self.__mul

    def get_input_depth(self):
        return self.__input_depth

    def get_data(self):
        for n in self.__populations[self.__topology[0]:]:
            s = n.get_data('spikes')
            v = n.get_data('v')
            self.__v.append(v.segments[0].filter(name='v')[0].magnitude)
            self.__spikes.append(s.segments[0].spiketrains[0].magnitude)

    def get_spikes(self):
        return self.__spikes

    def get_v(self):
        return self.__v

    def record(self, layer='output'):
        for n in self.__populations[self.__topology[0]:]:
            n.record(['v', 'spikes'])

    def get_necessary_runtime(self, full_net=False):
        if full_net is True:
            return 1 + (self.get_k() + 1) * (2 * (self.get_input_depth() + len(self.__topology[1:])) - 2)
        return 1 + (self.get_k() + 1) * (2 * (self.get_input_depth() + len(self.__topology[1:])) - 3)

    def summary(self):
        total_spikes = 0
        for spike_train in self.__spikes[:10]:
            total_spikes += len(spike_train)   

        print("===============STATS===============")
        print("= K                      : %d ms" % self.__k)
        print("= multiplier             : %f ms" % self.__mul)
        print("= Number of neurons      : %d" % len(self.__populations))
        print("= Number of connections  : %d" % len(self.__projections))
        print("= Total Number of spikes : %d ms" % total_spikes)
        print("= runtime                : %d ms" % self.get_necessary_runtime())
        print("===================================")

    def __to_spike_train(self, value, offset):
        if value >= 2**self.__k:
            raise ValueError("the decimal value {} must not exceed 2^k!".format(value))
        
        spike_train = []
        spike_time = 0.0

        exp = self.__k - 1
        while exp >= 0:
            temp = value - (2**exp * self.__mul) 
            if temp >= 0:
                value = temp 
                spike_train.append(spike_time + offset)
            spike_time += 1
            exp -= 1

        return spike_train
    
    def to_number(self, spike_train):
        value = 0.0
        
        for spike in spike_train:
            pow = self.__k - (spike % (self.__k + 1))
            value += (2**pow * self.__mul)  

        return value 

def load_model(filename):
    with open(filename, 'r') as content:
        snn_dict = json.load(content)
        content.close()

    return FSNN(**snn_dict)     