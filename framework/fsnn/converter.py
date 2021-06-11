import json 
from fsnn.utils import flatten

class FSNNConverter:
    def __init__(self, k=0, alpha=0):
        self.__input = []
        self.__weights = [] 
        self.__biases = []

        self.__payload = {
            "k" : k,
            "alpha" : alpha,
            "topology" : [],
            "populations" : [],
            "projections": []
        }

    def convert(self, model):
        with open(model, 'w') as o:
            json.dump(self.__payload, o, indent=4)
        o.close()
    
    def set_input(self, input):
        self.__input = input

    def set_weights(self, weights):
        if len(self.__input) == 0:
            raise AttributeError('No input detected! Please provide input before setting the weights!')

        # extract biases from the weights array 
        for i in range(0, len(weights) - 1, 2):
            self.__biases.append(weights[i + 1])
            self.__weights.append(weights[i])

        # flatten the bias list 
        self.__biases = flatten(self.__biases)

        for w in self.__weights:
            self.__payload['topology'].append(len(w))
        self.__payload['topology'].append(len(self.__weights[-1][0]))

        self.__payload['populations'] = self.__set_populations()
        self.__payload['projections'] = self.__set_projections()

    def set_k(self, k):
        self.__payload['k'] = k

    def set_alpha(self, alpha):
        self.__payload['alpha'] = alpha

    def __set_projections(self):
        projections = []

        indent = 0
        l = 0
        while l < len(self.__payload['topology']) - 1:
            weights_on_layer = self.__weights[l]
            for src in range(indent, indent + self.__payload['topology'][l]):
                weights_of_neuron = weights_on_layer[src - indent]
                for i in range(self.__payload['topology'][l + 1]):
                    dst = indent + self.__payload['topology'][l] + i
                    connection_weight = weights_of_neuron[i]

                    if connection_weight < 0.0:
                        conn_type = "inhibitory"
                        connection_weight = abs(connection_weight)
                    else:
                        conn_type = "excitatory"
                    
                    projections.append({
                        "src" : src, 
                        "dst" : dst, 
                        "weight" : str(connection_weight), 
                        "type" : conn_type
                    })
            
            indent += self.__payload['topology'][l]
            l += 1

        return projections

    def __set_populations(self):
        populations = []

        # create the input layer
        input_neurons = self.__payload['topology'][0]   

        # for each input neuron  
        for n in range(input_neurons):
            values = []

            # for each input value per input neuron 
            for val in range(len(self.__input)):
                values.append(str(self.__input[val][n]))

            # create the afferent JSON 
            pop = { 
                "type" : { 
                    "name" : "spike_source", 
                    "values" :  values
                }, 
                "label" : "input_{}".format(n) 
            }

            # add to the current assembly of populations 
            populations.append(pop)

        bias_index = 0
        # create the hidden layers
        i = 0
        for layer in self.__payload['topology'][1:(len(self.__payload['topology']) - 1)]:
            for n in range(layer): 
                pop = { 
                    "type" : { 
                        "name" : "fs_neuron", 
                        "bias" : str(self.__biases[bias_index])
                    }, 
                    "label" : "hidden_{}_{}".format(i, n) 
                }
                populations.append(pop)
                bias_index += 1
            i += 1    

        # create the output layer
        for n in range(self.__payload['topology'][-1]):
            pop = { 
                "type" : { 
                    "name" : "fs_neuron",
                    "bias" : str(self.__biases[bias_index]) 
                }, 
                "label" : "output_{}".format(n) 
            }
            populations.append(pop)
            bias_index += 1    

        return populations