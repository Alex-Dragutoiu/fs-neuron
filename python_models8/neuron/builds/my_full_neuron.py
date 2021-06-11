from spynnaker.pyNN.models.neuron import AbstractPyNNNeuronModelStandard    
from spynnaker.pyNN.models.defaults import default_initial_values

from python_models8.neuron.neuron_models.my_neuron_model import FSNeuronModel

from spynnaker.pyNN.models.neuron.input_types import InputTypeDelta
from spynnaker.pyNN.models.neuron.synapse_types import SynapseTypeDelta
from spynnaker.pyNN.models.neuron.threshold_types import ThresholdTypeStatic

class FSNeuron(AbstractPyNNNeuronModelStandard):

    @default_initial_values({"v", "ts", "t", "v_thresh", "isyn_exc", "isyn_inh"})
    def __init__(
            self,

            # neuron model parameters and state variables
            v=0.0,
            bias=0.0,
            ts=0.0,
            t=0.0,
            k=8,
            mul=0.0,

            # threshold types parameters
            v_thresh=0.0,

            # synapse type parameters
            isyn_exc=0.0,
            isyn_inh=0.0
        ):

        # create neuron model class
        neuron_model = FSNeuronModel(v, bias, ts, t, k, mul)

        # create synapse type model
        synapse_type = SynapseTypeDelta(isyn_exc, isyn_inh)

        # create input type model
        input_type = InputTypeDelta()

        # create threshold type model
        threshold_type = ThresholdTypeStatic(v_thresh)

        # Create the model using the superclass
        super(FSNeuron, self).__init__(
            # the model a name (shown in reports)
            model_name="FSNeuron",

            # the matching binary name
            binary="my_full_neuron_impl.aplx",

            # the various model types
            neuron_model=neuron_model, input_type=input_type,
            synapse_type=synapse_type, threshold_type=threshold_type)
