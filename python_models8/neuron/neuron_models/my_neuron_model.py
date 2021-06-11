from spinn_front_end_common.utilities.constants import MICRO_TO_MILLISECOND_CONVERSION
from spinn_utilities.overrides import overrides
from data_specification.enums import DataType
from spynnaker.pyNN.models.neuron.neuron_models import AbstractNeuronModel
from spynnaker.pyNN.models.neuron.implementations import (AbstractStandardNeuronComponent)

# create constants to match the parameter names
V = "v"
Bias = "bias"
TS = "ts"
T = "t"
K = "k"
Mul = "mul"

# create units for each parameter
UNITS = {
    V: "mV",
    Bias: "mV",
    TS: "ms",
    T: "mV",
    K: "ms",
    Mul: "mV"
}

class FSNeuronModel(AbstractNeuronModel):
    __slots__ = [
        "__v", "__bias", "__ts", "__t", "__k", "__mul"
    ]

    def __init__(self, v, bias, ts, t, k, mul):

        # Update the data types - this must match the structs exactly
        super(FSNeuronModel, self).__init__(
            data_types=[
                DataType.S1615,   # V
                DataType.S1615,   # V_init
                DataType.UINT32,  # timestep
                DataType.S1615,   # T
                DataType.UINT32,  # K
                DataType.S1615    # UNIT 
            ])

        # Store any parameters and state variables
        self.__v = v
        self.__bias = bias
        self.__ts = ts
        self.__t = t
        self.__k = k
        self.__mul = mul

    @overrides(AbstractNeuronModel.get_n_cpu_cycles)
    def get_n_cpu_cycles(self, n_neurons):
        # Calculate (or guess) the CPU cycles
        # return self.__k * n_neurons
        return 150 * n_neurons

    @overrides(AbstractNeuronModel.add_parameters)
    def add_parameters(self, parameters):
        # Add initial values of the parameters that the user can change
        parameters[K] = self.__k
        parameters[Mul] = self.__mul
        parameters[Bias] = self.__bias

    @overrides(AbstractNeuronModel.add_state_variables)
    def add_state_variables(self, state_variables):
        # Add initial values of the state variables that the user can change
        state_variables[V] = self.__v
        state_variables[TS] = self.__ts
        state_variables[T] = self.__t

    @overrides(AbstractNeuronModel.get_units)
    def get_units(self, variable):
        # This works from the UNITS dict, so no changes are required
        return UNITS[variable]

    @overrides(AbstractNeuronModel.has_variable)
    def has_variable(self, variable):
        # This works from the UNITS dict, so no changes are required
        return variable in UNITS

    @overrides(AbstractNeuronModel.get_values)
    def get_values(self, parameters, state_variables, vertex_slice, ts):
        # Return, in order of the struct, the values from the parameters, state variables, or other
        return [
            state_variables[V], parameters[Bias], state_variables[TS], state_variables[T],
            parameters[K], parameters[Mul]
        ]

    @overrides(AbstractNeuronModel.update_values)
    def update_values(self, values, parameters, state_variables):
        # From the list of values given in order of the struct, update
        # the parameters and state variables
        (_v, _bias, _ts, _t, _k, _mul) = values

        # If you know that the value doesn't change, you don't have to
        # assign it (hint: often only state variables are likely to change)!
        state_variables[V] = _v
        state_variables[TS] = _ts
        state_variables[T] = _t

    # Add getters and setters for the parameters
    
    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, v):
        self.__v = v

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = bias

    @property
    def ts(self):
        return self.__ts

    @ts.setter
    def ts(self, ts):
        self.__ts = ts
    
    @property
    def t(self):
        return self.__t

    @t.setter
    def t(self, t):
        self.__t = t
    
    @property
    def k(self):
        return self.__k

    @k.setter
    def k(self, k):
        self.__k = k

    @property
    def mul(self):
        return self.__mul

    @mul.setter
    def mul(self, mul):
        self.__mul = mul
