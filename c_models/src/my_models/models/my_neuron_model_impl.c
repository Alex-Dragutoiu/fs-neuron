//! \file
//! \brief Few Spikes Neuron Model Implementation
#include "my_neuron_model_impl.h"

#include <debug.h>

void neuron_model_set_global_neuron_params(UNUSED const global_neuron_params_t *params) { }

state_t neuron_model_state_update(uint16_t num_excitatory_inputs, const input_t* exc_input,
                                  uint16_t num_inhibitory_inputs, const input_t* inh_input,
                                  input_t external_bias, neuron_t *restrict neuron) {

    // This takes the input and generates an input value, assumed to be a
    // current.  Note that the conversion to current from conductance is done
    // outside of this function, so does not need to be repeated here.

    REAL diff = -1.0;

    uint16_t t          = (neuron->TS % (neuron->K + 1));   
    uint16_t full_cycle = (neuron->TS % ((neuron->K + 1) * 2));    
    
    neuron->T           = (1 << (neuron->K - t)) * neuron->Mul;
    
    REAL total_exc = 0;
    REAL total_inh = 0;

    for (uint32_t i = 0; i < num_excitatory_inputs; i++) {
        total_exc += exc_input[i];
    }

    for (uint32_t i = 0; i < num_inhibitory_inputs; i++) {
        total_inh += inh_input[i];
    }

    input_t I = total_exc - total_inh + external_bias;

    // reset the membrane potential every 2K time steps to allow pipelining
    if (full_cycle == 0) {
        neuron->V = neuron->Bias;
    }
    
    if (full_cycle <= neuron->K) { 
        // accumulate synaptic input for the first K time steps 
        neuron->V += I * neuron->T;
    } else { 
        // send spikes for the next K time steps  
        diff = neuron->V - neuron->T;
    }

    neuron->TS++;

    return diff;
}

/* Get the state value representing the membrane voltage */
state_t neuron_model_get_membrane_voltage(const neuron_t *neuron) {
    return neuron->V;
}

/* We assume that the membrane potential v(t) has no leak, but is reset to v(t + 1) = v(t) − T(t) after a spike at time t */
void neuron_model_has_spiked(neuron_t *restrict neuron) {
    neuron->V = neuron->V - neuron->T;
}

void neuron_model_print_state_variables(const neuron_t *neuron) {
    // Print all state variables
    log_info("TS = %11.4k ms", neuron->TS);
    log_info("T = %11.4k mV", neuron->T);
    log_info("V = %11.4k mV", neuron->V);
}

void neuron_model_print_parameters(const neuron_t *neuron) { 
    log_info("K = %11.4k ms", neuron->K);
}