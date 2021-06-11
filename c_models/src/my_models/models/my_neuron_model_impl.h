#ifndef _NEURON_MODEL_MY_IMPL_H_
#define _NEURON_MODEL_MY_IMPL_H_

#include <neuron/models/neuron_model.h>

typedef struct neuron_t {
    //! membrane voltage [mV]
    REAL V;

    //! membrane voltage [mV]
    REAL Bias;

    //! timestep [ms]
    int32_t TS;

    //! variable that approximate an activation function [mV]
    REAL T;

    //! total number of timesteps to pass an integer value [ms]
    uint32_t K;

    // if we want to represent floating point numbers between [0, alpha)
    REAL Mul;
} neuron_t;

// often these are not user supplied, but computed parameters
typedef struct global_neuron_params_t { } global_neuron_params_t;

#endif // _NEURON_MODEL_MY_IMPL_H_
