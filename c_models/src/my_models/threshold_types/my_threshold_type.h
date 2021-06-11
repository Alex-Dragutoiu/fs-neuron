#ifndef _MY_THRESHOLD_TYPE_H_
#define _MY_THRESHOLD_TYPE_H_

#include <neuron/threshold_types/threshold_type.h>

typedef struct threshold_type_t {
    REAL threshold_value; 
} threshold_type_t;


//! \brief Determines if the value given is above the threshold value
//! \param[in] value: The value to determine if it is above the threshold
//! \param[in] threshold_type: The parameters to use to determine the result
//! \return True if the neuron should fire
static inline bool threshold_type_is_above_threshold(state_t value, threshold_type_t *threshold_type) {
    /* Update to return true or false depending on if the threshold has been reached */
    return REAL_COMPARE(value, >=, threshold_type->threshold_value);
}

#endif // _MY_THRESHOLD_TYPE_H_
