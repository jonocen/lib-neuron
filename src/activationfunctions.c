#include "../include/activationfunctions.h"

float act_apply(float x, Activation a) {
    switch (a) {
        case ACT_RELU:    return x > 0.0f ? x : 0.0f;
        case ACT_SIGMOID: return 1.0f / (1.0f + expf(-x));
        case ACT_TANH:    return tanhf(x);
        case ACT_LINEAR:  /* fall-through */
        default:          return x;
    }
}

float act_deriv(float x, Activation a) {
    float s;
    switch (a) {
        case ACT_RELU:    return x > 0.0f ? 1.0f : 0.0f;
        case ACT_SIGMOID: s = act_apply(x, ACT_SIGMOID); return s * (1.0f - s);
        case ACT_TANH:    s = tanhf(x); return 1.0f - s * s;
        case ACT_LINEAR:  /* fall-through */
        default:          return 1.0f;
    }
}
