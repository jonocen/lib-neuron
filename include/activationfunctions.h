#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <math.h>

typedef enum {
	ACT_LINEAR,
	ACT_RELU,
	ACT_SIGMOID,
	ACT_TANH
} Activation;

float act_apply(float x, Activation a);  /* f(x)  */
float act_deriv(float x, Activation a);  /* f'(x) */

#endif /* ACTIVATIONFUNCTIONS_H */
