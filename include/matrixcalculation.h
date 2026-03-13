#ifndef MATRIXCALCULATION_H
#define MATRIXCALCULATION_H

#include <math.h>

typedef enum {
	ACT_LINEAR,
	ACT_RELU,
	ACT_SIGMOID,
	ACT_TANH
} Activation;

float act_apply(float x, Activation a);  /* f(x)  */
float act_deriv(float x, Activation a);  /* f'(x) */

typedef struct {
	int        input_size;
	int        output_size;
	Activation activation;
	float     *weights;
	float     *biases;
	float     *cache_input;
	float     *cache_z;
} Layer;

/* Allocates all internal buffers. Returns 0 on success, -1 on failure. */
int  layer_init(Layer *layer, int input_size, int output_size, Activation activation);
void layer_free(Layer *layer);

/* Forward pass: output = activation(weights * input + biases). */
int layer_forward(Layer *layer, const float *input, float *output);

/* Backward pass for one layer. */
int layer_backward(const Layer *layer,
				   const float *delta_in,
				   float       *delta_out,
				   float       *grad_w,
				   float       *grad_b);

#endif /* MATRIXCALCULATION_H */
