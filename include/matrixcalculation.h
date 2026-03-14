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

typedef struct {
	int        input_width;
	int        input_height;
	int        input_channels;
	int        output_channels;
	int        kernel_width;
	int        kernel_height;
	int        stride;
	int        padding;
	int        output_width;
	int        output_height;
	Activation activation;
	float     *weights;
	float     *biases;
	float     *cache_input;
	float     *cache_z;
} Conv2DLayer;

typedef struct {
	int    input_width;
	int    input_height;
	int    channels;
	int    pool_width;
	int    pool_height;
	int    stride;
	int    padding;
	int    output_width;
	int    output_height;
	float *cache_input;
	int   *cache_max_indices;
	/* Dummy params keep plugin/optimizer pipeline compatible. */
	float *dummy_weight;
	float *dummy_bias;
} MaxPool2DLayer;

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

/* Allocates all internal buffers for Conv2D layer. Returns 0 on success. */
int conv2d_layer_init(Conv2DLayer *layer,
				  int input_width,
				  int input_height,
				  int input_channels,
				  int output_channels,
				  int kernel_width,
				  int kernel_height,
				  int stride,
				  int padding,
				  Activation activation);

void conv2d_layer_free(Conv2DLayer *layer);

/* Forward pass on flattened CHW input/output buffers. */
int conv2d_layer_forward(Conv2DLayer *layer, const float *input, float *output);

/* Backward pass for Conv2D layer on flattened CHW tensors. */
int conv2d_layer_backward(const Conv2DLayer *layer,
					  const float *delta_in,
					  float       *delta_out,
					  float       *grad_w,
					  float       *grad_b);

/* Allocates all internal buffers for MaxPool2D layer. Returns 0 on success. */
int maxpool2d_layer_init(MaxPool2DLayer *layer,
				 int input_width,
				 int input_height,
				 int channels,
				 int pool_width,
				 int pool_height,
				 int stride,
				 int padding);

void maxpool2d_layer_free(MaxPool2DLayer *layer);

/* Forward pass on flattened CHW input/output buffers. */
int maxpool2d_layer_forward(MaxPool2DLayer *layer, const float *input, float *output);

/* Backward pass for MaxPool2D layer on flattened CHW tensors. */
int maxpool2d_layer_backward(const MaxPool2DLayer *layer,
				     const float *delta_in,
				     float       *delta_out,
				     float       *grad_w,
				     float       *grad_b);

#endif /* MATRIXCALCULATION_H */
