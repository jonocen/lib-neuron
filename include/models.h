#ifndef MODELS_H
#define MODELS_H

#include "lossfunctions.h"
#include "matrixcalculation.h"
#include "optimizers.h"

/*
 * Runs a forward pass through all layers.
 * `output` must have at least layers[num_layers - 1].output_size elements.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_forward(Layer *layers, int num_layers, const float *input, float *output);

/*
 * Performs one training step with MSE loss + SGD updates.
 * - `grads_w[i]` must point to a buffer of size output_size * input_size for layer i.
 * - `grads_b[i]` must point to a buffer of size output_size for layer i.
 * - `loss_out` is optional and may be NULL.
 * Returns 0 on success, -1 on invalid input or failure.
 */
int sequential_train_step_sgd(Layer *layers, int num_layers,
							  const float *input, const float *target,
							  float *output,
							  float **grads_w, float **grads_b,
							  float learning_rate,
							  float *loss_out);

#endif /* MODELS_H */
