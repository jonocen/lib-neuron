#ifndef MODELS_LEGACY_H
#define MODELS_LEGACY_H

#include "models_types.h"

int sequential_forward(Layer *layers, int num_layers, const float *input, float *output);

int sequential_train_step(Layer *layers, int num_layers,
                          const float *input, const float *target,
                          float *output,
                          float **grads_w, float **grads_b,
                          LossFunctionType loss_function,
                          OptimizerType optimizer,
                          float learning_rate,
                          OptimizerState *optimizer_state,
                          float *loss_out);

int sequential_optimize_from_prediction(Layer *layers, int num_layers,
                                        const float *prediction, const float *target,
                                        float **grads_w, float **grads_b,
                                        LossFunctionType loss_function,
                                        OptimizerType optimizer,
                                        float learning_rate,
                                        OptimizerState *optimizer_state,
                                        float *loss_out);

#endif /* MODELS_LEGACY_H */
