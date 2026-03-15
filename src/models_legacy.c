#include "../include/models_legacy.h"

#include "models_internal.h"

#include <stdlib.h>

int sequential_forward(Layer *layers, int num_layers, const float *input, float *output) {
    if (!layers || num_layers <= 0 || !input || !output) return -1;

    if (num_layers == 1) {
        return layer_forward(&layers[0], input, output);
    }

    int width = lnn_max_layer_width(layers, num_layers);
    float *buffer_a = malloc((size_t)width * sizeof(float));
    float *buffer_b = malloc((size_t)width * sizeof(float));
    if (!buffer_a || !buffer_b) {
        free(buffer_a);
        free(buffer_b);
        return -1;
    }

    const float *current_input = input;
    for (int i = 0; i < num_layers; i++) {
        int is_last = (i == num_layers - 1);
        float *current_output = is_last ? output : ((i % 2 == 0) ? buffer_a : buffer_b);

        if (layer_forward(&layers[i], current_input, current_output) != 0) {
            free(buffer_a);
            free(buffer_b);
            return -1;
        }

        current_input = current_output;
    }

    free(buffer_a);
    free(buffer_b);
    return 0;
}

int sequential_train_step(Layer *layers, int num_layers,
                          const float *input, const float *target,
                          float *output,
                          float **grads_w, float **grads_b,
                          LossFunctionType loss_function,
                          OptimizerType optimizer,
                          float learning_rate,
                          OptimizerState *optimizer_state,
                          float *loss_out) {
    if (!layers || num_layers <= 0 || !input || !target || !output || !grads_w || !grads_b) {
        return -1;
    }

    if (sequential_forward(layers, num_layers, input, output) != 0) {
        return -1;
    }

    return sequential_optimize_from_prediction(layers,
                                               num_layers,
                                               output,
                                               target,
                                               grads_w,
                                               grads_b,
                                               loss_function,
                                               optimizer,
                                               learning_rate,
                                               optimizer_state,
                                               loss_out);
}

int sequential_optimize_from_prediction(Layer *layers, int num_layers,
                                        const float *prediction, const float *target,
                                        float **grads_w, float **grads_b,
                                        LossFunctionType loss_function,
                                        OptimizerType optimizer,
                                        float learning_rate,
                                        OptimizerState *optimizer_state,
                                        float *loss_out) {
    if (!layers || num_layers <= 0 || !prediction || !target || !grads_w || !grads_b) {
        return -1;
    }

    if (!lnn_optimizer_state_valid(optimizer, optimizer_state)) {
        return -1;
    }

    int output_size = layers[num_layers - 1].output_size;

    int width = lnn_max_layer_width(layers, num_layers);
    float *delta_curr = malloc((size_t)width * sizeof(float));
    float *delta_prev = malloc((size_t)width * sizeof(float));
    if (!delta_curr || !delta_prev) {
        free(delta_curr);
        free(delta_prev);
        return -1;
    }

    if (lnn_compute_loss_and_grad(loss_function, prediction, target, output_size, loss_out, delta_curr) != 0) {
        free(delta_curr);
        free(delta_prev);
        return -1;
    }

    for (int i = num_layers - 1; i >= 0; i--) {
        float *next_delta = (i > 0) ? delta_prev : NULL;
        int grad_w_size = layers[i].output_size * layers[i].input_size;
        int grad_b_size = layers[i].output_size;

        if (!grads_w[i] || !grads_b[i]) {
            free(delta_curr);
            free(delta_prev);
            return -1;
        }

        if (layer_backward(&layers[i], delta_curr, next_delta, grads_w[i], grads_b[i]) != 0) {
            free(delta_curr);
            free(delta_prev);
            return -1;
        }

        float *opt_state_w_a = NULL;
        float *opt_state_w_b = NULL;
        float *opt_state_b_a = NULL;
        float *opt_state_b_b = NULL;

        if (optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_ADAMW) {
            if (!optimizer_state->m_w[i] || !optimizer_state->v_w[i] ||
                !optimizer_state->m_b[i] || !optimizer_state->v_b[i]) {
                free(delta_curr);
                free(delta_prev);
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_w_b = optimizer_state->v_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
            opt_state_b_b = optimizer_state->v_b[i];
        } else if (optimizer == OPTIMIZER_RMSPROP) {
            if (!optimizer_state->m_w[i] || !optimizer_state->m_b[i]) {
                free(delta_curr);
                free(delta_prev);
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
        } else if (optimizer == OPTIMIZER_ADAGRAD) {
            if (!optimizer_state->m_w[i] || !optimizer_state->m_b[i]) {
                free(delta_curr);
                free(delta_prev);
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
        }

        if (lnn_apply_optimizer_update(layers[i].weights,
                                       grads_w[i],
                                       grad_w_size,
                                       optimizer,
                                       learning_rate,
                                       optimizer_state,
                                       opt_state_w_a,
                                       opt_state_w_b) != 0 ||
            lnn_apply_optimizer_update(layers[i].biases,
                                       grads_b[i],
                                       grad_b_size,
                                       optimizer,
                                       learning_rate,
                                       optimizer_state,
                                       opt_state_b_a,
                                       opt_state_b_b) != 0) {
            free(delta_curr);
            free(delta_prev);
            return -1;
        }

        if (i > 0) {
            float *tmp = delta_curr;
            delta_curr = delta_prev;
            delta_prev = tmp;
        }
    }

    free(delta_curr);
    free(delta_prev);

    if (optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_ADAMW) {
        optimizer_state->step += 1;
    }

    return 0;
}
