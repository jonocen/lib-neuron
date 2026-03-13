#include "../include/models.h"

#include <stdlib.h>
#include <string.h>

static int max_layer_width(const Layer *layers, int num_layers) {
    int max_width = 0;

    for (int i = 0; i < num_layers; i++) {
        if (layers[i].input_size > max_width) {
            max_width = layers[i].input_size;
        }
        if (layers[i].output_size > max_width) {
            max_width = layers[i].output_size;
        }
    }

    return max_width;
}

int sequential_forward(Layer *layers, int num_layers, const float *input, float *output) {
    if (!layers || num_layers <= 0 || !input || !output) return -1;

    if (num_layers == 1) {
        return layer_forward(&layers[0], input, output);
    }

    int width = max_layer_width(layers, num_layers);
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

int sequential_train_step_sgd(Layer *layers, int num_layers,
                              const float *input, const float *target,
                              float *output,
                              float **grads_w, float **grads_b,
                              float learning_rate,
                              float *loss_out) {
    if (!layers || num_layers <= 0 || !input || !target || !output || !grads_w || !grads_b) {
        return -1;
    }

    if (sequential_forward(layers, num_layers, input, output) != 0) {
        return -1;
    }

    int output_size = layers[num_layers - 1].output_size;
    if (loss_out) {
        *loss_out = loss_mse(output, target, output_size);
    }

    int width = max_layer_width(layers, num_layers);
    float *delta_curr = malloc((size_t)width * sizeof(float));
    float *delta_prev = malloc((size_t)width * sizeof(float));
    if (!delta_curr || !delta_prev) {
        free(delta_curr);
        free(delta_prev);
        return -1;
    }

    if (loss_mse_grad(output, target, output_size, delta_curr) != 0) {
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

        if (sgd_optimizer(layers[i].weights, grads_w[i], learning_rate, grad_w_size) != 0 ||
            sgd_optimizer(layers[i].biases, grads_b[i], learning_rate, grad_b_size) != 0) {
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
    return 0;
}