#include "../include/matrixcalculation.h"
#include <stdlib.h>
#include <string.h>

/* Activation functions */

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
 
int layer_init(Layer *layer, int input_size, int output_size, Activation activation) {
    if (!layer) return -1;
    layer->input_size  = input_size;
    layer->output_size = output_size;
    layer->activation  = activation;
    layer->weights     = calloc(output_size * input_size, sizeof(float));
    layer->biases      = calloc(output_size,              sizeof(float));
    layer->cache_input = calloc(input_size,               sizeof(float));
    layer->cache_z     = calloc(output_size,              sizeof(float));
    if (!layer->weights || !layer->biases || !layer->cache_input || !layer->cache_z) {
        layer_free(layer);
        return -1;
    }
    return 0;
}

void layer_free(Layer *layer) {
    if (!layer) return;
    free(layer->weights);     layer->weights     = NULL;
    free(layer->biases);      layer->biases      = NULL;
    free(layer->cache_input); layer->cache_input = NULL;
    free(layer->cache_z);     layer->cache_z     = NULL;
}

/* Forward pass */
int layer_forward(Layer *layer, const float *input, float *output) {
    if (!layer || !input || !output) return -1;
    int in  = layer->input_size;
    int out = layer->output_size;

    memcpy(layer->cache_input, input, in * sizeof(float));

    for (int i = 0; i < out; i++) {
        float z = layer->biases[i];
        for (int j = 0; j < in; j++)
            z += layer->weights[i * in + j] * input[j];
        layer->cache_z[i] = z;
        output[i]         = act_apply(z, layer->activation);
    }
    return 0;
}

/* Backpropagation */

int layer_backward(const Layer *layer,
                   const float *delta_in,
                   float       *delta_out,
                   float       *grad_w,
                   float       *grad_b) {
    if (!layer || !delta_in || !grad_w || !grad_b) return -1;
    int in  = layer->input_size;
    int out = layer->output_size;

    float *delta = malloc(out * sizeof(float));
    if (!delta) return -1;

    /* apply activation derivative */
    for (int i = 0; i < out; i++)
        delta[i] = delta_in[i] * act_deriv(layer->cache_z[i], layer->activation);

    /* weight and bias gradients */
    for (int i = 0; i < out; i++) {
        grad_b[i] = delta[i];
        for (int j = 0; j < in; j++)
            grad_w[i * in + j] = delta[i] * layer->cache_input[j];
    }

    /* propagate error to previous layer: delta_out = W^T * delta */
    if (delta_out) {
        for (int j = 0; j < in; j++) {
            float sum = 0.0f;
            for (int i = 0; i < out; i++)
                sum += layer->weights[i * in + j] * delta[i];
            delta_out[j] = sum;
        }
    }

    free(delta);
    return 0;
}