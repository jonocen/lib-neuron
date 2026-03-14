#include "../include/layers.h"

#include <stdlib.h>

static int dense_forward(void *ctx, const float *input, float *output) {
    return layer_forward((Layer *)ctx, input, output);
}

static int dense_backward(const void *ctx,
                         const float *delta_in,
                         float       *delta_out,
                         float       *grad_w,
                         float       *grad_b) {
    return layer_backward((const Layer *)ctx, delta_in, delta_out, grad_w, grad_b);
}

static int dense_input_size(const void *ctx) {
    return ((const Layer *)ctx)->input_size;
}

static int dense_output_size(const void *ctx) {
    return ((const Layer *)ctx)->output_size;
}

static float *dense_weights(void *ctx) {
    return ((Layer *)ctx)->weights;
}

static float *dense_biases(void *ctx) {
    return ((Layer *)ctx)->biases;
}

static int dense_weights_size(const void *ctx) {
    const Layer *layer = (const Layer *)ctx;
    return layer->input_size * layer->output_size;
}

static int dense_biases_size(const void *ctx) {
    return ((const Layer *)ctx)->output_size;
}

static void dense_destroy(void *ctx) {
    Layer *layer = (Layer *)ctx;
    if (!layer) return;
    layer_free(layer);
    free(layer);
}

static int conv2d_forward(void *ctx, const float *input, float *output) {
    return conv2d_layer_forward((Conv2DLayer *)ctx, input, output);
}

static int conv2d_backward(const void *ctx,
                          const float *delta_in,
                          float       *delta_out,
                          float       *grad_w,
                          float       *grad_b) {
    return conv2d_layer_backward((const Conv2DLayer *)ctx, delta_in, delta_out, grad_w, grad_b);
}

static int conv2d_input_size(const void *ctx) {
    const Conv2DLayer *layer = (const Conv2DLayer *)ctx;
    return layer->input_width * layer->input_height * layer->input_channels;
}

static int conv2d_output_size(const void *ctx) {
    const Conv2DLayer *layer = (const Conv2DLayer *)ctx;
    return layer->output_width * layer->output_height * layer->output_channels;
}

static float *conv2d_weights(void *ctx) {
    return ((Conv2DLayer *)ctx)->weights;
}

static float *conv2d_biases(void *ctx) {
    return ((Conv2DLayer *)ctx)->biases;
}

static int conv2d_weights_size(const void *ctx) {
    const Conv2DLayer *layer = (const Conv2DLayer *)ctx;
    return layer->output_channels *
           layer->input_channels *
           layer->kernel_height *
           layer->kernel_width;
}

static int conv2d_biases_size(const void *ctx) {
    return ((const Conv2DLayer *)ctx)->output_channels;
}

static void conv2d_destroy(void *ctx) {
    Conv2DLayer *layer = (Conv2DLayer *)ctx;
    if (!layer) return;
    conv2d_layer_free(layer);
    free(layer);
}

static int maxpool2d_forward(void *ctx, const float *input, float *output) {
    return maxpool2d_layer_forward((MaxPool2DLayer *)ctx, input, output);
}

static int maxpool2d_backward(const void *ctx,
                              const float *delta_in,
                              float       *delta_out,
                              float       *grad_w,
                              float       *grad_b) {
    return maxpool2d_layer_backward((const MaxPool2DLayer *)ctx, delta_in, delta_out, grad_w, grad_b);
}

static int maxpool2d_input_size(const void *ctx) {
    const MaxPool2DLayer *layer = (const MaxPool2DLayer *)ctx;
    return layer->input_width * layer->input_height * layer->channels;
}

static int maxpool2d_output_size(const void *ctx) {
    const MaxPool2DLayer *layer = (const MaxPool2DLayer *)ctx;
    return layer->output_width * layer->output_height * layer->channels;
}

static float *maxpool2d_weights(void *ctx) {
    return ((MaxPool2DLayer *)ctx)->dummy_weight;
}

static float *maxpool2d_biases(void *ctx) {
    return ((MaxPool2DLayer *)ctx)->dummy_bias;
}

static int maxpool2d_weights_size(const void *ctx) {
    (void)ctx;
    return 1;
}

static int maxpool2d_biases_size(const void *ctx) {
    (void)ctx;
    return 1;
}

static void maxpool2d_destroy(void *ctx) {
    MaxPool2DLayer *layer = (MaxPool2DLayer *)ctx;
    if (!layer) return;
    maxpool2d_layer_free(layer);
    free(layer);
}

int layer_plugin_dense_create(int input_size,
                              int output_size,
                              Activation activation,
                              LayerPlugin *out_plugin) {
    if (!out_plugin || input_size <= 0 || output_size <= 0) return -1;

    Layer *layer = malloc(sizeof(Layer));
    if (!layer) return -1;

    if (layer_init(layer, input_size, output_size, activation) != 0) {
        free(layer);
        return -1;
    }

    out_plugin->ctx = layer;
    out_plugin->forward = dense_forward;
    out_plugin->backward = dense_backward;
    out_plugin->input_size = dense_input_size;
    out_plugin->output_size = dense_output_size;
    out_plugin->weights = dense_weights;
    out_plugin->biases = dense_biases;
    out_plugin->weights_size = dense_weights_size;
    out_plugin->biases_size = dense_biases_size;
    out_plugin->destroy = dense_destroy;

    return 0;
}

int layer_plugin_conv2d_create(int input_width,
                               int input_height,
                               int input_channels,
                               int output_channels,
                               int kernel_width,
                               int kernel_height,
                               int stride,
                               int padding,
                               Activation activation,
                               LayerPlugin *out_plugin) {
    if (!out_plugin) return -1;

    Conv2DLayer *layer = calloc(1, sizeof(Conv2DLayer));
    if (!layer) return -1;

    if (conv2d_layer_init(layer,
                          input_width,
                          input_height,
                          input_channels,
                          output_channels,
                          kernel_width,
                          kernel_height,
                          stride,
                          padding,
                          activation) != 0) {
        free(layer);
        return -1;
    }

    out_plugin->ctx = layer;
    out_plugin->forward = conv2d_forward;
    out_plugin->backward = conv2d_backward;
    out_plugin->input_size = conv2d_input_size;
    out_plugin->output_size = conv2d_output_size;
    out_plugin->weights = conv2d_weights;
    out_plugin->biases = conv2d_biases;
    out_plugin->weights_size = conv2d_weights_size;
    out_plugin->biases_size = conv2d_biases_size;
    out_plugin->destroy = conv2d_destroy;

    return 0;
}

int layer_plugin_maxpool2d_create(int input_width,
                                  int input_height,
                                  int channels,
                                  int pool_width,
                                  int pool_height,
                                  int stride,
                                  int padding,
                                  LayerPlugin *out_plugin) {
    if (!out_plugin) return -1;

    MaxPool2DLayer *layer = calloc(1, sizeof(MaxPool2DLayer));
    if (!layer) return -1;

    if (maxpool2d_layer_init(layer,
                             input_width,
                             input_height,
                             channels,
                             pool_width,
                             pool_height,
                             stride,
                             padding) != 0) {
        free(layer);
        return -1;
    }

    out_plugin->ctx = layer;
    out_plugin->forward = maxpool2d_forward;
    out_plugin->backward = maxpool2d_backward;
    out_plugin->input_size = maxpool2d_input_size;
    out_plugin->output_size = maxpool2d_output_size;
    out_plugin->weights = maxpool2d_weights;
    out_plugin->biases = maxpool2d_biases;
    out_plugin->weights_size = maxpool2d_weights_size;
    out_plugin->biases_size = maxpool2d_biases_size;
    out_plugin->destroy = maxpool2d_destroy;

    return 0;
}

void layer_plugin_free(LayerPlugin *plugin) {
    if (!plugin) return;

    if (plugin->destroy) {
        plugin->destroy(plugin->ctx);
    }

    plugin->ctx = NULL;
    plugin->forward = NULL;
    plugin->backward = NULL;
    plugin->input_size = NULL;
    plugin->output_size = NULL;
    plugin->weights = NULL;
    plugin->biases = NULL;
    plugin->weights_size = NULL;
    plugin->biases_size = NULL;
    plugin->destroy = NULL;
}
