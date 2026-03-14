#ifndef LAYERS_H
#define LAYERS_H

#include "matrixcalculation.h"

typedef struct {
    void *ctx;

    int (*forward)(void *ctx, const float *input, float *output);
    int (*backward)(const void *ctx,
                    const float *delta_in,
                    float       *delta_out,
                    float       *grad_w,
                    float       *grad_b);

    int (*input_size)(const void *ctx);
    int (*output_size)(const void *ctx);

    float *(*weights)(void *ctx);
    float *(*biases)(void *ctx);
    int    (*weights_size)(const void *ctx);
    int    (*biases_size)(const void *ctx);

    void (*destroy)(void *ctx);
} LayerPlugin;

/*
 * Creates a dense (fully-connected) layer plugin backed by `Layer`.
 * On success, `out_plugin` owns the created layer and must be freed with
 * `layer_plugin_free`.
 * Returns 0 on success, -1 on failure.
 */
int layer_plugin_dense_create(int input_size,
                              int output_size,
                              Activation activation,
                              LayerPlugin *out_plugin);

/*
 * Creates a Conv2D layer plugin backed by `Conv2DLayer`.
 * Input and output buffers are flattened in CHW order.
 * Returns 0 on success, -1 on invalid geometry or allocation failure.
 */
int layer_plugin_conv2d_create(int input_width,
                               int input_height,
                               int input_channels,
                               int output_channels,
                               int kernel_width,
                               int kernel_height,
                               int stride,
                               int padding,
                               Activation activation,
                               LayerPlugin *out_plugin);

/*
 * Creates a MaxPool2D layer plugin backed by `MaxPool2DLayer`.
 * Input and output buffers are flattened in CHW order.
 * Returns 0 on success, -1 on invalid geometry or allocation failure.
 */
int layer_plugin_maxpool2d_create(int input_width,
                                  int input_height,
                                  int channels,
                                  int pool_width,
                                  int pool_height,
                                  int stride,
                                  int padding,
                                  LayerPlugin *out_plugin);

/*
 * Frees resources owned by the plugin and resets function pointers.
 * Safe to call on partially initialized plugins.
 */
void layer_plugin_free(LayerPlugin *plugin);

#endif /* LAYERS_H */
