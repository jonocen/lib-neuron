#ifndef MODELS_CORE_H
#define MODELS_CORE_H

#include "models_types.h"

int sequential_model_init(SequentialModel *model, int initial_capacity);
void sequential_model_free(SequentialModel *model);

int sequential_model_add_layer(SequentialModel *model, LayerPlugin layer);
int sequential_model_add_dense(SequentialModel *model,
                               int input_size,
                               int output_size,
                               Activation activation);
int sequential_model_add_conv2d(SequentialModel *model,
                                int input_width,
                                int input_height,
                                int input_channels,
                                int output_channels,
                                int kernel_width,
                                int kernel_height,
                                int stride,
                                int padding,
                                Activation activation);
int sequential_model_add_maxpool2d(SequentialModel *model,
                                   int input_width,
                                   int input_height,
                                   int channels,
                                   int pool_width,
                                   int pool_height,
                                   int stride,
                                   int padding);
int sequential_model_add_flatten(SequentialModel *model);

/* Set the current 2D input shape used by convenience conv/pool builders. */
int sequential_model_set_input_shape2d(SequentialModel *model,
                                       int width,
                                       int height,
                                       int channels);

/* Convenience Conv2D builder with default padding=kernel_size/2 and ACT_RELU. */
int sequential_model_add_conv2d_simple(SequentialModel *model,
                                       int input_channels,
                                       int output_channels,
                                       int kernel_size,
                                       int stride);

/* Convenience MaxPool2D builder with square pool and padding=0. */
int sequential_model_add_maxpool2d_simple(SequentialModel *model,
                                          int pool_size,
                                          int stride);

int sequential_model_forward(SequentialModel *model,
                             const float *input,
                             float *output);
int sequential_model_randomize(SequentialModel *model, float init_scale);
int sequential_model_predict(SequentialModel *model,
                             const float *input,
                             float *output);

int sequential_model_save_lnn(const SequentialModel *model,
                              const char *file_path);
int sequential_model_load_lnn(SequentialModel *model,
                              const char *file_path);

#endif /* MODELS_CORE_H */
