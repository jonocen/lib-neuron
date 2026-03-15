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
