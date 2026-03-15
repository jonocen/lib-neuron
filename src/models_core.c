#include "../include/models_core.h"

#include "models_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LNN_MAGIC "LNN1"
#define LNN_MAGIC_SIZE 4

static int lnn_calc_out_dim(int in_dim, int kernel, int stride, int padding) {
    int padded;
    int numer;

    if (in_dim <= 0 || kernel <= 0 || stride <= 0 || padding < 0) return -1;

    padded = in_dim + 2 * padding;
    if (padded < kernel) return -1;

    numer = padded - kernel;
    if ((numer % stride) != 0) return -1;

    return (numer / stride) + 1;
}

int sequential_model_init(SequentialModel *model, int initial_capacity) {
    if (!model || initial_capacity <= 0) return -1;

    model->layers = calloc((size_t)initial_capacity, sizeof(LayerPlugin));
    if (!model->layers) return -1;

    model->num_layers = 0;
    model->capacity = initial_capacity;
    model->compiled = 0;
    model->compiled_loss = LOSS_MSE;
    model->compiled_optimizer = OPTIMIZER_SGD;
    model->compiled_learning_rate = 0.0f;
    model->compiled_owns_optimizer_state = 0;
    model->compiled_optimizer_state = (OptimizerState){0};
    model->work_forward_a = NULL;
    model->work_forward_b = NULL;
    model->work_forward_size = 0;
    model->work_delta_a = NULL;
    model->work_delta_b = NULL;
    model->work_delta_size = 0;
    model->work_grad_w = NULL;
    model->work_grad_b = NULL;
    model->work_grad_w_size = 0;
    model->work_grad_b_size = 0;
    model->shape2d_width = 0;
    model->shape2d_height = 0;
    model->shape2d_channels = 0;
    model->shape2d_valid = 0;
    return 0;
}

void sequential_model_free(SequentialModel *model) {
    if (!model) return;

    if (model->compiled_owns_optimizer_state) {
        sequential_model_optimizer_state_free(model, &model->compiled_optimizer_state);
        model->compiled_owns_optimizer_state = 0;
    }

    lnn_model_workspace_free(model);

    for (int i = 0; i < model->num_layers; i++) {
        layer_plugin_free(&model->layers[i]);
    }

    free(model->layers);
    model->layers = NULL;
    model->num_layers = 0;
    model->capacity = 0;
    model->compiled = 0;
}

int sequential_model_add_layer(SequentialModel *model, LayerPlugin layer) {
    if (!model || !lnn_plugin_layer_valid(&layer)) return -1;

    if (model->compiled_owns_optimizer_state) {
        sequential_model_optimizer_state_free(model, &model->compiled_optimizer_state);
        model->compiled_owns_optimizer_state = 0;
    }

    lnn_model_workspace_free(model);
    model->compiled = 0;

    if (model->num_layers >= model->capacity) {
        int new_capacity = model->capacity * 2;
        LayerPlugin *new_layers = realloc(model->layers, (size_t)new_capacity * sizeof(LayerPlugin));
        if (!new_layers) {
            return -1;
        }

        model->layers = new_layers;
        model->capacity = new_capacity;
    }

    model->layers[model->num_layers++] = layer;
    return 0;
}

int sequential_model_add_dense(SequentialModel *model,
                               int input_size,
                               int output_size,
                               Activation activation) {
    LayerPlugin layer;
    memset(&layer, 0, sizeof(LayerPlugin));

    if (!model) return -1;

    if (model->shape2d_valid) {
        int expected = model->shape2d_width * model->shape2d_height * model->shape2d_channels;
        if (expected != input_size) return -1;
    }

    if (layer_plugin_dense_create(input_size, output_size, activation, &layer) != 0) {
        return -1;
    }

    if (sequential_model_add_layer(model, layer) != 0) {
        layer_plugin_free(&layer);
        return -1;
    }

    model->shape2d_valid = 0;

    return 0;
}

int sequential_model_add_conv2d(SequentialModel *model,
                                int input_width,
                                int input_height,
                                int input_channels,
                                int output_channels,
                                int kernel_width,
                                int kernel_height,
                                int stride,
                                int padding,
                                Activation activation) {
    LayerPlugin layer;
    memset(&layer, 0, sizeof(LayerPlugin));

    if (!model) return -1;

    if (model->num_layers > 0) {
        if (!model->shape2d_valid) return -1;
        if (model->shape2d_width != input_width ||
            model->shape2d_height != input_height ||
            model->shape2d_channels != input_channels) {
            return -1;
        }
    }

    if (layer_plugin_conv2d_create(input_width,
                                   input_height,
                                   input_channels,
                                   output_channels,
                                   kernel_width,
                                   kernel_height,
                                   stride,
                                   padding,
                                   activation,
                                   &layer) != 0) {
        return -1;
    }

    if (sequential_model_add_layer(model, layer) != 0) {
        layer_plugin_free(&layer);
        return -1;
    }

    model->shape2d_width = lnn_calc_out_dim(input_width, kernel_width, stride, padding);
    model->shape2d_height = lnn_calc_out_dim(input_height, kernel_height, stride, padding);
    model->shape2d_channels = output_channels;
    model->shape2d_valid = (model->shape2d_width > 0 && model->shape2d_height > 0) ? 1 : 0;

    return 0;
}

int sequential_model_add_maxpool2d(SequentialModel *model,
                                   int input_width,
                                   int input_height,
                                   int channels,
                                   int pool_width,
                                   int pool_height,
                                   int stride,
                                   int padding) {
    LayerPlugin layer;
    memset(&layer, 0, sizeof(LayerPlugin));

    if (!model) return -1;

    if (model->num_layers > 0) {
        if (!model->shape2d_valid) return -1;
        if (model->shape2d_width != input_width ||
            model->shape2d_height != input_height ||
            model->shape2d_channels != channels) {
            return -1;
        }
    }

    if (layer_plugin_maxpool2d_create(input_width,
                                      input_height,
                                      channels,
                                      pool_width,
                                      pool_height,
                                      stride,
                                      padding,
                                      &layer) != 0) {
        return -1;
    }

    if (sequential_model_add_layer(model, layer) != 0) {
        layer_plugin_free(&layer);
        return -1;
    }

    model->shape2d_width = lnn_calc_out_dim(input_width, pool_width, stride, padding);
    model->shape2d_height = lnn_calc_out_dim(input_height, pool_height, stride, padding);
    model->shape2d_channels = channels;
    model->shape2d_valid = (model->shape2d_width > 0 && model->shape2d_height > 0) ? 1 : 0;

    return 0;
}

int sequential_model_add_flatten(SequentialModel *model) {
    LayerPlugin layer;
    int in_size;

    if (!model || model->num_layers <= 0) return -1;

    in_size = model->layers[model->num_layers - 1].output_size(
        model->layers[model->num_layers - 1].ctx);
    if (in_size <= 0) return -1;

    memset(&layer, 0, sizeof(LayerPlugin));
    if (layer_plugin_flatten_create(in_size, &layer) != 0) {
        return -1;
    }

    if (sequential_model_add_layer(model, layer) != 0) {
        layer_plugin_free(&layer);
        return -1;
    }

    model->shape2d_valid = 0;
    return 0;
}

int sequential_model_set_input_shape2d(SequentialModel *model,
                                       int width,
                                       int height,
                                       int channels) {
    if (!model || width <= 0 || height <= 0 || channels <= 0) return -1;
    model->shape2d_width = width;
    model->shape2d_height = height;
    model->shape2d_channels = channels;
    model->shape2d_valid = 1;
    return 0;
}

int sequential_model_add_conv2d_simple(SequentialModel *model,
                                       int input_channels,
                                       int output_channels,
                                       int kernel_size,
                                       int stride) {
    int padding;

    if (!model || !model->shape2d_valid) return -1;
    if (input_channels <= 0 || output_channels <= 0 || kernel_size <= 0 || stride <= 0) return -1;
    if (model->shape2d_channels != input_channels) return -1;

    padding = kernel_size / 2;

    return sequential_model_add_conv2d(model,
                                       model->shape2d_width,
                                       model->shape2d_height,
                                       input_channels,
                                       output_channels,
                                       kernel_size,
                                       kernel_size,
                                       stride,
                                       padding,
                                       ACT_RELU);
}

int sequential_model_add_maxpool2d_simple(SequentialModel *model,
                                          int pool_size,
                                          int stride) {
    if (!model || !model->shape2d_valid) return -1;
    if (pool_size <= 0 || stride <= 0) return -1;

    return sequential_model_add_maxpool2d(model,
                                          model->shape2d_width,
                                          model->shape2d_height,
                                          model->shape2d_channels,
                                          pool_size,
                                          pool_size,
                                          stride,
                                          0);
}

int sequential_model_forward(SequentialModel *model,
                             const float *input,
                             float *output) {
    if (!model || model->num_layers <= 0 || !input || !output) return -1;

    if (model->num_layers == 1) {
        return model->layers[0].forward(model->layers[0].ctx, input, output);
    }

    if (!model->work_forward_a || !model->work_forward_b) {
        int width = lnn_max_plugin_layer_width(model);
        if (lnn_ensure_workspace(&model->work_forward_a, &model->work_forward_size, width) != 0 ||
            lnn_ensure_workspace(&model->work_forward_b, &model->work_forward_size, width) != 0) {
            return -1;
        }
    }

    const float *current_input = input;
    for (int i = 0; i < model->num_layers; i++) {
        int is_last = (i == model->num_layers - 1);
        float *current_output = is_last ? output : ((i % 2 == 0) ? model->work_forward_a : model->work_forward_b);

        if (model->layers[i].forward(model->layers[i].ctx, current_input, current_output) != 0) {
            return -1;
        }

        current_input = current_output;
    }

    return 0;
}

int sequential_model_randomize(SequentialModel *model, float init_scale) {
    if (!model || model->num_layers <= 0 || init_scale <= 0.0f) return -1;

    for (int l = 0; l < model->num_layers; l++) {
        float *w = model->layers[l].weights(model->layers[l].ctx);
        float *b = model->layers[l].biases(model->layers[l].ctx);
        int nw = model->layers[l].weights_size(model->layers[l].ctx);
        int nb = model->layers[l].biases_size(model->layers[l].ctx);

        if (!w || !b || nw <= 0 || nb <= 0) return -1;

        for (int i = 0; i < nw; i++) {
            w[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * init_scale;
        }
        for (int i = 0; i < nb; i++) {
            b[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * init_scale;
        }
    }

    return 0;
}

int sequential_model_predict(SequentialModel *model,
                             const float *input,
                             float *output) {
    return sequential_model_forward(model, input, output);
}

int sequential_model_save_lnn(const SequentialModel *model,
                              const char *file_path) {
    if (!model || !file_path || !model->layers || model->num_layers <= 0) return -1;
    if (!lnn_has_lnn_extension(file_path)) return -1;

    FILE *file = fopen(file_path, "wb");
    if (!file) return -1;

    uint32_t num_layers = (uint32_t)model->num_layers;

    if (fwrite(LNN_MAGIC, 1, LNN_MAGIC_SIZE, file) != LNN_MAGIC_SIZE ||
        fwrite(&num_layers, sizeof(num_layers), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    for (int i = 0; i < model->num_layers; i++) {
        const LayerPlugin *layer = &model->layers[i];
        float *weights = layer->weights(layer->ctx);
        float *biases = layer->biases(layer->ctx);
        int weights_size = layer->weights_size(layer->ctx);
        int biases_size = layer->biases_size(layer->ctx);

        if (!weights || !biases || weights_size <= 0 || biases_size <= 0) {
            fclose(file);
            return -1;
        }

        uint32_t weights_size_u32 = (uint32_t)weights_size;
        uint32_t biases_size_u32 = (uint32_t)biases_size;

        if (fwrite(&weights_size_u32, sizeof(weights_size_u32), 1, file) != 1 ||
            fwrite(&biases_size_u32, sizeof(biases_size_u32), 1, file) != 1 ||
            fwrite(weights, sizeof(float), (size_t)weights_size, file) != (size_t)weights_size ||
            fwrite(biases, sizeof(float), (size_t)biases_size, file) != (size_t)biases_size) {
            fclose(file);
            return -1;
        }
    }

    if (fclose(file) != 0) return -1;
    return 0;
}

int sequential_model_load_lnn(SequentialModel *model,
                              const char *file_path) {
    if (!model || !file_path || !model->layers || model->num_layers <= 0) return -1;
    if (!lnn_has_lnn_extension(file_path)) return -1;

    FILE *file = fopen(file_path, "rb");
    if (!file) return -1;

    char magic[LNN_MAGIC_SIZE];
    uint32_t num_layers = 0;

    if (fread(magic, 1, LNN_MAGIC_SIZE, file) != LNN_MAGIC_SIZE ||
        fread(&num_layers, sizeof(num_layers), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    if (memcmp(magic, LNN_MAGIC, LNN_MAGIC_SIZE) != 0 ||
        num_layers != (uint32_t)model->num_layers) {
        fclose(file);
        return -1;
    }

    float **weights_snapshots = calloc((size_t)model->num_layers, sizeof(float *));
    float **biases_snapshots = calloc((size_t)model->num_layers, sizeof(float *));
    if (!weights_snapshots || !biases_snapshots) {
        lnn_free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
        fclose(file);
        return -1;
    }

    for (int i = 0; i < model->num_layers; i++) {
        LayerPlugin *layer = &model->layers[i];
        int expected_weights_size = layer->weights_size(layer->ctx);
        int expected_biases_size = layer->biases_size(layer->ctx);
        uint32_t file_weights_size = 0;
        uint32_t file_biases_size = 0;

        if (expected_weights_size <= 0 || expected_biases_size <= 0 ||
            fread(&file_weights_size, sizeof(file_weights_size), 1, file) != 1 ||
            fread(&file_biases_size, sizeof(file_biases_size), 1, file) != 1 ||
            file_weights_size != (uint32_t)expected_weights_size ||
            file_biases_size != (uint32_t)expected_biases_size) {
            lnn_free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
            fclose(file);
            return -1;
        }

        weights_snapshots[i] = malloc((size_t)expected_weights_size * sizeof(float));
        biases_snapshots[i] = malloc((size_t)expected_biases_size * sizeof(float));
        if (!weights_snapshots[i] || !biases_snapshots[i]) {
            lnn_free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
            fclose(file);
            return -1;
        }

        if (fread(weights_snapshots[i], sizeof(float), (size_t)expected_weights_size, file) != (size_t)expected_weights_size ||
            fread(biases_snapshots[i], sizeof(float), (size_t)expected_biases_size, file) != (size_t)expected_biases_size) {
            lnn_free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
            fclose(file);
            return -1;
        }
    }

    if (fgetc(file) != EOF) {
        lnn_free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
        fclose(file);
        return -1;
    }

    for (int i = 0; i < model->num_layers; i++) {
        LayerPlugin *layer = &model->layers[i];
        float *weights = layer->weights(layer->ctx);
        float *biases = layer->biases(layer->ctx);
        int weights_size = layer->weights_size(layer->ctx);
        int biases_size = layer->biases_size(layer->ctx);

        if (!weights || !biases || weights_size <= 0 || biases_size <= 0) {
            lnn_free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);
            fclose(file);
            return -1;
        }

        memcpy(weights, weights_snapshots[i], (size_t)weights_size * sizeof(float));
        memcpy(biases, biases_snapshots[i], (size_t)biases_size * sizeof(float));
    }

    lnn_free_layer_snapshots(weights_snapshots, biases_snapshots, model->num_layers);

    if (fclose(file) != 0) return -1;
    return 0;
}
