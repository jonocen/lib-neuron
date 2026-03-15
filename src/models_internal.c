#include "models_internal.h"

#include <stdlib.h>
#include <string.h>

int lnn_has_lnn_extension(const char *file_path) {
    if (!file_path) return 0;

    const char *dot = strrchr(file_path, '.');
    if (!dot) return 0;

    return strcmp(dot, ".lnn") == 0;
}

void lnn_free_layer_snapshots(float **weights, float **biases, int num_layers) {
    if (weights) {
        for (int i = 0; i < num_layers; i++) {
            free(weights[i]);
        }
        free(weights);
    }

    if (biases) {
        for (int i = 0; i < num_layers; i++) {
            free(biases[i]);
        }
        free(biases);
    }
}

int lnn_plugin_layer_valid(const LayerPlugin *layer) {
    if (!layer) return 0;
    return layer->ctx &&
           layer->forward &&
           layer->backward &&
           layer->input_size &&
           layer->output_size &&
           layer->weights &&
           layer->biases &&
           layer->weights_size &&
           layer->biases_size &&
           layer->destroy;
}

int lnn_max_plugin_layer_width(const SequentialModel *model) {
    int max_width = 0;

    for (int i = 0; i < model->num_layers; i++) {
        int in = model->layers[i].input_size(model->layers[i].ctx);
        int out = model->layers[i].output_size(model->layers[i].ctx);

        if (in > max_width) {
            max_width = in;
        }
        if (out > max_width) {
            max_width = out;
        }
    }

    return max_width;
}

void lnn_model_workspace_free(SequentialModel *model) {
    if (!model) return;

    free(model->work_forward_a);
    free(model->work_forward_b);
    free(model->work_delta_a);
    free(model->work_delta_b);
    free(model->work_grad_w);
    free(model->work_grad_b);
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
}

int lnn_ensure_workspace(float **buffer, int *current_size, int required_size) {
    if (!buffer || !current_size || required_size <= 0) return -1;

    if (*buffer && *current_size >= required_size) {
        return 0;
    }

    float *new_buffer = realloc(*buffer, (size_t)required_size * sizeof(float));
    if (!new_buffer) return -1;

    *buffer = new_buffer;
    *current_size = required_size;
    return 0;
}

void lnn_free_grad_accumulators(float **acc_w, float **acc_b, int num_layers) {
    if (acc_w) {
        for (int i = 0; i < num_layers; i++) {
            free(acc_w[i]);
        }
        free(acc_w);
    }

    if (acc_b) {
        for (int i = 0; i < num_layers; i++) {
            free(acc_b[i]);
        }
        free(acc_b);
    }
}

static int adam_optimizer_state_valid(const OptimizerState *optimizer_state) {
    if (!optimizer_state) return 0;
    if (!optimizer_state->m_w || !optimizer_state->v_w || !optimizer_state->m_b || !optimizer_state->v_b) return 0;
    if (optimizer_state->step <= 0) return 0;
    if (optimizer_state->beta1 <= 0.0f || optimizer_state->beta1 >= 1.0f) return 0;
    if (optimizer_state->beta2 <= 0.0f || optimizer_state->beta2 >= 1.0f) return 0;
    return 1;
}

static int rmsprop_optimizer_state_valid(const OptimizerState *optimizer_state) {
    if (!optimizer_state) return 0;
    if (!optimizer_state->m_w || !optimizer_state->m_b) return 0;
    if (optimizer_state->beta1 <= 0.0f || optimizer_state->beta1 >= 1.0f) return 0;
    return 1;
}

static int adagrad_optimizer_state_valid(const OptimizerState *optimizer_state) {
    if (!optimizer_state) return 0;
    if (!optimizer_state->m_w || !optimizer_state->m_b) return 0;
    return 1;
}

static int adamw_optimizer_state_valid(const OptimizerState *optimizer_state) {
    if (!optimizer_state) return 0;
    if (!optimizer_state->m_w || !optimizer_state->v_w || !optimizer_state->m_b || !optimizer_state->v_b) return 0;
    if (optimizer_state->step <= 0) return 0;
    if (optimizer_state->beta1 <= 0.0f || optimizer_state->beta1 >= 1.0f) return 0;
    if (optimizer_state->beta2 <= 0.0f || optimizer_state->beta2 >= 1.0f) return 0;
    return 1;
}

int lnn_optimizer_state_valid(OptimizerType optimizer,
                              const OptimizerState *optimizer_state) {
    if (optimizer == OPTIMIZER_ADAM) {
        return adam_optimizer_state_valid(optimizer_state);
    }
    if (optimizer == OPTIMIZER_RMSPROP) {
        return rmsprop_optimizer_state_valid(optimizer_state);
    }
    if (optimizer == OPTIMIZER_ADAGRAD) {
        return adagrad_optimizer_state_valid(optimizer_state);
    }
    if (optimizer == OPTIMIZER_ADAMW) {
        return adamw_optimizer_state_valid(optimizer_state);
    }
    return 1;
}

int lnn_apply_optimizer_update(float *weights,
                               const float *grads,
                               int size,
                               OptimizerType optimizer,
                               float learning_rate,
                               OptimizerState *optimizer_state,
                               float *opt_state_a,
                               float *opt_state_b) {
    if (!weights || !grads || size <= 0) return -1;

    if (optimizer == OPTIMIZER_SGD) {
        return sgd_optimizer(weights, (float *)grads, learning_rate, size);
    }
    if (optimizer == OPTIMIZER_ADAM) {
        if (!optimizer_state || !opt_state_a || !opt_state_b) return -1;
        return adam_optimizer(weights,
                              (float *)grads,
                              opt_state_a,
                              opt_state_b,
                              optimizer_state->beta1,
                              optimizer_state->beta2,
                              learning_rate,
                              optimizer_state->step,
                              size);
    }

    if (optimizer == OPTIMIZER_RMSPROP) {
        if (!optimizer_state || !opt_state_a) return -1;
        return rmsprop_optimizer(weights,
                                 (float *)grads,
                                 opt_state_a,
                                 optimizer_state->beta1,
                                 learning_rate,
                                 size);
    }

    if (optimizer == OPTIMIZER_ADAGRAD) {
        if (!optimizer_state || !opt_state_a) return -1;
        return adagrad_optimizer(weights,
                                 (float *)grads,
                                 opt_state_a,
                                 learning_rate,
                                 size);
    }

    if (optimizer == OPTIMIZER_ADAMW) {
        if (!optimizer_state || !opt_state_a || !opt_state_b) return -1;
        return adamw_optimizer(weights,
                               (float *)grads,
                               opt_state_a,
                               opt_state_b,
                               optimizer_state->beta1,
                               optimizer_state->beta2,
                               learning_rate,
                               optimizer_state->step,
                               size);
    }

    return -1;
}

int lnn_compute_loss_and_grad(LossFunctionType loss_function,
                              const float *prediction,
                              const float *target,
                              int size,
                              float *loss_out,
                              float *grad_out) {
    const float huber_delta = 1.0f;

    if (!prediction || !target || !grad_out || size <= 0) return -1;

    if (loss_function == LOSS_MSE) {
        if (loss_out) {
            *loss_out = loss_mse(prediction, target, size);
        }
        return loss_mse_grad(prediction, target, size, grad_out);
    }

    if (loss_function == LOSS_BCE) {
        if (loss_out) {
            *loss_out = loss_bce(prediction, target, size);
        }
        return loss_bce_grad(prediction, target, size, grad_out);
    }

    if (loss_function == LOSS_HUBER) {
        if (loss_out) {
            *loss_out = loss_huber(prediction, target, size, huber_delta);
        }
        return loss_huber_grad(prediction, target, size, huber_delta, grad_out);
    }

    return -1;
}

int lnn_max_layer_width(const Layer *layers, int num_layers) {
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
