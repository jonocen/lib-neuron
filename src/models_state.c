#include "../include/models_training.h"

#include "models_internal.h"

#include <stdlib.h>

void sequential_train_config_init_sgd(SequentialTrainConfig *cfg,
                                      LossFunctionType loss_function,
                                      float learning_rate) {
    if (!cfg) return;
    cfg->loss_function = loss_function;
    cfg->optimizer = OPTIMIZER_SGD;
    cfg->learning_rate = learning_rate;
    cfg->optimizer_state = NULL;
    cfg->adam_state = NULL;
}

void sequential_train_config_init_optimizer(SequentialTrainConfig *cfg,
                                            LossFunctionType loss_function,
                                            OptimizerType optimizer,
                                            float learning_rate,
                                            OptimizerState *optimizer_state) {
    if (!cfg) return;
    cfg->loss_function = loss_function;
    cfg->optimizer = optimizer;
    cfg->learning_rate = learning_rate;
    cfg->optimizer_state = optimizer_state;
    cfg->adam_state = optimizer_state;
}

void sequential_train_config_init_rmsprop(SequentialTrainConfig *cfg,
                                          LossFunctionType loss_function,
                                          float learning_rate,
                                          OptimizerState *optimizer_state) {
    sequential_train_config_init_optimizer(cfg,
                                           loss_function,
                                           OPTIMIZER_RMSPROP,
                                           learning_rate,
                                           optimizer_state);
}

void sequential_train_config_init_adam(SequentialTrainConfig *cfg,
                                       LossFunctionType loss_function,
                                       float learning_rate,
                                       AdamOptimizerState *adam_state) {
    sequential_train_config_init_optimizer(cfg,
                                           loss_function,
                                           OPTIMIZER_ADAM,
                                           learning_rate,
                                           adam_state);
}

void sequential_model_optimizer_state_free(SequentialModel *model,
                                           OptimizerState *state) {
    if (!model || !state) return;

    if (state->m_w) {
        for (int i = 0; i < model->num_layers; i++) {
            free(state->m_w[i]);
        }
        free(state->m_w);
    }

    if (state->v_w) {
        for (int i = 0; i < model->num_layers; i++) {
            free(state->v_w[i]);
        }
        free(state->v_w);
    }

    if (state->m_b) {
        for (int i = 0; i < model->num_layers; i++) {
            free(state->m_b[i]);
        }
        free(state->m_b);
    }

    if (state->v_b) {
        for (int i = 0; i < model->num_layers; i++) {
            free(state->v_b[i]);
        }
        free(state->v_b);
    }

    state->m_w = NULL;
    state->v_w = NULL;
    state->m_b = NULL;
    state->v_b = NULL;
    state->step = 0;
    state->beta1 = 0.0f;
    state->beta2 = 0.0f;
}

int sequential_model_optimizer_state_init(SequentialModel *model,
                                          OptimizerState *out_state,
                                          OptimizerType optimizer,
                                          float beta1,
                                          float beta2) {
    if (!model || model->num_layers <= 0 || !out_state) return -1;
    if (optimizer != OPTIMIZER_SGD && optimizer != OPTIMIZER_ADAM && optimizer != OPTIMIZER_RMSPROP && optimizer != OPTIMIZER_ADAGRAD && optimizer != OPTIMIZER_ADAMW) return -1;
    if (optimizer == OPTIMIZER_SGD) return 0;
    if (beta1 <= 0.0f || beta1 >= 1.0f) return -1;
    if ((optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_ADAMW) && (beta2 <= 0.0f || beta2 >= 1.0f)) return -1;

    if (out_state->m_w || out_state->v_w || out_state->m_b || out_state->v_b) return -1;

    out_state->m_w = calloc((size_t)model->num_layers, sizeof(float *));
    out_state->v_w = calloc((size_t)model->num_layers, sizeof(float *));
    out_state->m_b = calloc((size_t)model->num_layers, sizeof(float *));
    out_state->v_b = calloc((size_t)model->num_layers, sizeof(float *));
    if (!out_state->m_w || !out_state->v_w || !out_state->m_b || !out_state->v_b) {
        sequential_model_optimizer_state_free(model, out_state);
        return -1;
    }

    for (int i = 0; i < model->num_layers; i++) {
        int w_size = model->layers[i].weights_size(model->layers[i].ctx);
        int b_size = model->layers[i].biases_size(model->layers[i].ctx);
        out_state->m_w[i] = calloc((size_t)w_size, sizeof(float));
        out_state->v_w[i] = (optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_ADAMW) ? calloc((size_t)w_size, sizeof(float)) : NULL;
        out_state->m_b[i] = calloc((size_t)b_size, sizeof(float));
        out_state->v_b[i] = (optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_ADAMW) ? calloc((size_t)b_size, sizeof(float)) : NULL;
        if (!out_state->m_w[i] || !out_state->m_b[i] ||
            ((optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_ADAMW) && (!out_state->v_w[i] || !out_state->v_b[i]))) {
            sequential_model_optimizer_state_free(model, out_state);
            return -1;
        }
    }

    out_state->step = 1;
    out_state->beta1 = beta1;
    out_state->beta2 = (optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_ADAMW) ? beta2 : 0.0f;
    return 0;
}

void sequential_model_adam_state_free(SequentialModel *model,
                                      AdamOptimizerState *state) {
    sequential_model_optimizer_state_free(model, state);
}

int sequential_model_adam_state_init(SequentialModel *model,
                                     AdamOptimizerState *out_state,
                                     float beta1,
                                     float beta2) {
    return sequential_model_optimizer_state_init(model,
                                                 out_state,
                                                 OPTIMIZER_ADAM,
                                                 beta1,
                                                 beta2);
}

int sequential_model_train_step_cfg(SequentialModel *model,
                                    const float *input,
                                    const float *target,
                                    float *output,
                                    const SequentialTrainConfig *cfg,
                                    float *loss_out) {
    if (!cfg) return -1;
    return sequential_model_train_step(model,
                                       input,
                                       target,
                                       output,
                                       cfg->loss_function,
                                       cfg->optimizer,
                                       cfg->learning_rate,
                                       cfg->optimizer_state ? cfg->optimizer_state : cfg->adam_state,
                                       loss_out);
}

int sequential_model_train_step(SequentialModel *model,
                                const float *input,
                                const float *target,
                                float *output,
                                LossFunctionType loss_function,
                                OptimizerType optimizer,
                                float learning_rate,
                                OptimizerState *optimizer_state,
                                float *loss_out) {
    if (!model || model->num_layers <= 0 || !input || !target || !output) {
        return -1;
    }

    if (sequential_model_forward(model, input, output) != 0) {
        return -1;
    }

    return sequential_model_optimize_from_prediction(model,
                                                     output,
                                                     target,
                                                     loss_function,
                                                     optimizer,
                                                     learning_rate,
                                                     optimizer_state,
                                                     loss_out);
}

int sequential_model_optimize_from_prediction(SequentialModel *model,
                                              const float *prediction,
                                              const float *target,
                                              LossFunctionType loss_function,
                                              OptimizerType optimizer,
                                              float learning_rate,
                                              OptimizerState *optimizer_state,
                                              float *loss_out) {
    if (!model || model->num_layers <= 0 || !prediction || !target) {
        return -1;
    }

    int output_size = model->layers[model->num_layers - 1].output_size(
        model->layers[model->num_layers - 1].ctx);

    if (!lnn_optimizer_state_valid(optimizer, optimizer_state)) {
        return -1;
    }

    if (!model->work_delta_a || !model->work_delta_b ||
        !model->work_grad_w  || !model->work_grad_b) {
        int width = 0, max_grad_w_size = 0, max_grad_b_size = 0;
        for (int i = 0; i < model->num_layers; i++) {
            int in  = model->layers[i].input_size(model->layers[i].ctx);
            int out = model->layers[i].output_size(model->layers[i].ctx);
            int ws  = model->layers[i].weights_size(model->layers[i].ctx);
            int bs  = model->layers[i].biases_size(model->layers[i].ctx);
            if (in  > width)            width            = in;
            if (out > width)            width            = out;
            if (ws  > max_grad_w_size)  max_grad_w_size  = ws;
            if (bs  > max_grad_b_size)  max_grad_b_size  = bs;
        }
        if (max_grad_w_size <= 0 || max_grad_b_size <= 0) return -1;
        if (lnn_ensure_workspace(&model->work_delta_a, &model->work_delta_size, width) != 0 ||
            lnn_ensure_workspace(&model->work_delta_b, &model->work_delta_size, width) != 0 ||
            lnn_ensure_workspace(&model->work_grad_w,  &model->work_grad_w_size, max_grad_w_size) != 0 ||
            lnn_ensure_workspace(&model->work_grad_b,  &model->work_grad_b_size, max_grad_b_size) != 0) {
            return -1;
        }
    }

    float *delta_curr = model->work_delta_a;
    float *delta_prev = model->work_delta_b;
    float *grad_w = model->work_grad_w;
    float *grad_b = model->work_grad_b;

    if (lnn_compute_loss_and_grad(loss_function, prediction, target, output_size, loss_out, delta_curr) != 0) {
        return -1;
    }

    for (int i = model->num_layers - 1; i >= 0; i--) {
        float *next_delta = (i > 0) ? delta_prev : NULL;

        int grad_w_size = model->layers[i].weights_size(model->layers[i].ctx);
        int grad_b_size = model->layers[i].biases_size(model->layers[i].ctx);

        if (model->layers[i].backward(model->layers[i].ctx, delta_curr, next_delta, grad_w, grad_b) != 0) {
            return -1;
        }

        float *opt_state_w_a = NULL;
        float *opt_state_w_b = NULL;
        float *opt_state_b_a = NULL;
        float *opt_state_b_b = NULL;

        if (optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_ADAMW) {
            if (!optimizer_state->m_w[i] || !optimizer_state->v_w[i] ||
                !optimizer_state->m_b[i] || !optimizer_state->v_b[i]) {
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_w_b = optimizer_state->v_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
            opt_state_b_b = optimizer_state->v_b[i];
        } else if (optimizer == OPTIMIZER_RMSPROP) {
            if (!optimizer_state->m_w[i] || !optimizer_state->m_b[i]) {
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
        } else if (optimizer == OPTIMIZER_ADAGRAD) {
            if (!optimizer_state->m_w[i] || !optimizer_state->m_b[i]) {
                return -1;
            }

            opt_state_w_a = optimizer_state->m_w[i];
            opt_state_b_a = optimizer_state->m_b[i];
        }

        if (lnn_apply_optimizer_update(model->layers[i].weights(model->layers[i].ctx),
                                       grad_w,
                                       grad_w_size,
                                       optimizer,
                                       learning_rate,
                                       optimizer_state,
                                       opt_state_w_a,
                                       opt_state_w_b) != 0 ||
            lnn_apply_optimizer_update(model->layers[i].biases(model->layers[i].ctx),
                                       grad_b,
                                       grad_b_size,
                                       optimizer,
                                       learning_rate,
                                       optimizer_state,
                                       opt_state_b_a,
                                       opt_state_b_b) != 0) {
            return -1;
        }

        if (i > 0) {
            float *tmp = delta_curr;
            delta_curr = delta_prev;
            delta_prev = tmp;
        }
    }

    if (optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_ADAMW) {
        optimizer_state->step += 1;
    }

    return 0;
}
