#include "../include/models_training.h"

#include "models_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int sequential_model_compile(SequentialModel *model,
                             LossFunctionType loss_function,
                             OptimizerType optimizer,
                             float learning_rate,
                             float optimizer_beta1,
                             float optimizer_beta2) {
    return sequential_model_compile_optimizer(model,
                                              loss_function,
                                              optimizer,
                                              learning_rate,
                                              optimizer_beta1,
                                              optimizer_beta2);
}

int sequential_model_compile_optimizer(SequentialModel *model,
                                       LossFunctionType loss_function,
                                       OptimizerType optimizer,
                                       float learning_rate,
                                       float optimizer_beta1,
                                       float optimizer_beta2) {
    if (!model || model->num_layers <= 0 || learning_rate <= 0.0f) return -1;
    if (loss_function != LOSS_MSE && loss_function != LOSS_BCE && loss_function != LOSS_HUBER) return -1;
    if (optimizer != OPTIMIZER_SGD && optimizer != OPTIMIZER_ADAM && optimizer != OPTIMIZER_RMSPROP && optimizer != OPTIMIZER_ADAGRAD && optimizer != OPTIMIZER_ADAMW) return -1;
    if (optimizer == OPTIMIZER_RMSPROP && learning_rate <= 0.0f) return -1;

    if (model->compiled_owns_optimizer_state) {
        sequential_model_optimizer_state_free(model, &model->compiled_optimizer_state);
        model->compiled_owns_optimizer_state = 0;
    }

    model->compiled = 0;
    model->compiled_loss = loss_function;
    model->compiled_optimizer = optimizer;
    model->compiled_learning_rate = learning_rate;

    if (optimizer == OPTIMIZER_ADAM || optimizer == OPTIMIZER_RMSPROP || optimizer == OPTIMIZER_ADAGRAD || optimizer == OPTIMIZER_ADAMW) {
        if (sequential_model_optimizer_state_init(model,
                                                  &model->compiled_optimizer_state,
                                                  optimizer,
                                                  optimizer_beta1,
                                                  optimizer_beta2) != 0) {
            return -1;
        }
        model->compiled_owns_optimizer_state = 1;
    }

    {
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
        if (width > 0 && max_grad_w_size > 0 && max_grad_b_size > 0) {
            if (lnn_ensure_workspace(&model->work_forward_a, &model->work_forward_size, width) != 0 ||
                lnn_ensure_workspace(&model->work_forward_b, &model->work_forward_size, width) != 0 ||
                lnn_ensure_workspace(&model->work_delta_a,   &model->work_delta_size,   width) != 0 ||
                lnn_ensure_workspace(&model->work_delta_b,   &model->work_delta_size,   width) != 0 ||
                lnn_ensure_workspace(&model->work_grad_w,    &model->work_grad_w_size,  max_grad_w_size) != 0 ||
                lnn_ensure_workspace(&model->work_grad_b,    &model->work_grad_b_size,  max_grad_b_size) != 0) {
                if (model->compiled_owns_optimizer_state) {
                    sequential_model_optimizer_state_free(model, &model->compiled_optimizer_state);
                    model->compiled_owns_optimizer_state = 0;
                }
                return -1;
            }
        }
    }

    model->compiled = 1;
    return 0;
}

int sequential_model_train_with_progress(SequentialModel *model,
                                         const float *inputs,
                                         const float *targets,
                                         int num_samples,
                                         int input_size,
                                         int target_size,
                                         int epochs,
                                         int batch_size,
                                         int progress_percent,
                                         float *final_loss_out) {
    int status = -1;
    float epoch_loss = 0.0f;
    float *output = NULL;
    float **acc_w = NULL;
    float **acc_b = NULL;
    float **layer_weights = NULL;
    float **layer_biases = NULL;
    int *grad_w_sizes = NULL;
    int *grad_b_sizes = NULL;

    if (!model || !inputs || !targets || num_samples <= 0 || input_size <= 0 || target_size <= 0 || epochs <= 0) {
        return -1;
    }
    if (batch_size <= 0) return -1;
    if (progress_percent > 100) progress_percent = 100;
    if (!model->compiled) return -1;
    if (model->num_layers <= 0) return -1;

    int expected_input_size = model->layers[0].input_size(model->layers[0].ctx);
    int expected_target_size = model->layers[model->num_layers - 1].output_size(model->layers[model->num_layers - 1].ctx);
    if (input_size != expected_input_size || target_size != expected_target_size) return -1;

    if (batch_size > num_samples) {
        batch_size = num_samples;
    }

    OptimizerState *optimizer_state =
        (model->compiled_optimizer == OPTIMIZER_ADAM ||
         model->compiled_optimizer == OPTIMIZER_RMSPROP ||
         model->compiled_optimizer == OPTIMIZER_ADAGRAD ||
         model->compiled_optimizer == OPTIMIZER_ADAMW)
            ? &model->compiled_optimizer_state
            : NULL;

    if (!lnn_optimizer_state_valid(model->compiled_optimizer, optimizer_state)) {
        return -1;
    }

    output = malloc((size_t)expected_target_size * sizeof(float));
    if (!output) goto cleanup;

    layer_weights = calloc((size_t)model->num_layers, sizeof(float *));
    layer_biases = calloc((size_t)model->num_layers, sizeof(float *));
    grad_w_sizes = calloc((size_t)model->num_layers, sizeof(int));
    grad_b_sizes = calloc((size_t)model->num_layers, sizeof(int));
    if (!layer_weights || !layer_biases || !grad_w_sizes || !grad_b_sizes) {
        goto cleanup;
    }

    int max_grad_w_size = 0;
    int max_grad_b_size = 0;
    for (int i = 0; i < model->num_layers; i++) {
        layer_weights[i] = model->layers[i].weights(model->layers[i].ctx);
        layer_biases[i] = model->layers[i].biases(model->layers[i].ctx);
        grad_w_sizes[i] = model->layers[i].weights_size(model->layers[i].ctx);
        grad_b_sizes[i] = model->layers[i].biases_size(model->layers[i].ctx);

        if (!layer_weights[i] || !layer_biases[i] || grad_w_sizes[i] <= 0 || grad_b_sizes[i] <= 0) {
            goto cleanup;
        }

        if (grad_w_sizes[i] > max_grad_w_size) {
            max_grad_w_size = grad_w_sizes[i];
        }
        if (grad_b_sizes[i] > max_grad_b_size) {
            max_grad_b_size = grad_b_sizes[i];
        }
    }

    int width = lnn_max_plugin_layer_width(model);
    if (width <= 0 || max_grad_w_size <= 0 || max_grad_b_size <= 0 ||
        lnn_ensure_workspace(&model->work_delta_a, &model->work_delta_size, width) != 0 ||
        lnn_ensure_workspace(&model->work_delta_b, &model->work_delta_size, width) != 0 ||
        lnn_ensure_workspace(&model->work_grad_w, &model->work_grad_w_size, max_grad_w_size) != 0 ||
        lnn_ensure_workspace(&model->work_grad_b, &model->work_grad_b_size, max_grad_b_size) != 0) {
        goto cleanup;
    }

    int next_progress = progress_percent;

    if (batch_size == 1) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            epoch_loss = 0.0f;
            for (int sample_idx = 0; sample_idx < num_samples; sample_idx++) {
                const float *sample_input = inputs + ((size_t)sample_idx * (size_t)input_size);
                const float *sample_target = targets + ((size_t)sample_idx * (size_t)target_size);
                float sample_loss = 0.0f;
                float *delta_curr = model->work_delta_a;
                float *delta_prev = model->work_delta_b;

                if (sequential_model_forward(model, sample_input, output) != 0 ||
                    lnn_compute_loss_and_grad(model->compiled_loss,
                                              output,
                                              sample_target,
                                              target_size,
                                              &sample_loss,
                                              delta_curr) != 0) {
                    goto cleanup;
                }

                epoch_loss += sample_loss;

                for (int l = model->num_layers - 1; l >= 0; l--) {
                    float *next_delta = (l > 0) ? delta_prev : NULL;
                    float *opt_state_w_a = NULL;
                    float *opt_state_w_b = NULL;
                    float *opt_state_b_a = NULL;
                    float *opt_state_b_b = NULL;

                    if (model->layers[l].backward(model->layers[l].ctx,
                                                  delta_curr,
                                                  next_delta,
                                                  model->work_grad_w,
                                                  model->work_grad_b) != 0) {
                        goto cleanup;
                    }

                    if (model->compiled_optimizer == OPTIMIZER_ADAM ||
                        model->compiled_optimizer == OPTIMIZER_ADAMW) {
                        opt_state_w_a = optimizer_state->m_w[l];
                        opt_state_w_b = optimizer_state->v_w[l];
                        opt_state_b_a = optimizer_state->m_b[l];
                        opt_state_b_b = optimizer_state->v_b[l];
                    } else if (model->compiled_optimizer == OPTIMIZER_RMSPROP) {
                        opt_state_w_a = optimizer_state->m_w[l];
                        opt_state_b_a = optimizer_state->m_b[l];
                    } else if (model->compiled_optimizer == OPTIMIZER_ADAGRAD) {
                        opt_state_w_a = optimizer_state->m_w[l];
                        opt_state_b_a = optimizer_state->m_b[l];
                    }

                    if (lnn_apply_optimizer_update(layer_weights[l],
                                                   model->work_grad_w,
                                                   grad_w_sizes[l],
                                                   model->compiled_optimizer,
                                                   model->compiled_learning_rate,
                                                   optimizer_state,
                                                   opt_state_w_a,
                                                   opt_state_w_b) != 0 ||
                        lnn_apply_optimizer_update(layer_biases[l],
                                                   model->work_grad_b,
                                                   grad_b_sizes[l],
                                                   model->compiled_optimizer,
                                                   model->compiled_learning_rate,
                                                   optimizer_state,
                                                   opt_state_b_a,
                                                   opt_state_b_b) != 0) {
                        goto cleanup;
                    }

                    if (l > 0) {
                        float *tmp = delta_curr;
                        delta_curr = delta_prev;
                        delta_prev = tmp;
                    }
                }

                if (model->compiled_optimizer == OPTIMIZER_ADAM || model->compiled_optimizer == OPTIMIZER_ADAMW) {
                    optimizer_state->step += 1;
                }
            }

            if (progress_percent > 0) {
                float avg_loss = epoch_loss / (float)num_samples;
                int current_percent = ((epoch + 1) * 100) / epochs;
                int is_final_epoch = (epoch + 1) == epochs;

                if (current_percent >= next_progress || is_final_epoch) {
                    printf("[train] %d%% epoch=%d/%d loss=%.6f\n",
                           current_percent,
                           epoch + 1,
                           epochs,
                           avg_loss);

                    while (next_progress <= current_percent && next_progress > 0) {
                        next_progress += progress_percent;
                    }
                }
            }
        }

        if (final_loss_out) {
            *final_loss_out = epoch_loss / (float)num_samples;
        }

        status = 0;
        goto cleanup;
    }

    acc_w = calloc((size_t)model->num_layers, sizeof(float *));
    acc_b = calloc((size_t)model->num_layers, sizeof(float *));
    if (!acc_w || !acc_b) {
        goto cleanup;
    }

    for (int i = 0; i < model->num_layers; i++) {
        acc_w[i] = calloc((size_t)grad_w_sizes[i], sizeof(float));
        acc_b[i] = calloc((size_t)grad_b_sizes[i], sizeof(float));
        if (!acc_w[i] || !acc_b[i]) {
            goto cleanup;
        }
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        epoch_loss = 0.0f;
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int batch_end = batch_start + batch_size;
            if (batch_end > num_samples) batch_end = num_samples;
            int current_batch = batch_end - batch_start;

            for (int l = 0; l < model->num_layers; l++) {
                memset(acc_w[l], 0, (size_t)grad_w_sizes[l] * sizeof(float));
                memset(acc_b[l], 0, (size_t)grad_b_sizes[l] * sizeof(float));
            }

            for (int sample_idx = batch_start; sample_idx < batch_end; sample_idx++) {
                const float *sample_input = inputs + ((size_t)sample_idx * (size_t)input_size);
                const float *sample_target = targets + ((size_t)sample_idx * (size_t)target_size);
                float sample_loss = 0.0f;
                float *delta_curr = model->work_delta_a;
                float *delta_prev = model->work_delta_b;

                if (sequential_model_forward(model, sample_input, output) != 0 ||
                    lnn_compute_loss_and_grad(model->compiled_loss,
                                              output,
                                              sample_target,
                                              target_size,
                                              &sample_loss,
                                              delta_curr) != 0) {
                    goto cleanup;
                }

                epoch_loss += sample_loss;

                for (int l = model->num_layers - 1; l >= 0; l--) {
                    float *next_delta = (l > 0) ? delta_prev : NULL;

                    if (grad_w_sizes[l] > model->work_grad_w_size ||
                        grad_b_sizes[l] > model->work_grad_b_size ||
                        model->layers[l].backward(model->layers[l].ctx,
                                                  delta_curr,
                                                  next_delta,
                                                  model->work_grad_w,
                                                  model->work_grad_b) != 0) {
                        goto cleanup;
                    }

                    for (int k = 0; k < grad_w_sizes[l]; k++) {
                        acc_w[l][k] += model->work_grad_w[k];
                    }
                    for (int k = 0; k < grad_b_sizes[l]; k++) {
                        acc_b[l][k] += model->work_grad_b[k];
                    }

                    if (l > 0) {
                        float *tmp = delta_curr;
                        delta_curr = delta_prev;
                        delta_prev = tmp;
                    }
                }
            }

            float inv_batch = 1.0f / (float)current_batch;
            for (int l = 0; l < model->num_layers; l++) {
                float *opt_state_w_a = NULL;
                float *opt_state_w_b = NULL;
                float *opt_state_b_a = NULL;
                float *opt_state_b_b = NULL;

                for (int k = 0; k < grad_w_sizes[l]; k++) {
                    acc_w[l][k] *= inv_batch;
                }
                for (int k = 0; k < grad_b_sizes[l]; k++) {
                    acc_b[l][k] *= inv_batch;
                }

                if (model->compiled_optimizer == OPTIMIZER_ADAM ||
                    model->compiled_optimizer == OPTIMIZER_ADAMW) {
                    opt_state_w_a = optimizer_state->m_w[l];
                    opt_state_w_b = optimizer_state->v_w[l];
                    opt_state_b_a = optimizer_state->m_b[l];
                    opt_state_b_b = optimizer_state->v_b[l];
                } else if (model->compiled_optimizer == OPTIMIZER_RMSPROP) {
                    opt_state_w_a = optimizer_state->m_w[l];
                    opt_state_b_a = optimizer_state->m_b[l];
                } else if (model->compiled_optimizer == OPTIMIZER_ADAGRAD) {
                    opt_state_w_a = optimizer_state->m_w[l];
                    opt_state_b_a = optimizer_state->m_b[l];
                }

                if (lnn_apply_optimizer_update(layer_weights[l],
                                               acc_w[l],
                                               grad_w_sizes[l],
                                               model->compiled_optimizer,
                                               model->compiled_learning_rate,
                                               optimizer_state,
                                               opt_state_w_a,
                                               opt_state_w_b) != 0 ||
                    lnn_apply_optimizer_update(layer_biases[l],
                                               acc_b[l],
                                               grad_b_sizes[l],
                                               model->compiled_optimizer,
                                               model->compiled_learning_rate,
                                               optimizer_state,
                                               opt_state_b_a,
                                               opt_state_b_b) != 0) {
                    goto cleanup;
                }
            }

            if (model->compiled_optimizer == OPTIMIZER_ADAM || model->compiled_optimizer == OPTIMIZER_ADAMW) {
                optimizer_state->step += 1;
            }
        }

        if (progress_percent > 0) {
            float avg_loss = epoch_loss / (float)num_samples;
            int current_percent = ((epoch + 1) * 100) / epochs;
            int is_final_epoch = (epoch + 1) == epochs;

            if (current_percent >= next_progress || is_final_epoch) {
                printf("[train] %d%% epoch=%d/%d loss=%.6f\n",
                       current_percent,
                       epoch + 1,
                       epochs,
                       avg_loss);

                while (next_progress <= current_percent && next_progress > 0) {
                    next_progress += progress_percent;
                }
            }
        }
    }

    if (final_loss_out) {
        *final_loss_out = epoch_loss / (float)num_samples;
    }

    status = 0;

cleanup:
    lnn_free_grad_accumulators(acc_w, acc_b, model->num_layers);
    free(grad_b_sizes);
    free(grad_w_sizes);
    free(layer_biases);
    free(layer_weights);
    free(output);
    return status;
}

int sequential_model_train(SequentialModel *model,
                           const float *inputs,
                           const float *targets,
                           int num_samples,
                           int input_size,
                           int target_size,
                           int epochs,
                           int batch_size,
                           float *final_loss_out) {
    return sequential_model_train_with_progress(model,
                                                inputs,
                                                targets,
                                                num_samples,
                                                input_size,
                                                target_size,
                                                epochs,
                                                batch_size,
                                                0,
                                                final_loss_out);
}
