#ifndef MODELS_INTERNAL_H
#define MODELS_INTERNAL_H

#include "models.h"

int lnn_has_lnn_extension(const char *file_path);
void lnn_free_layer_snapshots(float **weights, float **biases, int num_layers);
int lnn_plugin_layer_valid(const LayerPlugin *layer);
int lnn_max_plugin_layer_width(const SequentialModel *model);
void lnn_model_workspace_free(SequentialModel *model);
int lnn_ensure_workspace(float **buffer, int *current_size, int required_size);
void lnn_free_grad_accumulators(float **acc_w, float **acc_b, int num_layers);

int lnn_optimizer_state_valid(OptimizerType optimizer,
                              const OptimizerState *optimizer_state);
int lnn_apply_optimizer_update(float *weights,
                               const float *grads,
                               int size,
                               OptimizerType optimizer,
                               float learning_rate,
                               OptimizerState *optimizer_state,
                               float *opt_state_a,
                               float *opt_state_b);
int lnn_compute_loss_and_grad(LossFunctionType loss_function,
                              const float *prediction,
                              const float *target,
                              int size,
                              float *loss_out,
                              float *grad_out);

int lnn_max_layer_width(const Layer *layers, int num_layers);

#endif /* MODELS_INTERNAL_H */
