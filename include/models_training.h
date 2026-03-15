#ifndef MODELS_TRAINING_H
#define MODELS_TRAINING_H

#include "models_types.h"

int sequential_model_compile(SequentialModel *model,
                             LossFunctionType loss_function,
                             OptimizerType optimizer,
                             float learning_rate,
                             float optimizer_beta1,
                             float optimizer_beta2);

int sequential_model_compile_optimizer(SequentialModel *model,
                                       LossFunctionType loss_function,
                                       OptimizerType optimizer,
                                       float learning_rate,
                                       float optimizer_beta1,
                                       float optimizer_beta2);

int sequential_model_train(SequentialModel *model,
                           const float *inputs,
                           const float *targets,
                           int num_samples,
                           int input_size,
                           int target_size,
                           int epochs,
                           int batch_size,
                           float *final_loss_out);

int sequential_model_train_with_progress(SequentialModel *model,
                                         const float *inputs,
                                         const float *targets,
                                         int num_samples,
                                         int input_size,
                                         int target_size,
                                         int epochs,
                                         int batch_size,
                                         int progress_percent,
                                         float *final_loss_out);

void sequential_train_config_init_sgd(SequentialTrainConfig *cfg,
                                      LossFunctionType loss_function,
                                      float learning_rate);
void sequential_train_config_init_optimizer(SequentialTrainConfig *cfg,
                                            LossFunctionType loss_function,
                                            OptimizerType optimizer,
                                            float learning_rate,
                                            OptimizerState *optimizer_state);
void sequential_train_config_init_rmsprop(SequentialTrainConfig *cfg,
                                          LossFunctionType loss_function,
                                          float learning_rate,
                                          OptimizerState *optimizer_state);
void sequential_train_config_init_adam(SequentialTrainConfig *cfg,
                                       LossFunctionType loss_function,
                                       float learning_rate,
                                       AdamOptimizerState *adam_state);

int sequential_model_optimizer_state_init(SequentialModel *model,
                                          OptimizerState *out_state,
                                          OptimizerType optimizer,
                                          float beta1,
                                          float beta2);
void sequential_model_optimizer_state_free(SequentialModel *model,
                                           OptimizerState *state);

int sequential_model_adam_state_init(SequentialModel *model,
                                     AdamOptimizerState *out_state,
                                     float beta1,
                                     float beta2);
void sequential_model_adam_state_free(SequentialModel *model,
                                      AdamOptimizerState *state);

int sequential_model_train_step_cfg(SequentialModel *model,
                                    const float *input,
                                    const float *target,
                                    float *output,
                                    const SequentialTrainConfig *cfg,
                                    float *loss_out);

int sequential_model_train_step(SequentialModel *model,
                                const float *input,
                                const float *target,
                                float *output,
                                LossFunctionType loss_function,
                                OptimizerType optimizer,
                                float learning_rate,
                                OptimizerState *optimizer_state,
                                float *loss_out);

int sequential_model_optimize_from_prediction(SequentialModel *model,
                                              const float *prediction,
                                              const float *target,
                                              LossFunctionType loss_function,
                                              OptimizerType optimizer,
                                              float learning_rate,
                                              OptimizerState *optimizer_state,
                                              float *loss_out);

#endif /* MODELS_TRAINING_H */
