#ifndef MODELS_TYPES_H
#define MODELS_TYPES_H

#include "layers.h"
#include "lossfunctions.h"
#include "matrixcalculation.h"
#include "optimizers.h"

typedef struct {
    float **m_w;
    float **v_w;
    float **m_b;
    float **v_b;
    int     step;
    float   beta1;
    float   beta2;
} OptimizerState;

/* Backward-compatible alias. */
typedef OptimizerState AdamOptimizerState;

typedef struct {
    LayerPlugin *layers;
    int          num_layers;
    int          capacity;
    int          compiled;
    LossFunctionType compiled_loss;
    OptimizerType    compiled_optimizer;
    float            compiled_learning_rate;
    int              compiled_owns_optimizer_state;
    OptimizerState   compiled_optimizer_state;
    /* Internal reusable workspaces for faster forward/backward passes. */
    float           *work_forward_a;
    float           *work_forward_b;
    int              work_forward_size;
    float           *work_delta_a;
    float           *work_delta_b;
    int              work_delta_size;
    float           *work_grad_w;
    float           *work_grad_b;
    int              work_grad_w_size;
    int              work_grad_b_size;
} SequentialModel;

typedef struct {
    LossFunctionType   loss_function;
    OptimizerType      optimizer;
    float              learning_rate;
    OptimizerState     *optimizer_state;
    /* Backward-compatible alias. */
    AdamOptimizerState *adam_state;
} SequentialTrainConfig;

#endif /* MODELS_TYPES_H */
