# API Reference

This document describes the public API exposed by `lib-neuron` headers.

It is written as a practical reference:

- exact signatures grouped by header
- tensor/shape rules
- ownership and lifetime expectations
- common failure conditions (`-1` cases)
- compact usage patterns

## Return Convention

For most non-loss APIs:

- `0` = success
- `-1` = invalid input and/or internal failure

For loss value APIs:

- returns `float`

## Data and Layout Conventions

Dense layout:

- vectors are flat arrays of length `N`

Conv2D and MaxPool2D layout:

- flattened CHW (`channel`, `height`, `width`)
- flattened index order is channel-major

Conv2D output shape:

- `out_w = ((in_w + 2 * padding - kernel_w) / stride) + 1`
- `out_h = ((in_h + 2 * padding - kernel_h) / stride) + 1`

MaxPool2D output shape:

- `out_w = ((in_w + 2 * padding - pool_w) / stride) + 1`
- `out_h = ((in_h + 2 * padding - pool_h) / stride) + 1`

Important:

- shape divisions above must be exact
- invalid geometry returns `-1`

## Ownership and Lifetime

- `layer_init`, `conv2d_layer_init`, `maxpool2d_layer_init` allocate internal buffers.
- matching `*_free` functions release those buffers.
- plugin constructors (`layer_plugin_*_create`) allocate plugin context and transfer ownership to `LayerPlugin`.
- always release plugin layers with `layer_plugin_free`.
- `SequentialModel` owns layers added via `sequential_model_add_layer` and convenience add helpers.

## Naming Patterns in Training Helpers

- `*_with_loss`: runtime-selectable loss (`LOSS_MSE` or `LOSS_BCE`)
- `*_mse`, `*_bce`: fixed-loss wrapper
- no suffix (`sequential_model_train_step`, `sequential_train_step`): compatibility helper using MSE

## Core Types

### `Activation` (`include/matrixcalculation.h`)

- `ACT_LINEAR`
- `ACT_RELU`
- `ACT_SIGMOID`
- `ACT_TANH`

### `Layer` (Dense)

Fields:

- `input_size`
- `output_size`
- `activation`
- `weights` (`[output_size * input_size]`)
- `biases` (`[output_size]`)
- `cache_input` (`[input_size]`)
- `cache_z` (`[output_size]`)

### `Conv2DLayer`

Fields include geometry, parameters, and caches:

- input geometry: `input_width`, `input_height`, `input_channels`
- output channels: `output_channels`
- kernel: `kernel_width`, `kernel_height`
- geometry: `stride`, `padding`
- derived output geometry: `output_width`, `output_height`
- activation: `activation`
- parameters:
  - `weights` (`[out_c * in_c * kernel_h * kernel_w]`)
  - `biases` (`[out_c]`)
- caches:
  - `cache_input`
  - `cache_z`

### `MaxPool2DLayer`

Fields include geometry and argmax cache:

- input geometry: `input_width`, `input_height`, `channels`
- pooling window: `pool_width`, `pool_height`
- geometry: `stride`, `padding`
- derived output geometry: `output_width`, `output_height`
- caches:
  - `cache_input`
  - `cache_max_indices`

Compatibility note:

- maxpool has no trainable weights conceptually
- internal dummy parameter slots (`dummy_weight`, `dummy_bias`) exist for generic plugin training-pipeline compatibility

### `LayerPlugin` (`include/layers.h`)

Function-pointer interface used by `SequentialModel`:

- `forward`
- `backward`
- `input_size`, `output_size`
- `weights`, `biases`
- `weights_size`, `biases_size`
- `destroy`

### `OptimizerType` (`include/optimizers.h`)

- `OPTIMIZER_SGD`
- `OPTIMIZER_ADAM`
- `OPTIMIZER_RMSPROP`

### `LossFunctionType` (`include/lossfunctions.h`)

- `LOSS_MSE`
- `LOSS_BCE`

### `AdamOptimizerState` (`include/models.h`)

Used by sequential/layer-array optimizer helpers.

Adam mode:

- `m_w`, `v_w`, `m_b`, `v_b`
- `step`
- `beta1`, `beta2`

RMSProp mode (reusing same struct):

- uses `m_w`, `m_b` as caches
- uses `beta1` as RMSProp decay
- ignores `v_w`, `v_b`, `step` for RMSProp updates

### `SequentialTrainConfig` (`include/models.h`)

- `loss_function`
- `optimizer`
- `learning_rate`
- `adam_state`

## Header: `matrixcalculation.h`

### Activation API

`float act_apply(float x, Activation a)`

- applies activation to scalar `x`

`float act_deriv(float x, Activation a)`

- returns derivative at scalar `x`

### Dense Layer API

`int layer_init(Layer *layer, int input_size, int output_size, Activation activation)`

- allocates internal dense buffers
- expected sizes:
  - weights: `output_size * input_size`
  - biases: `output_size`
- returns `-1` for invalid sizes or allocation failure

`void layer_free(Layer *layer)`

- releases all dense buffers
- safe on partially initialized layers

`int layer_forward(Layer *layer, const float *input, float *output)`

- input shape: `[input_size]`
- output shape: `[output_size]`
- caches current input and pre-activation `z`

`int layer_backward(const Layer *layer, const float *delta_in, float *delta_out, float *grad_w, float *grad_b)`

- `delta_in`: `[output_size]`
- `delta_out`: `[input_size]` or `NULL`
- `grad_w`: `[output_size * input_size]`
- `grad_b`: `[output_size]`
- applies activation derivative using cached `cache_z`

### Conv2D API

`int conv2d_layer_init(Conv2DLayer *layer, int input_width, int input_height, int input_channels, int output_channels, int kernel_width, int kernel_height, int stride, int padding, Activation activation)`

- validates geometry and divisibility
- computes `output_width` and `output_height`
- allocates parameters and caches
- returns `-1` for invalid geometry or allocation failure

`void conv2d_layer_free(Conv2DLayer *layer)`

- frees conv2d buffers

`int conv2d_layer_forward(Conv2DLayer *layer, const float *input, float *output)`

- input shape: `[in_c * in_h * in_w]` in CHW
- output shape: `[out_c * out_h * out_w]` in CHW
- operation: convolution + bias + activation

`int conv2d_layer_backward(const Conv2DLayer *layer, const float *delta_in, float *delta_out, float *grad_w, float *grad_b)`

- `delta_in`: `[out_c * out_h * out_w]`
- `delta_out`: `[in_c * in_h * in_w]` or `NULL`
- `grad_w`: `[out_c * in_c * kernel_h * kernel_w]`
- `grad_b`: `[out_c]`
- computes gradients and optional propagated delta

### MaxPool2D API

`int maxpool2d_layer_init(MaxPool2DLayer *layer, int input_width, int input_height, int channels, int pool_width, int pool_height, int stride, int padding)`

- validates geometry and divisibility
- computes output dimensions
- allocates input and argmax caches
- returns `-1` on invalid geometry or allocation failure

`void maxpool2d_layer_free(MaxPool2DLayer *layer)`

- frees maxpool buffers

`int maxpool2d_layer_forward(MaxPool2DLayer *layer, const float *input, float *output)`

- input shape: `[c * in_h * in_w]` in CHW
- output shape: `[c * out_h * out_w]` in CHW
- stores winning input indices into `cache_max_indices`

`int maxpool2d_layer_backward(const MaxPool2DLayer *layer, const float *delta_in, float *delta_out, float *grad_w, float *grad_b)`

- routes each output gradient element back to its cached argmax source index
- `delta_in`: `[c * out_h * out_w]`
- `delta_out`: `[c * in_h * in_w]` or `NULL`
- `grad_w` and `grad_b` are required by plugin interface but are dummy outputs for maxpool

## Header: `layers.h`

### Plugin constructors

`int layer_plugin_dense_create(int input_size, int output_size, Activation activation, LayerPlugin *out_plugin)`

- creates dense plugin context
- wires callback table

`int layer_plugin_conv2d_create(int input_width, int input_height, int input_channels, int output_channels, int kernel_width, int kernel_height, int stride, int padding, Activation activation, LayerPlugin *out_plugin)`

- creates conv2d plugin context
- CHW flattened input/output

`int layer_plugin_maxpool2d_create(int input_width, int input_height, int channels, int pool_width, int pool_height, int stride, int padding, LayerPlugin *out_plugin)`

- creates maxpool2d plugin context
- CHW flattened input/output

All plugin constructors:

- return `0` on success, `-1` on invalid input/allocation failure
- transfer context ownership into `out_plugin`

### Plugin cleanup

`void layer_plugin_free(LayerPlugin *plugin)`

- calls `plugin->destroy(plugin->ctx)` if present
- sets pointers to `NULL`

## Header: `lossfunctions.h`

`float loss_mse(const float *pred, const float *target, int size)`

- computes mean squared error over `size`

`int loss_mse_grad(const float *pred, const float *target, int size, float *grad_out)`

- writes gradient with respect to `pred`

`float loss_bce(const float *pred, const float *target, int size)`

- computes binary cross-entropy over `size`

`int loss_bce_grad(const float *pred, const float *target, int size, float *grad_out)`

- writes BCE gradient with respect to `pred`

Expected for all loss APIs:

- `pred`, `target`, and `grad_out` (where used) must be valid pointers
- vector length is `size`

## Header: `optimizers.h`

`int sgd_optimizer(float *weights, float *grads, float learning_rate, int size)`

- in-place SGD parameter update

`int adam_optimizer(float *weights, float *grads, float *m, float *v, float beta1, float beta2, float learning_rate, int t, int size)`

- in-place Adam update with bias correction step `t`

`int rmsprop_optimizer(float *weights, float *grads, float *cache, float beta, float learning_rate, int size)`

- in-place RMSProp update

For all optimizers:

- pointers must be valid
- `size > 0`
- learning rate must be positive

## Header: `models.h`

### `SequentialModel` lifecycle

`int sequential_model_init(SequentialModel *model, int initial_capacity)`

- allocates plugin array
- `initial_capacity` must be `> 0`

`void sequential_model_free(SequentialModel *model)`

- frees owned plugin layers
- releases compiled optimizer state if internally owned

### Layer composition helpers

`int sequential_model_add_layer(SequentialModel *model, LayerPlugin layer)`

- takes ownership of `layer`
- validates callback set
- grows internal capacity as needed

`int sequential_model_add_dense(SequentialModel *model, int input_size, int output_size, Activation activation)`

`int sequential_model_add_conv2d(SequentialModel *model, int input_width, int input_height, int input_channels, int output_channels, int kernel_width, int kernel_height, int stride, int padding, Activation activation)`

`int sequential_model_add_maxpool2d(SequentialModel *model, int input_width, int input_height, int channels, int pool_width, int pool_height, int stride, int padding)`

All add helpers:

- create plugin layer and append to model
- return `-1` on invalid inputs or allocation failure

Dimension chaining is manual. Example:

- `conv: 28x28x1 -> 28x28x8` with `k=3, s=1, p=1`
- `pool: 28x28x8 -> 14x14x8` with `2x2, s=2`
- dense input must then be `14 * 14 * 8`

### Inference and initialization

`int sequential_model_forward(SequentialModel *model, const float *input, float *output)`

- runs forward through every plugin layer

`int sequential_model_predict(SequentialModel *model, const float *input, float *output)`

- alias to `sequential_model_forward`

`int sequential_model_randomize(SequentialModel *model, float init_scale)`

- randomizes each layer's parameter buffers
- maxpool dummy parameter buffers are also touched for uniform pipeline behavior

### Compiled train loop API

`int sequential_model_compile(SequentialModel *model, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, float adam_beta1, float adam_beta2)`

- stores selected loss/optimizer/rate in model
- allocates optimizer state internally for Adam and RMSProp

`int sequential_model_train(SequentialModel *model, const float *inputs, const float *targets, int num_samples, int input_size, int target_size, int epochs, float *final_loss_out)`

- loops epochs and samples with compiled settings
- expected batch layout:
  - `inputs`: `[num_samples * input_size]` contiguous row-major
  - `targets`: `[num_samples * target_size]` contiguous row-major

### External optimizer-state helpers

`int sequential_model_adam_state_init(SequentialModel *model, AdamOptimizerState *out_state, float beta1, float beta2)`

- allocates per-layer state buffers
- `out_state` should be empty (`NULL` pointers / zeroed fields)

`void sequential_model_adam_state_free(SequentialModel *model, AdamOptimizerState *state)`

- releases buffers created by state init

### Config-driven one-step API

`void sequential_train_config_init_sgd(SequentialTrainConfig *cfg, LossFunctionType loss_function, float learning_rate)`

`void sequential_train_config_init_adam(SequentialTrainConfig *cfg, LossFunctionType loss_function, float learning_rate, AdamOptimizerState *adam_state)`

`int sequential_model_train_step_cfg(SequentialModel *model, const float *input, const float *target, float *output, const SequentialTrainConfig *cfg, float *loss_out)`

- compact repeated training interface

### Plugin-sequential training helpers

`int sequential_model_train_step_sgd(SequentialModel *model, const float *input, const float *target, float *output, float learning_rate, float *loss_out)`

`int sequential_model_train_step(SequentialModel *model, const float *input, const float *target, float *output, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_model_train_step_with_loss(SequentialModel *model, const float *input, const float *target, float *output, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_model_train_step_mse(SequentialModel *model, const float *input, const float *target, float *output, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_model_train_step_bce(SequentialModel *model, const float *input, const float *target, float *output, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

Notes:

- these perform forward + loss + backward + parameter update in one call
- for `OPTIMIZER_ADAM` and `OPTIMIZER_RMSPROP`, valid optimizer state is required
- `loss_out` may be `NULL`

### Plugin-sequential optimize-from-prediction helpers

`int sequential_model_optimize_from_prediction(SequentialModel *model, const float *prediction, const float *target, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_model_optimize_from_prediction_with_loss(SequentialModel *model, const float *prediction, const float *target, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_model_optimize_from_prediction_mse(SequentialModel *model, const float *prediction, const float *target, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_model_optimize_from_prediction_bce(SequentialModel *model, const float *prediction, const float *target, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

Use when you already called `sequential_model_forward` and want split forward/optimize phases.

### Layer-array inference and training API

`int sequential_forward(Layer *layers, int num_layers, const float *input, float *output)`

`int sequential_train_step_sgd(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, float learning_rate, float *loss_out)`

`int sequential_train_step(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_train_step_with_loss(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_train_step_mse(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_train_step_bce(Layer *layers, int num_layers, const float *input, const float *target, float *output, float **grads_w, float **grads_b, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_optimize_from_prediction(Layer *layers, int num_layers, const float *prediction, const float *target, float **grads_w, float **grads_b, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_optimize_from_prediction_with_loss(Layer *layers, int num_layers, const float *prediction, const float *target, float **grads_w, float **grads_b, LossFunctionType loss_function, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_optimize_from_prediction_mse(Layer *layers, int num_layers, const float *prediction, const float *target, float **grads_w, float **grads_b, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

`int sequential_optimize_from_prediction_bce(Layer *layers, int num_layers, const float *prediction, const float *target, float **grads_w, float **grads_b, OptimizerType optimizer, float learning_rate, AdamOptimizerState *adam_state, float *loss_out)`

For layer-array APIs, caller responsibilities:

- allocate all `grads_w[i]` and `grads_b[i]`
- ensure each gradient buffer matches that layer's dense dimensions
- keep optimizer state persistent across steps for Adam/RMSProp

## Practical Failure Checklist (`-1`)

Common causes across APIs:

- null required pointers
- non-positive sizes
- invalid conv/pool geometry
- output/input shape mismatch between chained layers
- missing optimizer state for Adam/RMSProp helpers
- allocation failure

## Minimal End-to-End Snippets

### Dense-only sequential

```c
SequentialModel model;
float out[1];

sequential_model_init(&model, 2);
sequential_model_add_dense(&model, 2, 4, ACT_RELU);
sequential_model_add_dense(&model, 4, 1, ACT_SIGMOID);
sequential_model_randomize(&model, 0.1f);
sequential_model_forward(&model, input_vec, out);
sequential_model_free(&model);
```

### Conv2D + MaxPool2D + Dense sequential

```c
SequentialModel model;
float logits[10];

sequential_model_init(&model, 4);
sequential_model_add_conv2d(&model, 28, 28, 1, 8, 3, 3, 1, 1, ACT_RELU);
sequential_model_add_maxpool2d(&model, 28, 28, 8, 2, 2, 2, 0);
sequential_model_add_dense(&model, 14 * 14 * 8, 10, ACT_SIGMOID);

sequential_model_randomize(&model, 0.1f);
sequential_model_forward(&model, image_chw_flat, logits);
sequential_model_free(&model);
```

### Split forward and optimize phase

```c
float loss = 0.0f;
sequential_model_forward(&model, input_vec, output_vec);
sequential_model_optimize_from_prediction_with_loss(&model,
                                                    output_vec,
                                                    target_vec,
                                                    LOSS_BCE,
                                                    OPTIMIZER_ADAM,
                                                    0.001f,
                                                    &adam_state,
                                                    &loss);
```
