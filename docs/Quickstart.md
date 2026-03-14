# Quickstart

This quickstart shows how to build `lib-neuron`, run shipped examples, and start using the new Conv2D and MaxPool2D APIs.

## 1) Build the library

From project root:

```sh
make
```

This produces `libneuron.a` and `libneuron.so`.

## 2) Build all examples from root

```sh
make examples
```

This builds:

- `examples/simple_compact`
- `examples/sequential_xor_plugin`
- `examples/Other_Exaple`

## 3) Run the examples

```sh
./examples/simple_compact
./examples/sequential_xor_plugin
./examples/Other_Exaple
```

`Other_Exaple` is a compact layer-array training example.

## 4) Conv2D + MaxPool2D quick usage

`SequentialModel` supports plugin helpers for dense, conv2d, and maxpool2d layers.

```c
SequentialModel model;

sequential_model_init(&model, 4);
sequential_model_add_conv2d(&model, 28, 28, 1, 8, 3, 3, 1, 1, ACT_RELU);
sequential_model_add_maxpool2d(&model, 28, 28, 8, 2, 2, 2, 0);
/* output shape after pool: 14x14x8 -> 1568 */
sequential_model_add_dense(&model, 14 * 14 * 8, 10, ACT_SIGMOID);
```

Conv/pool buffers are flattened in CHW order.

## 5) Compile your own program

```sh
gcc your_program.c -Iinclude ./libneuron.a -lm -o your_program
```

Include all public APIs with:

```c
#include <lib-neuron.h>
```

For full signatures and shape rules, see `docs/APIReference.md`.
