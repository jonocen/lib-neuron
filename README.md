# lib-neuron

`lib-neuron` is a lightweight C neural-network library focused on small projects and educational use.

## Current features

- Dense layers with activation and backpropagation
- Conv2D layers with configurable kernel/stride/padding
- MaxPool2D layers with argmax-based backward pass
- Sequential plugin model API
- Loss functions: MSE and BCE
- Optimizers: SGD, Adam, RMSProp

## Public modules

- `matrixcalculation`: activations, dense layers, conv2d, maxpool2d
- `layers`: plugin wrappers (`LayerPlugin`) for sequential models
- `models`: sequential training/inference helpers
- `lossfunctions`: MSE/BCE losses and gradients
- `optimizers`: SGD/Adam/RMSProp updates

Include everything with:

```c
#include <lib-neuron.h>
```

## Return convention

- `0` means success
- `-1` means invalid input or internal failure

Array data (weights, gradients, activations) is updated in place via pointers.

## Build

Build static library:

```sh
make
```

Build shared library:

```sh
make shared
```

Clean artifacts:

```sh
make clean
```

## Build and run examples

Build all:

```sh
make examples
```

Run:

```sh
./examples/simple_compact
./examples/sequential_xor_plugin
./examples/Other_Exaple
```

## Compile your own program

```sh
gcc your_program.c -Iinclude -L. -lneuron -lm
```

## Docs

- `docs/README.md`
- `docs/Quickstart.md`
- `docs/Examples.md`
- `docs/FirstScript.md`
- `docs/Training.md`
- `docs/APIReference.md`

`docs/APIReference.md` now includes detailed Conv2D and MaxPool2D API behavior, tensor layout rules, and sequential plugin notes.
