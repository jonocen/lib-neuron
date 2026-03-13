# lib-neuron

`lib-neuron` is work in progress C-libary for Neuronal Networks. its in a a "Beta" Phase with some vibe code that i will change.

## Status

Work in progress. a LOT of stuff needs to be added and a lot of AI stuff needs help.

## Why AI and where is it

i dont now a bit of the stuff thats needed but may you do, and i wanted to make it acceseble now

AI code is in models and matrixcalculations, so please check

## Modules

- **matrixcalculation**: layer structure, activations, forward pass, backpropagation
- **lossfunctions**: MSE and BCE losses, plus their gradients
- **optimizers**: SGD and Adam parameter updates
- **models**: simple sequential model helpers

You can include everything with a single header:

```c
#include <lib-neuron.h>
```

## Project layout

- `include/` — public headers
- `src/` — implementations

## Return convention

Most functions return a status code:

- `0` = success
- `-1` = invalid input or internal failure

Array data (weights, gradients, activations) is updated **in place** via pointers.

## Build

The repository includes a Makefile and builds **libraries only**.

Build the static library with:

```sh
make
```

This generates `libneuron.a`.

Build the shared library with:

```sh
make shared
```

This generates `libneuron.so`.

To clean build artifacts:

```sh
make clean
```

## Link example

When compiling your own executable, link against the library and math module:

```sh
gcc your_program.c -Iinclude -L. -lneuron -lm
```

## Contributing

its a little bit of bad code, if you find a pice of code that is bad please contribute to this project or make a issue report on github. Thanks a lot!!!!
