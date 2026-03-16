# Add a New Activation Function (Beginner Guide)

This guide is for beginners who want to add a new activation function safely in `lib-neuron`.

Goal:
- Add one new activation (example: `ACT_LEAKY_RELU`)
- Make it work in all layer types (dense, conv2d)
- Keep project behavior consistent

You can follow this exactly, top to bottom.

## Before You Start

You will edit these files:

1. `include/activationfunctions.h`
2. `src/activationfunctions.c`

That's it. The rest of the library (layers, training, backprop) calls `act_apply` and `act_deriv`
generically, so it picks up new activations automatically.

Run this after each step:

```sh
make
```

## Mental Model (Simple)

Every activation in this project has 2 parts:

1. A forward function: `act_apply(x, a)` returns the output value.
2. A derivative function: `act_deriv(x, a)` returns the slope at `x`.

Both are dispatched by a `switch` on the `Activation` enum.
If you add one without the other, training gradients will be wrong.

## Step 1. Add the enum value

Edit `include/activationfunctions.h` and add your new value to the enum.

```c
typedef enum {
    ACT_LINEAR,
    ACT_RELU,
    ACT_SIGMOID,
    ACT_TANH,
    ACT_LEAKY_RELU   /* <-- add here */
} Activation;
```

Rules:
- Place the new value before the closing `}`.
- Do not assign explicit numbers unless you have a stored/serialized model that depends on them.
- Save and run `make` to check for compile errors.

## Step 2. Add the forward case

Edit `src/activationfunctions.c` and add a `case` in `act_apply`.

```c
float act_apply(float x, Activation a) {
    switch (a) {
        case ACT_RELU:        return x > 0.0f ? x : 0.0f;
        case ACT_SIGMOID:     return 1.0f / (1.0f + expf(-x));
        case ACT_TANH:        return tanhf(x);
        case ACT_LEAKY_RELU:  return x > 0.0f ? x : 0.01f * x;  /* <-- add */
        case ACT_LINEAR:      /* fall-through */
        default:              return x;
    }
}
```

Rules:
- Add your case before `ACT_LINEAR`.
- Return a `float` expression — no side effects.
- Run `make` after changing this.

## Step 3. Add the derivative case

Still in `src/activationfunctions.c`, add a `case` in `act_deriv`.

```c
float act_deriv(float x, Activation a) {
    float s;
    switch (a) {
        case ACT_RELU:        return x > 0.0f ? 1.0f : 0.0f;
        case ACT_SIGMOID:     s = act_apply(x, ACT_SIGMOID); return s * (1.0f - s);
        case ACT_TANH:        s = tanhf(x); return 1.0f - s * s;
        case ACT_LEAKY_RELU:  return x > 0.0f ? 1.0f : 0.01f;  /* <-- add */
        case ACT_LINEAR:      /* fall-through */
        default:              return 1.0f;
    }
}
```

Rules:
- The derivative must match the forward function mathematically.
- If your forward uses a cached value (like `sigmoid` uses `s`), recompute it here using `act_apply`.
- Run `make` after changing this.

## Step 4. Verify it builds

```sh
make clean && make
```

If it compiles without errors, the new activation is ready to use:

```c
sequential_model_add_dense(&model, 64, 32, ACT_LEAKY_RELU);
```

## Common Mistakes

| Mistake | Effect |
|---|---|
| Added enum but no cases | Compiler warning about unhandled enum value |
| Added forward but not derivative | Gradients are wrong, training diverges |
| Returned the wrong sign in the derivative | Gradients flip, training diverges |
| Used `double` math instead of `float` | Performance loss; possible `-Wpedantic` warning |

## Reference

The activation system lives entirely in:
- `include/activationfunctions.h` — enum and declarations
- `src/activationfunctions.c` — implementations

All layers access activations through `act_apply` and `act_deriv` from `matrixcalculation.h`,
which includes `activationfunctions.h` automatically.
