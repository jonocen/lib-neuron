#include <lib-neuron.h>
#include <stdio.h>
#include <stdlib.h>

static void init_small_random(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }
}

int main(void) {
    Layer layers[2];

    /* 2 inputs -> 4 hidden neurons */
    if (layer_init(&layers[0], 2, 4, ACT_RELU) != 0) return 1;

    /* 4 hidden neurons -> 1 output */
    if (layer_init(&layers[1], 4, 1, ACT_SIGMOID) != 0) return 1;

    init_small_random(layers[0].weights, 2 * 4);
    init_small_random(layers[1].weights, 4 * 1);

    float *grads_w[2];
    float *grads_b[2];

    grads_w[0] = malloc(sizeof(float) * 2 * 4);
    grads_b[0] = malloc(sizeof(float) * 4);
    grads_w[1] = malloc(sizeof(float) * 4 * 1);
    grads_b[1] = malloc(sizeof(float) * 1);

    if (!grads_w[0] || !grads_b[0] || !grads_w[1] || !grads_b[1]) return 1;

    /* XOR dataset */
    float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    float targets[4][1] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    float output[1];
    float loss = 0.0f;
    float learning_rate = 0.05f;

    for (int epoch = 0; epoch < 5000; epoch++) {
        float epoch_loss = 0.0f;

        for (int i = 0; i < 4; i++) {
            if (sequential_train_step_sgd(
                    layers,
                    2,
                    inputs[i],
                    targets[i],
                    output,
                    grads_w,
                    grads_b,
                    learning_rate,
                    &loss) != 0) {
                return 1;
            }

            epoch_loss += loss;
        }

        if (epoch % 500 == 0) {
            printf("epoch %d loss = %f\n", epoch, epoch_loss / 4.0f);
        }
    }

    puts("predictions:");
    for (int i = 0; i < 4; i++) {
        if (sequential_forward(layers, 2, inputs[i], output) != 0) return 1;
        printf("[%g, %g] -> %f\n", inputs[i][0], inputs[i][1], output[0]);
    }

    free(grads_w[0]);
    free(grads_b[0]);
    free(grads_w[1]);
    free(grads_b[1]);

    layer_free(&layers[0]);
    layer_free(&layers[1]);

    return 0;
}