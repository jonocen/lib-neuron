#include <lib-neuron.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float x[2];
    float y[1];
} Sample;

static void init_small_random(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }
}

int main(void) {
    SequentialModel model;
    float output[1];
    float loss = 0.0f;

    Sample xor_data[4] = {
        {{0.0f, 0.0f}, {0.0f}},
        {{0.0f, 1.0f}, {1.0f}},
        {{1.0f, 0.0f}, {1.0f}},
        {{1.0f, 1.0f}, {0.0f}},
    };

    if (sequential_model_init(&model, 2) != 0) return 1;
    if (sequential_model_add_dense(&model, 2, 4, ACT_RELU) != 0) return 1;
    if (sequential_model_add_dense(&model, 4, 1, ACT_SIGMOID) != 0) return 1;

    /* Optional: randomize parameters (default init is zeros) */
    for (int i = 0; i < model.num_layers; i++) {
        float *w = model.layers[i].weights(model.layers[i].ctx);
        int w_size = model.layers[i].weights_size(model.layers[i].ctx);
        init_small_random(w, w_size);
    }

    for (int epoch = 0; epoch < 5000; epoch++) {
        float epoch_loss = 0.0f;

        for (int i = 0; i < 4; i++) {
            if (sequential_model_train_step_sgd(
                    &model,
                    xor_data[i].x,
                    xor_data[i].y,
                    output,
                    0.05f,
                    &loss) != 0) {
                sequential_model_free(&model);
                return 1;
            }
            epoch_loss += loss;
        }

        if (epoch % 500 == 0) {
            printf("epoch %d loss = %.6f\n", epoch, epoch_loss / 4.0f);
        }
    }

    puts("predictions:");
    for (int i = 0; i < 4; i++) {
        if (sequential_model_forward(&model, xor_data[i].x, output) != 0) {
            sequential_model_free(&model);
            return 1;
        }
        printf("[%g, %g] -> %.6f\n", xor_data[i].x[0], xor_data[i].x[1], output[0]);
    }

    sequential_model_free(&model);
    return 0;
}
