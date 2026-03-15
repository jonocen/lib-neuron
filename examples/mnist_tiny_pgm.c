#include <lib-neuron.h>

#include <stdio.h>
#include <stdlib.h>

#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28
#define MNIST_CLASSES 10

int main(int argc, char **argv) {
    int epochs;
    int batch_size;
    float lr;
    int samples_per_class;
    ImageDataset ds;
    SequentialModel model;
    float final_loss = 0.0f;
    float pred[MNIST_CLASSES];
    int true_digit = -1;
    int pred_digit = -1;

    epochs = 2000;
    batch_size = (argc > 2) ? atoi(argv[2]) : 32;
    lr = (argc > 3) ? (float)atof(argv[3]) : 0.001f;
    samples_per_class = (argc > 4) ? atoi(argv[4]) : 24;

    if (epochs <= 0 || batch_size <= 0 || lr <= 0.0f || samples_per_class <= 0) {
        puts("Usage: ./examples/mnist_tiny_pgm [epochs] [batch_size] [lr] [samples_per_class]");
        return 1;
    }

    ds.inputs = NULL;
    ds.targets = NULL;

    if (image_dataset_make_tiny_digits(&ds,
                                       samples_per_class,
                                       MNIST_WIDTH,
                                       MNIST_HEIGHT,
                                       1337u) != 0) {
        puts("Failed to build synthetic tiny-digit dataset.");
        return 1;
    }

    if (sequential_model_init(&model, 16) != 0 ||
        sequential_model_set_input_shape2d(&model, MNIST_WIDTH, MNIST_HEIGHT, 1) != 0 ||
        sequential_model_add_conv2d(&model, 28, 28, 1, 4, 4, 4, 1, 0, ACT_LINEAR) != 0 ||
        sequential_model_add_maxpool2d(&model, 25, 25, 4, 5, 5, 5, 0) != 0 ||
        sequential_model_add_conv2d(&model, 5, 5, 4, 4, 2, 2, 1, 0, ACT_LINEAR) != 0 ||
        sequential_model_add_maxpool2d(&model, 4, 4, 4, 2, 2, 2, 0) != 0 ||
        sequential_model_add_flatten(&model) != 0 ||
        sequential_model_add_dense(&model, 16, 32, ACT_RELU) != 0 ||
        sequential_model_add_dense(&model, 32, 64, ACT_RELU) != 0 ||
        sequential_model_add_dense(&model, 64, 64, ACT_RELU) != 0 ||
        sequential_model_add_dense(&model, 64, MNIST_CLASSES, ACT_LINEAR) != 0 ||
        sequential_model_randomize(&model, 0.1f) != 0) {
        image_dataset_free(&ds);
        puts("Failed to build model.");
        return 1;
    }

    if (sequential_model_compile(&model, LOSS_MSE, OPTIMIZER_ADAMW, lr, 0.9f, 0.999f) != 0) {
        sequential_model_free(&model);
        image_dataset_free(&ds);
        puts("Failed to compile model.");
        return 1;
    }

    if (sequential_model_train_with_progress(&model,
                                             ds.inputs,
                                             ds.targets,
                                             ds.num_samples,
                                             ds.input_size,
                                             ds.target_size,
                                             epochs,
                                             batch_size,
                                             10,
                                             &final_loss) != 0) {
        sequential_model_free(&model);
        image_dataset_free(&ds);
        puts("Training failed.");
        return 1;
    }

    printf("Trained on %d samples. final_loss=%.6f\n", ds.num_samples, final_loss);

    /* Predict first sample from dataset tensors. */
    if (sequential_model_predict(&model, ds.inputs, pred) == 0 &&
        image_argmax(pred, MNIST_CLASSES, &pred_digit) == 0 &&
        image_dataset_get_label(&ds, 0, &true_digit) == 0) {
        printf("First sample label=%d pred=%d\n", true_digit, pred_digit);
    }

    sequential_model_free(&model);
    image_dataset_free(&ds);

    return 0;
}
