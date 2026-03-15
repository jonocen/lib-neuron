#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "models.h"

typedef struct {
    float *inputs;
    float *targets;
    int    num_samples;
    int    image_width;
    int    image_height;
    int    image_channels;
    int    input_size;
    int    target_size;
} ImageDataset;

/* Convert uint8 image buffer to normalized float buffer in [0, 1]. */
int image_convert_u8_to_f32(const unsigned char *src,
                            int element_count,
                            float *dst);

/* Load a grayscale PGM image (P2 or P5), normalized to [0, 1]. */
int image_load_pgm(const char *file_path,
                   float **out_pixels,
                   int *out_width,
                   int *out_height);

/*
 * Build an in-memory dataset from labeled PGM image paths.
 * Labels are converted to one-hot vectors with size num_classes.
 */
int image_dataset_load_pgm_labeled(const char **image_paths,
                                   const int *labels,
                                   int num_samples,
                                   int num_classes,
                                   int expected_width,
                                   int expected_height,
                                   ImageDataset *out_dataset);

/*
 * Load a labeled dataset from a text manifest with lines:
 * <path_to_image.pgm> <label>
 */
int image_dataset_load_pgm_manifest(const char *manifest_path,
                                    int num_classes,
                                    int expected_width,
                                    int expected_height,
                                    ImageDataset *out_dataset);

/* Build a tiny synthetic digit dataset (MNIST-like shape) in memory. */
int image_dataset_make_tiny_digits(ImageDataset *out_dataset,
                                   int samples_per_class,
                                   int width,
                                   int height,
                                   unsigned int seed);

void image_dataset_free(ImageDataset *dataset);

/* Return index of max value in a score vector. */
int image_argmax(const float *scores,
                 int count,
                 int *out_index);

/* Extract integer class label from one-hot target row for a sample. */
int image_dataset_get_label(const ImageDataset *dataset,
                            int sample_index,
                            int *out_label);

/* Train a compiled model using a prepared image dataset. */
int sequential_model_train_image_dataset(SequentialModel *model,
                                         const ImageDataset *dataset,
                                         int epochs,
                                         int batch_size,
                                         float *final_loss_out);

/* Convenience helper: load one PGM and run prediction. */
int sequential_model_predict_pgm(SequentialModel *model,
                                 const char *file_path,
                                 int expected_width,
                                 int expected_height,
                                 float *output);

#endif /* IMAGE_PROCESSING_H */
