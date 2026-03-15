#include "../include/image_processing.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_token(FILE *f, char *buf, size_t buf_size) {
	int c;
	size_t i = 0;

	if (!f || !buf || buf_size == 0) return -1;

	do {
		c = fgetc(f);
		if (c == '#') {
			while (c != '\n' && c != EOF) c = fgetc(f);
		}
	} while (c != EOF && isspace((unsigned char)c));

	if (c == EOF) return -1;

	while (c != EOF && !isspace((unsigned char)c)) {
		if (c == '#') {
			while (c != '\n' && c != EOF) c = fgetc(f);
			break;
		}
		if (i + 1 >= buf_size) return -1;
		buf[i++] = (char)c;
		c = fgetc(f);
	}

	if (i == 0) return -1;
	buf[i] = '\0';
	return 0;
}

static char *image_strdup_local(const char *s) {
	size_t n;
	char *out;

	if (!s) return NULL;

	n = strlen(s) + 1;
	out = malloc(n);
	if (!out) return NULL;

	memcpy(out, s, n);
	return out;
}

static unsigned int image_rand_next(unsigned int *state) {
	*state = (*state * 1664525u) + 1013904223u;
	return *state;
}

static int image_rand_range(unsigned int *state, int min_v, int max_v) {
	unsigned int span;

	if (min_v >= max_v) return min_v;
	span = (unsigned int)(max_v - min_v + 1);
	return min_v + (int)(image_rand_next(state) % span);
}

static float image_rand_unit(unsigned int *state) {
	return (float)(image_rand_next(state) & 0x00FFFFFFu) / 16777215.0f;
}

static void image_draw_rect(float *img,
						int w,
						int h,
						int x0,
						int y0,
						int x1,
						int y1,
						float value) {
	if (!img || w <= 0 || h <= 0) return;

	if (x0 > x1) {
		int t = x0;
		x0 = x1;
		x1 = t;
	}
	if (y0 > y1) {
		int t = y0;
		y0 = y1;
		y1 = t;
	}

	if (x1 < 0 || y1 < 0 || x0 >= w || y0 >= h) return;

	if (x0 < 0) x0 = 0;
	if (y0 < 0) y0 = 0;
	if (x1 >= w) x1 = w - 1;
	if (y1 >= h) y1 = h - 1;

	for (int y = y0; y <= y1; y++) {
		for (int x = x0; x <= x1; x++) {
			img[(size_t)y * (size_t)w + (size_t)x] = value;
		}
	}
}

static void image_draw_7seg_digit(float *img,
							 int w,
							 int h,
							 int digit,
							 int jitter_x,
							 int jitter_y,
							 int thickness) {
	/* Segment order: a b c d e f g */
	static const unsigned char segmap[10] = {
		0x3Fu, /* 0 */
		0x06u, /* 1 */
		0x5Bu, /* 2 */
		0x4Fu, /* 3 */
		0x66u, /* 4 */
		0x6Du, /* 5 */
		0x7Du, /* 6 */
		0x07u, /* 7 */
		0x7Fu, /* 8 */
		0x6Fu  /* 9 */
	};
	unsigned char mask;
	int left;
	int right;
	int top;
	int mid;
	int bottom;

	if (!img || w < 8 || h < 8 || digit < 0 || digit > 9) return;

	left = 5 + jitter_x;
	right = (w - 6) + jitter_x;
	top = 3 + jitter_y;
	mid = h / 2 + jitter_y;
	bottom = (h - 4) + jitter_y;
	if (thickness < 1) thickness = 1;

	mask = segmap[digit];

	/* a */
	if (mask & 0x01u) image_draw_rect(img, w, h, left, top, right, top + thickness, 1.0f);
	/* b */
	if (mask & 0x02u) image_draw_rect(img, w, h, right - thickness, top, right, mid, 1.0f);
	/* c */
	if (mask & 0x04u) image_draw_rect(img, w, h, right - thickness, mid, right, bottom, 1.0f);
	/* d */
	if (mask & 0x08u) image_draw_rect(img, w, h, left, bottom - thickness, right, bottom, 1.0f);
	/* e */
	if (mask & 0x10u) image_draw_rect(img, w, h, left, mid, left + thickness, bottom, 1.0f);
	/* f */
	if (mask & 0x20u) image_draw_rect(img, w, h, left, top, left + thickness, mid, 1.0f);
	/* g */
	if (mask & 0x40u) image_draw_rect(img, w, h, left, mid - (thickness / 2), right, mid + (thickness / 2), 1.0f);
}

int image_convert_u8_to_f32(const unsigned char *src,
						int element_count,
						float *dst) {
	if (!src || !dst || element_count <= 0) return -1;

	for (int i = 0; i < element_count; i++) {
		dst[i] = (float)src[i] / 255.0f;
	}

	return 0;
}

int image_load_pgm(const char *file_path,
				   float **out_pixels,
				   int *out_width,
				   int *out_height) {
	FILE *f = NULL;
	char token[64];
	int width, height, maxval;
	int is_binary;
	float *pixels = NULL;

	if (!file_path || !out_pixels || !out_width || !out_height) return -1;

	*out_pixels = NULL;
	*out_width = 0;
	*out_height = 0;

	f = fopen(file_path, "rb");
	if (!f) return -1;

	if (read_token(f, token, sizeof(token)) != 0) {
		fclose(f);
		return -1;
	}

	if (strcmp(token, "P5") == 0) {
		is_binary = 1;
	} else if (strcmp(token, "P2") == 0) {
		is_binary = 0;
	} else {
		fclose(f);
		return -1;
	}

	if (read_token(f, token, sizeof(token)) != 0) {
		fclose(f);
		return -1;
	}
	width = atoi(token);

	if (read_token(f, token, sizeof(token)) != 0) {
		fclose(f);
		return -1;
	}
	height = atoi(token);

	if (read_token(f, token, sizeof(token)) != 0) {
		fclose(f);
		return -1;
	}
	maxval = atoi(token);

	if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) {
		fclose(f);
		return -1;
	}

	pixels = malloc((size_t)width * (size_t)height * sizeof(float));
	if (!pixels) {
		fclose(f);
		return -1;
	}

	if (is_binary) {
		unsigned char *raw = NULL;
		size_t n = (size_t)width * (size_t)height;
		int c = fgetc(f);
		if (c == EOF) {
			free(pixels);
			fclose(f);
			return -1;
		}
		if (!isspace((unsigned char)c)) {
			if (ungetc(c, f) == EOF) {
				free(pixels);
				fclose(f);
				return -1;
			}
		}

		raw = malloc(n);
		if (!raw) {
			free(pixels);
			fclose(f);
			return -1;
		}

		if (fread(raw, 1, n, f) != n) {
			free(raw);
			free(pixels);
			fclose(f);
			return -1;
		}

		if (maxval == 255) {
			if (image_convert_u8_to_f32(raw, width * height, pixels) != 0) {
				free(raw);
				free(pixels);
				fclose(f);
				return -1;
			}
		} else {
			for (int i = 0; i < width * height; i++) {
				pixels[i] = (float)raw[i] / (float)maxval;
			}
		}

		free(raw);
	} else {
		for (int i = 0; i < width * height; i++) {
			int px;
			if (read_token(f, token, sizeof(token)) != 0) {
				free(pixels);
				fclose(f);
				return -1;
			}
			px = atoi(token);
			if (px < 0 || px > maxval) {
				free(pixels);
				fclose(f);
				return -1;
			}
			pixels[i] = (float)px / (float)maxval;
		}
	}

	fclose(f);
	*out_pixels = pixels;
	*out_width = width;
	*out_height = height;
	return 0;
}

int image_dataset_load_pgm_labeled(const char **image_paths,
								   const int *labels,
								   int num_samples,
								   int num_classes,
								   int expected_width,
								   int expected_height,
								   ImageDataset *out_dataset) {
	float *inputs = NULL;
	float *targets = NULL;

	if (!image_paths || !labels || !out_dataset || num_samples <= 0 ||
		num_classes <= 1 || expected_width <= 0 || expected_height <= 0) {
		return -1;
	}

	memset(out_dataset, 0, sizeof(*out_dataset));

	inputs = calloc((size_t)num_samples * (size_t)expected_width * (size_t)expected_height, sizeof(float));
	targets = calloc((size_t)num_samples * (size_t)num_classes, sizeof(float));
	if (!inputs || !targets) {
		free(inputs);
		free(targets);
		return -1;
	}

	for (int i = 0; i < num_samples; i++) {
		float *sample_pixels = NULL;
		int w = 0, h = 0;

		if (labels[i] < 0 || labels[i] >= num_classes) {
			free(inputs);
			free(targets);
			return -1;
		}

		if (image_load_pgm(image_paths[i], &sample_pixels, &w, &h) != 0) {
			free(inputs);
			free(targets);
			return -1;
		}

		if (w != expected_width || h != expected_height) {
			free(sample_pixels);
			free(inputs);
			free(targets);
			return -1;
		}

		memcpy(inputs + ((size_t)i * (size_t)expected_width * (size_t)expected_height),
			   sample_pixels,
			   (size_t)expected_width * (size_t)expected_height * sizeof(float));
		free(sample_pixels);

		targets[(size_t)i * (size_t)num_classes + (size_t)labels[i]] = 1.0f;
	}

	out_dataset->inputs = inputs;
	out_dataset->targets = targets;
	out_dataset->num_samples = num_samples;
	out_dataset->image_width = expected_width;
	out_dataset->image_height = expected_height;
	out_dataset->image_channels = 1;
	out_dataset->input_size = expected_width * expected_height;
	out_dataset->target_size = num_classes;
	return 0;
}

int image_dataset_load_pgm_manifest(const char *manifest_path,
								int num_classes,
								int expected_width,
								int expected_height,
								ImageDataset *out_dataset) {
	FILE *f = NULL;
	char line[4096];
	char path_buf[3072];
	char **paths = NULL;
	int *labels = NULL;
	const char **paths_const = NULL;
	int count = 0;
	int capacity = 0;
	int rc = -1;

	if (!manifest_path || !out_dataset || num_classes <= 1 || expected_width <= 0 || expected_height <= 0) {
		return -1;
	}

	f = fopen(manifest_path, "r");
	if (!f) return -1;

	while (fgets(line, (int)sizeof(line), f)) {
		char *p = line;
		int label = -1;
		char *path = NULL;

		while (*p == ' ' || *p == '\t') {
			++p;
		}

		if (*p == '\0' || *p == '\n' || *p == '#') {
			continue;
		}

		if (sscanf(p, "%3071s %d", path_buf, &label) != 2) {
			continue;
		}

		if (label < 0 || label >= num_classes) {
			continue;
		}

		if (count == capacity) {
			int new_capacity = (capacity == 0) ? 64 : (capacity * 2);
			char **new_paths = realloc(paths, (size_t)new_capacity * sizeof(char *));
			int *new_labels = realloc(labels, (size_t)new_capacity * sizeof(int));
			if (!new_paths || !new_labels) {
				free(new_paths);
				free(new_labels);
				goto cleanup;
			}
			paths = new_paths;
			labels = new_labels;
			capacity = new_capacity;
		}

		path = image_strdup_local(path_buf);
		if (!path) goto cleanup;

		paths[count] = path;
		labels[count] = label;
		count++;
	}

	if (count <= 0) goto cleanup;

	paths_const = malloc((size_t)count * sizeof(const char *));
	if (!paths_const) goto cleanup;

	for (int i = 0; i < count; i++) {
		paths_const[i] = paths[i];
	}

	rc = image_dataset_load_pgm_labeled(paths_const,
									   labels,
									   count,
									   num_classes,
									   expected_width,
									   expected_height,
									   out_dataset);

cleanup:
	free(paths_const);

	if (paths) {
		for (int i = 0; i < count; i++) {
			free(paths[i]);
		}
	}
	free(paths);
	free(labels);

	fclose(f);
	return rc;
}

int image_dataset_make_tiny_digits(ImageDataset *out_dataset,
						   int samples_per_class,
						   int width,
						   int height,
						   unsigned int seed) {
	const int classes = 10;
	const int input_size = width * height;
	const int num_samples = classes * samples_per_class;
	float *inputs = NULL;
	float *targets = NULL;
	unsigned int rng;

	if (!out_dataset || samples_per_class <= 0 || width <= 4 || height <= 4) return -1;

	memset(out_dataset, 0, sizeof(*out_dataset));

	inputs = calloc((size_t)num_samples * (size_t)input_size, sizeof(float));
	targets = calloc((size_t)num_samples * (size_t)classes, sizeof(float));
	if (!inputs || !targets) {
		free(inputs);
		free(targets);
		return -1;
	}

	rng = (seed == 0u) ? 123456789u : seed;

	for (int digit = 0; digit < classes; digit++) {
		for (int s = 0; s < samples_per_class; s++) {
			int sample_idx = digit * samples_per_class + s;
			float *img = inputs + ((size_t)sample_idx * (size_t)input_size);
			int jitter_x = image_rand_range(&rng, -2, 2);
			int jitter_y = image_rand_range(&rng, -2, 2);
			int thickness = image_rand_range(&rng, 1, 3);

			image_draw_7seg_digit(img, width, height, digit, jitter_x, jitter_y, thickness);

			/* Add light background and sensor-like noise. */
			for (int i = 0; i < input_size; i++) {
				float base = img[i] * (0.75f + 0.25f * image_rand_unit(&rng));
				float noise = 0.06f * image_rand_unit(&rng);
				float v = base + noise;
				if (v > 1.0f) v = 1.0f;
				img[i] = v;
			}

			targets[(size_t)sample_idx * (size_t)classes + (size_t)digit] = 1.0f;
		}
	}

	out_dataset->inputs = inputs;
	out_dataset->targets = targets;
	out_dataset->num_samples = num_samples;
	out_dataset->image_width = width;
	out_dataset->image_height = height;
	out_dataset->image_channels = 1;
	out_dataset->input_size = input_size;
	out_dataset->target_size = classes;

	return 0;
}

void image_dataset_free(ImageDataset *dataset) {
	if (!dataset) return;

	free(dataset->inputs);
	free(dataset->targets);
	dataset->inputs = NULL;
	dataset->targets = NULL;
	dataset->num_samples = 0;
	dataset->image_width = 0;
	dataset->image_height = 0;
	dataset->image_channels = 0;
	dataset->input_size = 0;
	dataset->target_size = 0;
}

int image_argmax(const float *scores,
				 int count,
				 int *out_index) {
	int best_i = 0;
	float best_v;

	if (!scores || !out_index || count <= 0) return -1;

	best_v = scores[0];
	for (int i = 1; i < count; i++) {
		if (scores[i] > best_v) {
			best_v = scores[i];
			best_i = i;
		}
	}

	*out_index = best_i;
	return 0;
}

int image_dataset_get_label(const ImageDataset *dataset,
					int sample_index,
					int *out_label) {
	const float *row;

	if (!dataset || !out_label || !dataset->targets ||
		sample_index < 0 || sample_index >= dataset->num_samples ||
		dataset->target_size <= 0) {
		return -1;
	}

	row = dataset->targets + ((size_t)sample_index * (size_t)dataset->target_size);
	return image_argmax(row, dataset->target_size, out_label);
}

int sequential_model_train_image_dataset(SequentialModel *model,
										 const ImageDataset *dataset,
										 int epochs,
										 int batch_size,
										 float *final_loss_out) {
	if (!model || !dataset || !dataset->inputs || !dataset->targets) return -1;
	if (dataset->num_samples <= 0 || dataset->input_size <= 0 || dataset->target_size <= 0) return -1;

	return sequential_model_train(model,
								  dataset->inputs,
								  dataset->targets,
								  dataset->num_samples,
								  dataset->input_size,
								  dataset->target_size,
								  epochs,
								  batch_size,
								  final_loss_out);
}

int sequential_model_predict_pgm(SequentialModel *model,
								 const char *file_path,
								 int expected_width,
								 int expected_height,
								 float *output) {
	float *pixels = NULL;
	int width = 0;
	int height = 0;
	int rc;

	if (!model || !file_path || !output || expected_width <= 0 || expected_height <= 0) return -1;

	if (image_load_pgm(file_path, &pixels, &width, &height) != 0) return -1;

	if (width != expected_width || height != expected_height) {
		free(pixels);
		return -1;
	}

	rc = sequential_model_predict(model, pixels, output);
	free(pixels);
	return rc;
}
