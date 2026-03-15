CC      := gcc
AR      := ar
CFLAGS  := -Wall -Wextra -Wpedantic -std=c11 -Iinclude
RELEASE := -O2
DEBUG   := -O0 -g
PIC     := -fPIC

SRC := src/matrixcalculation.c \
	src/layers.c \
       src/lossfunctions.c \
       src/optimizers.c \
	src/image_processing.c \
	src/models_internal.c \
	src/models_core.c \
	src/models_train.c \
	src/models_state.c \
	src/models_legacy.c

OBJ := matrixcalculation.o layers.o lossfunctions.o optimizers.o image_processing.o models_internal.o models_core.o models_train.o models_state.o models_legacy.o
PIC_OBJ := matrixcalculation.pic.o layers.pic.o lossfunctions.pic.o optimizers.pic.o image_processing.pic.o models_internal.pic.o models_core.pic.o models_train.pic.o models_state.pic.o models_legacy.pic.o
LIB := libneuron.a
SHARED_LIB := libneuron.so
EXAMPLE_BINARIES := $(patsubst examples/%.c,examples/%,$(wildcard examples/*.c))

.PHONY: all lib static shared debug clean examples sequential_xor_plugin Other_Exaple simple_compact mnist_tiny_pgm

all: CFLAGS += $(RELEASE)
all: lib

lib: static shared

static: $(LIB)

shared: $(SHARED_LIB)

debug: CFLAGS += $(DEBUG)
debug: clean $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

$(SHARED_LIB): $(PIC_OBJ)
	$(CC) -shared -o $@ $^ -lm

matrixcalculation.o: src/matrixcalculation.c include/matrixcalculation.h
	$(CC) $(CFLAGS) -c $< -o $@

lossfunctions.o: src/lossfunctions.c include/lossfunctions.h
	$(CC) $(CFLAGS) -c $< -o $@

layers.o: src/layers.c include/layers.h include/matrixcalculation.h
	$(CC) $(CFLAGS) -c $< -o $@

optimizers.o: src/optimizers.c include/optimizers.h
	$(CC) $(CFLAGS) -c $< -o $@

image_processing.o: src/image_processing.c include/image_processing.h include/models.h
	$(CC) $(CFLAGS) -c $< -o $@

models_internal.o: src/models_internal.c include/models_internal.h include/models.h include/matrixcalculation.h include/lossfunctions.h include/optimizers.h
	$(CC) $(CFLAGS) -c $< -o $@

models_core.o: src/models_core.c include/models_internal.h include/models.h include/models_core.h
	$(CC) $(CFLAGS) -c $< -o $@

models_train.o: src/models_train.c include/models_internal.h include/models.h include/models_training.h
	$(CC) $(CFLAGS) -c $< -o $@

models_state.o: src/models_state.c include/models_internal.h include/models.h include/models_training.h
	$(CC) $(CFLAGS) -c $< -o $@

models_legacy.o: src/models_legacy.c include/models_internal.h include/models.h include/models_legacy.h
	$(CC) $(CFLAGS) -c $< -o $@

matrixcalculation.pic.o: src/matrixcalculation.c include/matrixcalculation.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

lossfunctions.pic.o: src/lossfunctions.c include/lossfunctions.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

layers.pic.o: src/layers.c include/layers.h include/matrixcalculation.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

optimizers.pic.o: src/optimizers.c include/optimizers.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

image_processing.pic.o: src/image_processing.c include/image_processing.h include/models.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

models_internal.pic.o: src/models_internal.c include/models_internal.h include/models.h include/matrixcalculation.h include/lossfunctions.h include/optimizers.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

models_core.pic.o: src/models_core.c include/models_internal.h include/models.h include/models_core.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

models_train.pic.o: src/models_train.c include/models_internal.h include/models.h include/models_training.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

models_state.pic.o: src/models_state.c include/models_internal.h include/models.h include/models_training.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

models_legacy.pic.o: src/models_legacy.c include/models_internal.h include/models.h include/models_legacy.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

clean:
	rm -f $(OBJ) $(PIC_OBJ) $(LIB) $(SHARED_LIB) $(EXAMPLE_BINARIES)

examples: lib
	$(MAKE) -C examples

sequential_xor_plugin: lib
	$(MAKE) -C examples sequential_xor_plugin

Other_Exaple: lib
	$(MAKE) -C examples Other_Exaple

simple_compact: lib
	$(MAKE) -C examples simple_compact

mnist_tiny_pgm: lib
	$(MAKE) -C examples mnist_tiny_pgm
