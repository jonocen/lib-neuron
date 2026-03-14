CC := gcc
CFLAGS := -Wall -Wextra -Wpedantic -std=c11 -I../include
LDFLAGS := -L.. -Wl,-rpath,'$$ORIGIN/..'
LDLIBS := -lneuron -lm
SOURCES := $(wildcard *.c)
TARGETS := $(patsubst %.c,%,$(SOURCES))

.PHONY: all lib clean

all: lib $(TARGETS)

lib:
	$(MAKE) -C ..

%: %.c
	$(CC) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

clean:
	rm -f $(TARGETS)
	$(MAKE) -C .. clean
