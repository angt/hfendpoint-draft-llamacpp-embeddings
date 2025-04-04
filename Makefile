CFLAGS = -O3 -Wall -march=native -mcpu=native

CFLAGS  += $(shell pkg-config --cflags llama)
LDFLAGS += $(shell pkg-config --libs-only-L llama)
LDLIBS  += $(shell pkg-config --libs-only-l llama)

CFLAGS  += $(shell pkg-config --cflags msgpack-c)
LDFLAGS += $(shell pkg-config --libs-only-L msgpack-c)
LDLIBS  += $(shell pkg-config --libs-only-l msgpack-c)

worker:
