# hfendpoint llamacpp-embeddings
Do not use it, just a draft

## Prepare

You'll need the following:

 - [llama.cpp](https://github.com/ggml-org/llama.cpp)
 - [msgpack-c](https://github.com/msgpack/msgpack-c/tree/c_master)
 - [hfendpoint-draft](https://github.com/angt/hfendpoint-draft/releases/tag/v0.1.0)

## Build

    make

## Run

    export HFENDPOINT_GGUF=model.gguf
    hfendpoint -- ./worker
