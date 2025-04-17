#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <llama.h>
#include <msgpack.h>

#define BUFFER_SIZE 4096U
#define MAX_SIZE    128U

#define LOG(format, ...)     fprintf(stdout, format "\n", ##__VA_ARGS__)
#define LOG_ERR(format, ...) fprintf(stderr, format "\n", ##__VA_ARGS__)

static struct {
    char name[128];
    const char *gguf;
    int fd;
    struct llama_model *model;
    struct llama_context *ctx;
    int n_ctx;
    int n_batch;
    int n_threads;
    llama_token add_bos;
    llama_token *tokens;
    const struct llama_vocab *vocab;
    int32_t n_embd;
    enum llama_pooling_type pool_type;
    int use_encode;
} worker = {
    .fd = -1,
};

struct embeddings {
    size_t size;
    uint32_t n_tokens;
    float *list[MAX_SIZE];
};

static void
pack_string(msgpack_packer *packer, const char *str)
{
    size_t len = strlen(str);
    msgpack_pack_str(packer, len);
    msgpack_pack_str_body(packer, str, len);
}

static int
key_matches(msgpack_object key, const char *str)
{
    return key.type == MSGPACK_OBJECT_STR &&
           key.via.str.size == strlen(str) &&
           !memcmp(key.via.str.ptr, str, key.via.str.size);
}

static void
send_buffer(msgpack_sbuffer *buffer)
{
    ssize_t written = 0;
    size_t remaining = buffer->size;

    while (remaining) {
        ssize_t ret = write(worker.fd, buffer->data + written, remaining);

        if (ret == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd = {
                    .fd = worker.fd,
                    .events = POLLOUT
                };
                poll(&pfd, 1, -1);
                continue;
            }
            LOG_ERR("Failed write() (errno %d)", errno);
            break;
        }
        written += ret;
        remaining -= ret;
    }
}

static void
worker_error(uint64_t id, const char *fmt, ...)
{
    char error[1024];
    va_list ap;

    va_start(ap, fmt);
    int ret = vsnprintf(error, sizeof(error), fmt, ap);
    va_end(ap);

    if (ret <= 0 || (size_t)ret >= sizeof(error)) {
        LOG_ERR("Failed vsnprintf()");
        return;
    }
    msgpack_sbuffer buffer;
    msgpack_sbuffer_init(&buffer);
    msgpack_packer packer;
    msgpack_packer_init(&packer, &buffer, msgpack_sbuffer_write);

    msgpack_pack_map(&packer, 2);
    pack_string(&packer, "id");
    msgpack_pack_uint64(&packer, id);
    pack_string(&packer, "error");
    pack_string(&packer, error);

    send_buffer(&buffer);
    msgpack_sbuffer_destroy(&buffer);
}

static void
reply(uint64_t id, struct embeddings *embds)
{
    msgpack_sbuffer buffer;
    msgpack_sbuffer_init(&buffer);
    msgpack_packer packer;
    msgpack_packer_init(&packer, &buffer, msgpack_sbuffer_write);

    msgpack_pack_map(&packer, 2);
    pack_string(&packer, "id");
    msgpack_pack_uint64(&packer, id);
    pack_string(&packer, "data");

    msgpack_pack_map(&packer, 3);
    pack_string(&packer, "model");
    pack_string(&packer, worker.name);
    pack_string(&packer, "usage");
    msgpack_pack_map(&packer, 2);
    pack_string(&packer, "prompt_tokens");
    msgpack_pack_uint32(&packer, embds->n_tokens);
    pack_string(&packer, "total_tokens");
    msgpack_pack_uint32(&packer, embds->n_tokens);
    pack_string(&packer, "embeddings");
    msgpack_pack_array(&packer, embds->size);

    for (size_t i = 0; i < embds->size; i++) {
        float *embd = embds->list[i];
        if (embd) {
            msgpack_pack_array(&packer, worker.n_embd);
            for (int j = 0; j < worker.n_embd; j++) {
                msgpack_pack_float(&packer, embd[j]);
            }
        } else {
            msgpack_pack_nil(&packer);
        }
    }
    send_buffer(&buffer);
    msgpack_sbuffer_destroy(&buffer);
}

static void
handle_request(uint64_t id, msgpack_object input)
{
    if (input.type == MSGPACK_OBJECT_NIL)
        return;

    struct embeddings embds = {
        .size = input.via.array.size
    };
    int seqs[MAX_SIZE] = {0};
    struct llama_batch batch = llama_batch_init(worker.n_batch, 0, 1);

    if (!batch.token) {
        LOG_ERR("Failed llama_batch_init() (req %" PRIu64 ")", id);
        return;
    }
    for (size_t i = 0; i < embds.size; i++) {
        llama_kv_self_seq_rm(worker.ctx, i, -1, -1);
    }
    int pos = 0;

    for (size_t i = 0; i < embds.size; i++) {
        msgpack_object_str text = input.via.array.ptr[i].via.str;

        int n_tokens = llama_tokenize(worker.vocab, text.ptr, text.size,
                                      worker.tokens, worker.n_ctx,
                                      worker.add_bos, 0);
        if (n_tokens <= 0) {
            LOG_ERR("Couldn't tokenize (req %" PRIu64 ", seq %zu)", id, i);
            continue;
        }
        if (pos + n_tokens > worker.n_batch) {
            LOG_ERR("Couldn't fit input in batch (req %" PRIu64 ", seq %zu)", id, i);
            worker_error(id,
                    "Batch size limit exceeded (max %d tokens). "
                    "Input sequence %zu requires %d tokens, but only %d space remaining in batch.",
                    worker.n_batch, i, n_tokens, worker.n_batch - pos);
            llama_batch_free(batch);
            return;
        }
        for (int j = 0; j < n_tokens; j++) {
            batch.token[pos] = worker.tokens[j];
            batch.pos[pos] = j;
            batch.n_seq_id[pos] = 1;
            batch.seq_id[pos][0] = i;
            batch.logits[pos] = 0;
            pos++;
        }
        if (n_tokens > 0)
            batch.logits[pos - 1] = 1;

        seqs[i] = 1;
    }
    batch.n_tokens = pos;
    embds.n_tokens = pos;

    if (pos) {
        int rc = worker.use_encode ? llama_encode(worker.ctx, batch)
                                   : llama_decode(worker.ctx, batch);
        if (rc) {
            LOG_ERR("Failed llama_%s() (req %" PRIu64 ", rc %d)",
                    worker.use_encode ? "encode" : "decode", id, rc);
        } else {
            for (size_t i = 0; i < embds.size; i++) {
                if (seqs[i])
                    embds.list[i] = llama_get_embeddings_seq(worker.ctx, i);
            }
        }
    }
    reply(id, &embds);
    llama_batch_free(batch);
}

static msgpack_object
check_input(msgpack_object obj)
{
    msgpack_object nil = { .type = MSGPACK_OBJECT_NIL };

    if (obj.type != MSGPACK_OBJECT_ARRAY)
        return nil;

    size_t size = obj.via.array.size;

    if (size == 0 || size > MAX_SIZE)
        return nil;

    for (size_t i = 0; i < size; i++) {
        if (obj.via.array.ptr[i].type != MSGPACK_OBJECT_STR)
            return nil;
    }
    return obj;
}

static msgpack_object
extract_input(msgpack_object obj)
{
    msgpack_object nil = { .type = MSGPACK_OBJECT_NIL };

    if (obj.type != MSGPACK_OBJECT_MAP)
        return nil;

    msgpack_object_kv *ptr = obj.via.map.ptr;
    msgpack_object_kv *end = ptr + obj.via.map.size;

    for (; ptr < end; ptr++) {
        if (key_matches(ptr->key, "input"))
            return check_input(ptr->val);
    }
    return nil;
}

static void
dispatch_request(msgpack_object obj)
{
    if (obj.type != MSGPACK_OBJECT_MAP) {
        LOG_ERR("Skipping malformed request (type %d)", obj.type);
        return;
    }
    uint64_t id = 0;
    msgpack_object data = { .type = MSGPACK_OBJECT_NIL };

    int found_id = 0;
    int found_name = 0;
    int found_data = 0;

    msgpack_object_kv *ptr = obj.via.map.ptr;
    msgpack_object_kv *end = ptr + obj.via.map.size;

    for (; ptr < end; ptr++) {
        if (key_matches(ptr->key, "id")) {
            if (ptr->val.type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
                id = ptr->val.via.u64;
                found_id = 1;
            }
        } else if (key_matches(ptr->key, "name")) {
            found_name = 1;
            if (!key_matches(ptr->val, "embeddings")) {
                LOG("Skipping non embeddings request");
                return;
            }
        } else if (key_matches(ptr->key, "data")) {
            data = ptr->val;
            found_data = 1;
        }
        if (found_id && found_name && found_data)
            break;
    }
    if (!found_id || !found_name || !found_data) {
        LOG_ERR("Skipping malformed request");
        return;
    }
    msgpack_object input = extract_input(data);
    handle_request(id, input);
}

static ssize_t
read_data(int fd, void *buffer, size_t size)
{
    ssize_t bytes = read(fd, buffer, size);

    if (bytes < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
            return 0;

        LOG_ERR("Failed read() (fd %d, errno %d)", fd, errno);
        return -1;
    }
    if (bytes == 0)
        return -2;

    return bytes;
}

static int
process_data(msgpack_unpacker *unpacker, int fd)
{
    if (!msgpack_unpacker_reserve_buffer(unpacker, BUFFER_SIZE)) {
        LOG_ERR("Failed msgpack_unpacker_reserve_buffer()");
        return -1;
    }
    ssize_t bytes = read_data(fd, msgpack_unpacker_buffer(unpacker), BUFFER_SIZE);

    if (bytes <= 0)
        return (int)bytes;

    msgpack_unpacker_buffer_consumed(unpacker, bytes);
    msgpack_unpacked result;

    for (;;) {
        msgpack_unpacked_init(&result);

        int ret = msgpack_unpacker_next(unpacker, &result);

        if (ret == MSGPACK_UNPACK_SUCCESS)
            dispatch_request(result.data);

        msgpack_unpacked_destroy(&result);

        if (ret == MSGPACK_UNPACK_CONTINUE)
            break;

        if (ret == MSGPACK_UNPACK_PARSE_ERROR) {
            LOG_ERR("Msgpack parse error. Resetting unpacker.");
            msgpack_unpacker_reset(unpacker);
            break;
        }
        if (ret != MSGPACK_UNPACK_SUCCESS) {
            LOG_ERR("Unexpected unpacker state: %d", ret);
            return -1;
        }
    }
    return 0;
}

static int
parse_int(const char *val, int def)
{
    if (!val)
        return def;

    char *end;
    errno = 0;
    long x = strtol(val, &end, 10);

    if (errno || end == val || *end != '\0' || x < 0 || x > INT_MAX) {
        LOG_ERR("Invalid integer in `%s`", val);
        return def;
    }
    return (int)x;
}

static int
set_nonblock(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);

    if (flags == -1) {
        LOG_ERR("Couldn't get fd flags (fd %d, errno %d)", fd, errno);
        return 1;
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
        LOG_ERR("Couldn't set fd flags (fd %d, errno %d)", fd, errno);
        return 1;
    }
    return 0;
}

static int
setup_llama(void)
{
    struct llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = 1;
    worker.model = llama_model_load_from_file(worker.gguf, model_params);

    if (!worker.model) {
        LOG_ERR("Couldn't load model `%s`", worker.gguf);
        return 1;
    }
    llama_model_desc(worker.model, worker.name, sizeof(worker.name) - 1);

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = worker.n_ctx;
    ctx_params.n_batch         = worker.n_batch;
    ctx_params.n_threads       = worker.n_threads;
    ctx_params.n_threads_batch = worker.n_threads;
    ctx_params.embeddings      = 1;
    ctx_params.pooling_type    = LLAMA_POOLING_TYPE_MEAN;
    worker.ctx = llama_init_from_model(worker.model, ctx_params);

    if (!worker.ctx) {
        LOG_ERR("Couldn't create llama context");
        return 1;
    }
    worker.n_ctx = llama_n_ctx(worker.ctx);

    if (worker.n_ctx <= 0) {
        LOG_ERR("Invalid n_ctx: %d", worker.n_ctx);
        return 1;
    }
    worker.n_batch = llama_n_batch(worker.ctx);

    if (worker.n_batch <= 0) {
        LOG_ERR("Invalid n_batch: %d", worker.n_batch);
        return 1;
    }
    worker.vocab = llama_model_get_vocab(worker.model);

    if (!worker.vocab) {
        LOG_ERR("Couldn't get vocab");
        return 1;
    }
    worker.n_embd = llama_model_n_embd(worker.model);

    if (worker.n_embd <= 0) {
        LOG_ERR("Invalid n_embd: %d", worker.n_embd);
        return 1;
    }
    worker.pool_type = llama_pooling_type(worker.ctx);
    worker.add_bos = llama_vocab_get_add_bos(worker.vocab);

    int has_encoder = llama_model_has_encoder(worker.model);
    int has_decoder = llama_model_has_decoder(worker.model);

    if (has_encoder && has_decoder) {
        LOG_ERR("Encoder-decoder models not supported");
        return 1;
    }
    worker.use_encode = has_encoder;

    worker.tokens = malloc(worker.n_ctx * sizeof(llama_token));

    if (!worker.tokens) {
        LOG_ERR("Couldn't alloc %d tokens", worker.n_ctx);
        return 1;
    }
    return 0;
}

static int
get_n_threads(void)
{
#if __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);

    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask)) {
        LOG_ERR("Failed sched_getaffinity() (errno %d)", errno);
        return 1;
    }
    int count = 0;

    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &mask))
            count++;
    }
    if (count <= 0) {
        LOG_ERR("Couldn't find the number of threads");
        return 1;
    }
    LOG("Detected %d threads", count);
    return count;
#else
    return 1;
#endif
}

static int
setup(int argc, char **argv)
{
    llama_backend_init();

    worker.fd = parse_int(getenv("HFENDPOINT_FD"), -1);

    if (worker.fd == -1) {
        LOG_ERR("Run this binary from hfendpoint");
        return 1;
    }
    if (set_nonblock(worker.fd))
        return 1;

    worker.n_threads = parse_int(getenv("HFENDPOINT_THREADS"), -1);

    if (worker.n_threads <= 0)
        worker.n_threads = get_n_threads();

    worker.gguf = getenv("HFENDPOINT_GGUF");

    if (!worker.gguf) {
        LOG_ERR("HFENDPOINT_GGUF is required");
        return 1;
    }
    worker.n_ctx   = parse_int(getenv("HFENDPOINT_CTX"), 0);
    worker.n_batch = parse_int(getenv("HFENDPOINT_BATCH"), 0);

    return setup_llama();
}

static int
cleanup(int exit_code)
{
    if (worker.tokens)
        free(worker.tokens);

    if (worker.ctx)
        llama_free(worker.ctx);

    if (worker.model)
        llama_model_free(worker.model);

    llama_backend_free();
    return exit_code;
}

int
main(int argc, char **argv)
{
    if (setup(argc, argv))
        return cleanup(EXIT_FAILURE);

    msgpack_unpacker unpacker;

    if (!msgpack_unpacker_init(&unpacker, BUFFER_SIZE)) {
        LOG_ERR("Failed msgpack_unpacker_init()");
        return cleanup(EXIT_FAILURE);
    }
    struct pollfd pfd = {
        .fd = worker.fd,
        .events = POLLIN
    };
    int exit_code = EXIT_FAILURE;

    while (1) {
        int ret = poll(&pfd, 1, -1);

        if (ret == -1) {
            if (errno == EINTR)
                continue;

            LOG_ERR("Failed poll() (errno %d)", errno);
            break;
        }
        if (pfd.revents & (POLLERR | POLLNVAL)) {
            LOG_ERR("Channel error (fd %d, revents 0x%x)", pfd.fd, pfd.revents);
            break;
        }
        if (pfd.revents & (POLLIN | POLLHUP)) {
            int ret = process_data(&unpacker, pfd.fd);

            if (ret == -1)
                break;

            if (ret == -2) {
                process_data(&unpacker, pfd.fd);
                exit_code = EXIT_SUCCESS;
                break;
            }
        }
    }
    msgpack_unpacker_destroy(&unpacker);
    return cleanup(exit_code);
}
