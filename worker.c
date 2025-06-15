#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <poll.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <llama.h>
#include <msgpack.h>

#define BUFFER_SIZE  (1024U * 1024U)

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
    const struct llama_vocab *vocab;
    int n_embd;
    enum llama_pooling_type pooling;
    struct llama_batch batch;
    llama_token *tokens;
    struct {
        msgpack_sbuffer buffer;
        msgpack_packer packer;
        size_t written;
        size_t checkpoint;
    } send;
    struct {
        int next;
    } seq;
} worker = {
    .fd = -1,
};

static void
commit_buffer(void)
{
    worker.send.checkpoint = worker.send.buffer.size;
}

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
worker_error(uint64_t id, const char *fmt, ...)
{
    worker.send.buffer.size = worker.send.checkpoint;

    char error[1024];
    va_list ap;

    va_start(ap, fmt);
    int ret = vsnprintf(error, sizeof(error), fmt, ap);
    va_end(ap);

    if (ret <= 0 || (size_t)ret >= sizeof(error)) {
        LOG_ERR("Failed vsnprintf()");
        return;
    }
    msgpack_packer *packer = &worker.send.packer;

    msgpack_pack_map(packer, 2);
    pack_string(packer, "id");
    msgpack_pack_uint64(packer, id);
    pack_string(packer, "error");
    pack_string(packer, error);

    commit_buffer();
}

static float
l2_norm(const float *embd, int n)
{
    float sum_sq = 0.0f;

    for (int i = 0; i < n; i++) {
        sum_sq += embd[i] * embd[i];
    }
    return sqrtf(sum_sq);
}

static void
handle_tokenize(uint64_t id, msgpack_object input)
{
    if (input.type != MSGPACK_OBJECT_ARRAY)
        return worker_error(id, "Expected input to be an array");

    size_t size = input.via.array.size;

    for (size_t i = 0; i < size; i++) {
        if (input.via.array.ptr[i].type != MSGPACK_OBJECT_STR)
            return worker_error(id, "input[%zu] is not a string", i);
    }
    msgpack_packer *packer = &worker.send.packer;

    msgpack_pack_map(packer, 2);
    pack_string(packer, "id");
    msgpack_pack_uint64(packer, id);
    pack_string(packer, "data");
    msgpack_pack_map(packer, 1);
    pack_string(packer, "tokens");
    msgpack_pack_array(packer, size);

    for (size_t i = 0; i < size; i++) {
        msgpack_object_str text = input.via.array.ptr[i].via.str;

        int n_tokens = llama_tokenize(worker.vocab, text.ptr, text.size,
                                      worker.tokens, worker.n_ctx, 1, 1);
        if (n_tokens <= 0)
            return worker_error(id,
                    "Token size limit exceeded (max %d tokens). "
                    "Input sequence %zu requires %d tokens.",
                    worker.n_ctx, i, -n_tokens);

        msgpack_pack_array(packer, n_tokens);

        for (int j = 0; j < n_tokens; j++)
            msgpack_pack_uint32(packer, worker.tokens[j]);
    }
    if (size == 0)
        msgpack_pack_nil(packer);

    commit_buffer();
}

static void
batch_add(llama_token token, llama_pos pos, llama_seq_id seq_id)
{
    size_t n = worker.batch.n_tokens;
    worker.batch.token[n] = token;
    worker.batch.pos[n] = pos;
    worker.batch.n_seq_id[n] = 1;
    worker.batch.seq_id[n][0] = seq_id;
    worker.batch.logits[n] = 1;
    worker.batch.n_tokens++;
}

static void
run_batch(void)
{
    if (!worker.batch.n_tokens)
        return;

    llama_memory_clear(llama_get_memory(worker.ctx), true);

    int rc = llama_decode(worker.ctx, worker.batch);

    msgpack_packer *packer = &worker.send.packer;
    int last_seq_id = -1;

    for (int i = 0; i < worker.batch.n_tokens; i++) {
        int seq_id = worker.batch.seq_id[i][0];

        if (seq_id == last_seq_id)
            continue;

        last_seq_id = seq_id;

        float *embd = rc ? NULL : llama_get_embeddings_seq(worker.ctx, seq_id);

        if (!embd) {
            msgpack_pack_nil(packer);
            commit_buffer();
            continue;
        }
        float inorm = 1.0 / l2_norm(embd, worker.n_embd);

        msgpack_pack_array(packer, worker.n_embd);

        for (int j = 0; j < worker.n_embd; j++)
            msgpack_pack_float(packer, inorm * embd[j]);

        commit_buffer();
    }
    worker.batch.n_tokens = 0;
    worker.seq.next = 0;
}

static void
handle_embeddings(uint64_t id, msgpack_object input)
{
    if (input.type != MSGPACK_OBJECT_ARRAY)
        return worker_error(id, "Expected input to be an array");

    size_t size = input.via.array.size;

    for (size_t i = 0; i < size; ++i) {
        if (input.via.array.ptr[i].type != MSGPACK_OBJECT_ARRAY)
            return worker_error(id, "input[%zu] is not an array", i);

        size_t n_tokens = input.via.array.ptr[i].via.array.size;

        if (n_tokens == 0)
            return worker_error(id, "input[%zu] is empty", i);

        if (n_tokens > worker.n_batch)
            return worker_error(id, "input[%zu] exceeds %d", i, worker.n_batch);
    }
    msgpack_packer *packer = &worker.send.packer;

    msgpack_pack_map(packer, 2);
    pack_string(packer, "id");
    msgpack_pack_uint64(packer, id);
    pack_string(packer, "data");
    msgpack_pack_map(packer, 1);
    pack_string(packer, "embeddings");
    msgpack_pack_array(packer, size);

    llama_memory_clear(llama_get_memory(worker.ctx), true);

    for (int j = 0; j < size; j++) {
        msgpack_object_array seq = input.via.array.ptr[j].via.array;

        if (worker.batch.n_tokens + seq.size > worker.n_batch)
            run_batch();

        int seq_id = worker.seq.next;
        worker.seq.next++;

        for (int i = 0; i < seq.size; i++)
            batch_add(seq.ptr[i].via.u64, i, seq_id);
    }
    run_batch();
}

static msgpack_object
extract_input(msgpack_object obj)
{
    msgpack_object nil = { .type = MSGPACK_OBJECT_NIL };

    if (obj.type != MSGPACK_OBJECT_MAP)
        return nil;

    if (obj.via.map.size != 1)
        return nil;

    msgpack_object_kv *kv = obj.via.map.ptr;

    if (!key_matches(kv->key, "input"))
        return nil;

    return kv->val;
}

static void
dispatch_request(msgpack_object obj)
{
    if (obj.type != MSGPACK_OBJECT_MAP) {
        LOG_ERR("Skipping malformed request (type %d)", obj.type);
        return;
    }
    uint64_t id = 0;
    int found_id = 0;
    int found = 0;

    msgpack_object name = { .type = MSGPACK_OBJECT_NIL };
    msgpack_object data = { .type = MSGPACK_OBJECT_NIL };

    msgpack_object_kv *ptr = obj.via.map.ptr;
    msgpack_object_kv *end = ptr + obj.via.map.size;

    for (; ptr < end; ptr++) {
        if (key_matches(ptr->key, "id")) {
            if (ptr->val.type == MSGPACK_OBJECT_POSITIVE_INTEGER) {
                id = ptr->val.via.u64;
                found_id = 1;
            }
        } else if (key_matches(ptr->key, "name")) {
            name = ptr->val;
        } else if (key_matches(ptr->key, "data")) {
            data = ptr->val;
        }
        if (found_id &&
            name.type == MSGPACK_OBJECT_STR &&
            data.type == MSGPACK_OBJECT_MAP) {
            found = 1;
            break;
        }
    }
    if (!found) {
        LOG_ERR("Skipping malformed request");
        return;
    }
    if (key_matches(name, "tokenize"))
        return handle_tokenize(id, extract_input(data));

    if (key_matches(name, "embeddings"))
        return handle_embeddings(id, extract_input(data));

    LOG("Skipping unknown request");
}

static int
process_data(msgpack_unpacker *unpacker, msgpack_unpacked *unpacked, int fd)
{
    if (!msgpack_unpacker_reserve_buffer(unpacker, BUFFER_SIZE)) {
        LOG_ERR("Failed msgpack_unpacker_reserve_buffer()");
        return -1;
    }
    ssize_t bytes = read(fd, msgpack_unpacker_buffer(unpacker), BUFFER_SIZE);

    if (bytes < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
            return 0;

        LOG_ERR("Failed read() (fd %d, errno %d)", fd, errno);
        return -1;
    }
    if (bytes == 0)
        return -2;

    msgpack_unpacker_buffer_consumed(unpacker, bytes);
    int ret;

    for (;;) {
        ret = msgpack_unpacker_next(unpacker, unpacked);

        if (ret != MSGPACK_UNPACK_SUCCESS)
            break;

        dispatch_request(unpacked->data);
    }
    if (ret == MSGPACK_UNPACK_CONTINUE)
        return 0;

    if (ret == MSGPACK_UNPACK_PARSE_ERROR) {
        LOG_ERR("Msgpack parse error. Resetting unpacker.");
        msgpack_unpacker_reset(unpacker);
        msgpack_unpacked_destroy(unpacked);
        return 0;
    }
    LOG_ERR("Unexpected unpacker state: %d", ret);
    return -1;
}

static void
send_buffer(void)
{
    size_t written = worker.send.written;

    if (worker.send.buffer.size <= written)
        return;

    size_t size = worker.send.buffer.size - written;
    const char *data = worker.send.buffer.data + written;
    ssize_t ret = write(worker.fd, data, size);

    if (ret == -1) {
        if (errno != EAGAIN && errno != EWOULDBLOCK)
            LOG_ERR("Failed write() (errno %d)", errno);
        // TODO: LATER
        return;
    }
    worker.send.written += ret;

    if (worker.send.written >= worker.send.buffer.size) {
        msgpack_sbuffer_clear(&worker.send.buffer);
        worker.send.written = 0;
        worker.send.checkpoint = 0;
    }
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
setup_llama_context(void)
{
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx           = worker.n_ctx;
    params.n_batch         = worker.n_batch;
    params.n_ubatch        = worker.n_batch;
    params.n_threads       = worker.n_threads;
    params.n_threads_batch = worker.n_threads;
    params.embeddings      = 1;
    params.pooling_type    = worker.pooling;
    worker.ctx = llama_init_from_model(worker.model, params);

    if (!worker.ctx) {
        LOG_ERR("Couldn't create llama context");
        return 1;
    }
    worker.n_ctx = llama_n_ctx(worker.ctx);

    if (worker.n_ctx <= 0 || worker.n_ctx != params.n_ctx) {
        LOG_ERR("Invalid n_ctx: %d (params: %d)", worker.n_ctx, params.n_ctx);
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
    int has_encoder = llama_model_has_encoder(worker.model);
    int has_decoder = llama_model_has_decoder(worker.model);

    if (has_encoder && has_decoder) {
        LOG_ERR("Encoder-decoder models not supported");
        return 1;
    }
    worker.tokens = malloc(worker.n_ctx * sizeof(llama_token));

    if (!worker.tokens) {
        LOG_ERR("Couldn't alloc %d tokens", worker.n_ctx);
        return 1;
    }
    worker.batch = llama_batch_init(worker.n_batch, 0, 1);

    if (!worker.batch.token) {
        LOG_ERR("Failed llama_batch_init()");
        return 1;
    }
    return 0;
}

static int
setup_llama_model(void)
{
    struct llama_model_params params = llama_model_default_params();
    params.use_mmap = 1;
    worker.model = llama_model_load_from_file(worker.gguf, params);

    if (!worker.model) {
        LOG_ERR("Couldn't load model `%s`", worker.gguf);
        return 1;
    }
    llama_model_desc(worker.model, worker.name, sizeof(worker.name) - 1);

    int32_t n_ctx = llama_model_n_ctx_train(worker.model);

    worker.n_ctx = parse_int(getenv("HFENDPOINT_N_CTX"), n_ctx);
    worker.n_batch = parse_int(getenv("HFENDPOINT_N_BATCH"), n_ctx);

    const char *pooling = getenv("HFENDPOINT_POOLING");

    if (pooling) {
        if (!strcmp(pooling, "MEAN")) {
            worker.pooling = LLAMA_POOLING_TYPE_MEAN;
        } else if (!strcmp(pooling, "CLS")) {
            worker.pooling = LLAMA_POOLING_TYPE_CLS;
        } else if (!strcmp(pooling, "LAST")) {
            worker.pooling = LLAMA_POOLING_TYPE_LAST;
        } else {
            LOG_ERR("Unsupported pooling: %s", pooling);
            return 1;
        }
    } else {
        worker.pooling = LLAMA_POOLING_TYPE_MEAN;
    }
    return setup_llama_context();
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

static void
log_callback(enum ggml_log_level level, const char *text, void *user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fprintf(stderr, "%s", text);
    } else if (level != GGML_LOG_LEVEL_DEBUG) {
        fprintf(stdout, "%s", text);
    }
}

static int
setup(int argc, char **argv)
{
    llama_backend_init();
    llama_log_set(log_callback, NULL);

    worker.fd = parse_int(getenv("HFENDPOINT_FD"), -1);

    if (worker.fd == -1) {
        LOG_ERR("Run this binary from hfendpoint");
        return 1;
    }
    worker.n_threads = parse_int(getenv("HFENDPOINT_THREADS"), -1);

    if (worker.n_threads <= 0)
        worker.n_threads = get_n_threads();

    worker.gguf = getenv("HFENDPOINT_GGUF");

    if (!worker.gguf) {
        LOG_ERR("HFENDPOINT_GGUF is required");
        return 1;
    }
    worker.send.buffer.alloc = 2 * BUFFER_SIZE;
    worker.send.buffer.data = malloc(worker.send.buffer.alloc);

    if (!worker.send.buffer.data) {
        LOG_ERR("Failed malloc()");
        return 1;
    }
    msgpack_packer_init(&worker.send.packer, &worker.send.buffer, msgpack_sbuffer_write);

    return setup_llama_model();
}

static int
cleanup(int exit_code)
{
    msgpack_sbuffer_destroy(&worker.send.buffer);

    if (worker.batch.token)
        llama_batch_free(worker.batch);

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

    msgpack_unpacker unpacker = {0};
    msgpack_unpacked unpacked = {0};

    if (!msgpack_unpacker_init(&unpacker, BUFFER_SIZE)) {
        LOG_ERR("Failed msgpack_unpacker_init()");
        return cleanup(EXIT_FAILURE);
    }
    int exit_code = EXIT_FAILURE;

    while (1) {
        struct pollfd pfd = {
            .fd = worker.fd,
        };
        if (worker.send.buffer.alloc - worker.send.buffer.size >= BUFFER_SIZE)
            pfd.events |= POLLIN;

        if (worker.send.buffer.size > worker.send.written)
            pfd.events |= POLLOUT;

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
            int ret = process_data(&unpacker, &unpacked, pfd.fd);

            if (ret < 0) {
                if (ret == -2)
                    exit_code = EXIT_SUCCESS;
                break;
            }
        }
        if (pfd.revents & POLLOUT)
            send_buffer();
    }
    msgpack_unpacker_destroy(&unpacker);
    msgpack_unpacked_destroy(&unpacked);

    return cleanup(exit_code);
}
