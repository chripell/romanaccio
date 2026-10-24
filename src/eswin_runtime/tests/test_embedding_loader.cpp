/**
 * @file test_embedding_loader.cpp
 * @brief Milestone 8 test suite: zero-copy embeddings and .safetensors loading.
 *
 * Builds synthetic tables and containers on disk so the checks are real mmap/parse
 * paths rather than mocks: fp32/fp16/bf16 row lookup and gather, bounds rejection,
 * precision inference, O(1) open time against a 466 MB sparse table (the size of the
 * Qwen2.5-Coder embedding), and .safetensors header parsing with truncation and
 * shape/offset mismatch rejection.
 */

#include "../src/model/embedding_table.hpp"
#include "../src/model/safetensors_loader.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#define TEST_ASSERT(cond, msg)                                                  \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::cerr << "\033[1;31m[FAIL]\033[0m " << msg << " (" << #cond     \
                      << ") at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

using namespace eswin;

namespace {

const std::string kDir = "/tmp/eswin_m8_test";

std::string path(const std::string &name) { return kDir + "/" + name; }

uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

uint16_t f32_to_bf16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    return static_cast<uint16_t>(x >> 16);
}

void write_bytes(const std::string &p, const void *data, size_t n) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    TEST_ASSERT(out.good(), "cannot create " + p);
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(n));
    out.close();
}

// --- fp16/bf16 conversion primitives -----------------------------------------

void test_half_conversion() {
    std::cout << "[*] Testing fp16 / bf16 conversion primitives..." << std::endl;

    struct { uint16_t bits; float expect; } cases[] = {
        {0x3C00, 1.0f}, {0x3800, 0.5f}, {0xC000, -2.0f}, {0x0000, 0.0f},
        {0x4200, 3.0f}, {0xBC00, -1.0f}, {0x7C00, INFINITY},
    };
    for (const auto &c : cases) {
        float got = es_f16_to_f32(c.bits);
        TEST_ASSERT(got == c.expect || (std::isinf(got) && std::isinf(c.expect)),
                    "f16 bits 0x" + std::to_string(c.bits) + " converted wrong");
    }
    // Subnormal fp16 (0x0001 = 2^-24) must not come back as zero.
    float sub = es_f16_to_f32(0x0001);
    TEST_ASSERT(sub > 0.0f && sub < 1e-6f, "fp16 subnormal not renormalized");

    TEST_ASSERT(es_bf16_to_f32(f32_to_bf16_bits(1.5f)) == 1.5f, "bf16 round-trip failed");
    TEST_ASSERT(es_bf16_to_f32(f32_to_bf16_bits(-0.25f)) == -0.25f, "bf16 negative failed");
    TEST_ASSERT(es_f16_to_f32(f32_to_f16_bits(0.75f)) == 0.75f, "fp16 round-trip failed");

    std::cout << "    fp16 (incl. subnormal/inf) and bf16 conversions verified." << std::endl;
}

// --- EmbeddingTable ----------------------------------------------------------

void test_embedding_fp32() {
    std::cout << "\n[*] Testing fp32 embedding table lookup and gather..." << std::endl;

    const size_t rows = 4, cols = 3;
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
    };
    write_bytes(path("emb_f32.bin"), data.data(), data.size() * sizeof(float));

    EmbeddingTable t;
    std::string err;
    TEST_ASSERT(t.open(path("emb_f32.bin"), rows, cols, EmbeddingPrecision::Fp32, err),
                "fp32 open failed: " + err);
    TEST_ASSERT(t.is_open(), "table not open");
    TEST_ASSERT(t.rows() == rows && t.cols() == cols, "shape mismatch");
    TEST_ASSERT(t.precision() == EmbeddingPrecision::Fp32, "precision mismatch");
    TEST_ASSERT(t.mapped_bytes() == rows * cols * sizeof(float), "mapped size mismatch");

    std::vector<float> row(cols, -1.0f);
    TEST_ASSERT(t.get_row(2, row.data(), row.size()), "get_row(2) failed");
    TEST_ASSERT(row[0] == 7.0f && row[1] == 8.0f && row[2] == 9.0f, "row 2 values wrong");

    // Out-of-range and undersized destination must be rejected, not read blindly.
    TEST_ASSERT(!t.get_row(rows, row.data(), row.size()), "out-of-range row accepted");
    TEST_ASSERT(!t.get_row(0, row.data(), cols - 1), "undersized destination accepted");
    TEST_ASSERT(t.raw_row(rows) == nullptr, "raw_row out of range not null");

    // Gather with one bogus token in the middle: valid rows copied, bogus zero-filled.
    std::vector<float> gathered(3 * cols, -1.0f);
    const int32_t tokens[3] = {3, 99, 0};
    size_t n = t.gather(tokens, 3, gathered.data(), gathered.size());
    TEST_ASSERT(n == 3, "gather wrote the wrong number of rows");
    TEST_ASSERT(gathered[0] == 10.0f && gathered[2] == 12.0f, "gather row 0 wrong");
    TEST_ASSERT(gathered[3] == 0.0f && gathered[4] == 0.0f && gathered[5] == 0.0f,
                "out-of-range token was not zero-filled");
    TEST_ASSERT(gathered[6] == 1.0f, "gather row 2 wrong");

    t.close();
    TEST_ASSERT(!t.is_open(), "still open after close()");
    std::cout << "    fp32 lookup, bounds rejection and gather verified." << std::endl;
}

void test_embedding_fp16_bf16() {
    std::cout << "\n[*] Testing fp16 and bf16 tables..." << std::endl;

    const size_t rows = 2, cols = 2;
    std::vector<float> want = {0.5f, -1.5f, 2.0f, 0.25f};

    std::vector<uint16_t> h(rows * cols), b(rows * cols);
    for (size_t i = 0; i < want.size(); ++i) {
        h[i] = f32_to_f16_bits(want[i]);
        b[i] = f32_to_bf16_bits(want[i]);
    }
    write_bytes(path("emb_f16.bin"), h.data(), h.size() * sizeof(uint16_t));
    write_bytes(path("emb_bf16.bin"), b.data(), b.size() * sizeof(uint16_t));

    std::string err;
    EmbeddingTable th;
    TEST_ASSERT(th.open(path("emb_f16.bin"), rows, cols, EmbeddingPrecision::Fp16, err),
                "fp16 open failed: " + err);
    EmbeddingTable tb;
    TEST_ASSERT(tb.open(path("emb_bf16.bin"), rows, cols, EmbeddingPrecision::Bf16, err),
                "bf16 open failed: " + err);

    std::vector<float> rh(cols), rb(cols);
    for (size_t r = 0; r < rows; ++r) {
        TEST_ASSERT(th.get_row(r, rh.data(), rh.size()), "fp16 get_row failed");
        TEST_ASSERT(tb.get_row(r, rb.data(), rb.size()), "bf16 get_row failed");
        for (size_t c = 0; c < cols; ++c) {
            const float expect = want[r * cols + c];
            TEST_ASSERT(std::fabs(rh[c] - expect) < 1e-3f, "fp16 value wrong");
            TEST_ASSERT(std::fabs(rb[c] - expect) < 1e-2f, "bf16 value wrong");
        }
    }

    // Move construction must transfer the mapping, not duplicate or drop it.
    EmbeddingTable moved(std::move(th));
    TEST_ASSERT(moved.is_open() && !th.is_open(), "move did not transfer ownership");
    TEST_ASSERT(moved.get_row(0, rh.data(), rh.size()) && rh[0] == 0.5f,
                "moved table reads wrong");

    std::cout << "    fp16/bf16 decode and move semantics verified." << std::endl;
}

void test_embedding_auto_and_errors() {
    std::cout << "\n[*] Testing precision inference and error handling..." << std::endl;

    const size_t rows = 4, cols = 3;
    std::string err;

    // fp32-sized file is detected as fp32.
    std::vector<float> f32(rows * cols, 1.0f);
    write_bytes(path("auto_f32.bin"), f32.data(), f32.size() * sizeof(float));
    EmbeddingTable a;
    TEST_ASSERT(a.open_auto(path("auto_f32.bin"), rows, cols, err), "fp32 auto failed: " + err);
    TEST_ASSERT(a.precision() == EmbeddingPrecision::Fp32, "fp32 not inferred");

    // 2-byte-per-element file is detected as fp16 (the ESWIN convention).
    std::vector<uint16_t> f16(rows * cols, 0x3C00);
    write_bytes(path("auto_f16.bin"), f16.data(), f16.size() * sizeof(uint16_t));
    EmbeddingTable b;
    TEST_ASSERT(b.open_auto(path("auto_f16.bin"), rows, cols, err), "fp16 auto failed: " + err);
    TEST_ASSERT(b.precision() == EmbeddingPrecision::Fp16, "fp16 not inferred");

    // Size that matches neither encoding must be rejected with a specific message.
    std::vector<uint8_t> odd(rows * cols * 3, 0);
    write_bytes(path("auto_bad.bin"), odd.data(), odd.size());
    EmbeddingTable c;
    TEST_ASSERT(!c.open_auto(path("auto_bad.bin"), rows, cols, err), "bad size accepted");
    TEST_ASSERT(err.find("cannot infer precision") != std::string::npos,
                "unhelpful error for ambiguous size: " + err);

    // Exact-size validation catches a truncated table.
    std::vector<float> trunc(rows * cols - 1, 0.0f);
    write_bytes(path("trunc.bin"), trunc.data(), trunc.size() * sizeof(float));
    EmbeddingTable d;
    TEST_ASSERT(!d.open(path("trunc.bin"), rows, cols, EmbeddingPrecision::Fp32, err),
                "truncated table accepted");
    TEST_ASSERT(err.find("size mismatch") != std::string::npos,
                "unhelpful truncation error: " + err);

    // Missing file and zero dimensions.
    EmbeddingTable e;
    TEST_ASSERT(!e.open(path("does_not_exist.bin"), rows, cols, EmbeddingPrecision::Fp32, err),
                "missing file accepted");
    TEST_ASSERT(!e.open_auto("", rows, cols, err), "empty path accepted");
    EmbeddingTable f;
    TEST_ASSERT(!f.open(path("auto_f32.bin"), 0, cols, EmbeddingPrecision::Fp32, err),
                "zero vocab_size accepted");

    std::cout << "    Inference, truncation, missing-file and bad-dimension cases rejected."
              << std::endl;
}

void test_embedding_open_is_o1() {
    std::cout << "\n[*] Testing open() cost against a 466 MB sparse table..." << std::endl;

    // Exact size of the Qwen2.5-Coder-1.5B embedding: 151936 x 1536 x fp16.
    const size_t rows = 151936, cols = 1536;
    const size_t bytes = rows * cols * 2;
    const std::string p = path("big_sparse.bin");

    FILE *fp = std::fopen(p.c_str(), "wb");
    TEST_ASSERT(fp != nullptr, "cannot create sparse file");
    TEST_ASSERT(::ftruncate(fileno(fp), static_cast<off_t>(bytes)) == 0, "ftruncate failed");
    std::fclose(fp);

    EmbeddingTable t;
    std::string err;
    TEST_ASSERT(t.open(p, rows, cols, EmbeddingPrecision::Fp16, err), "big open failed: " + err);
    TEST_ASSERT(t.mapped_bytes() == bytes, "mapped size wrong");

    std::cout << "    Mapped " << bytes << " bytes in " << t.open_ms() << " ms" << std::endl;
    TEST_ASSERT(t.open_ms() < 5.0,
                "open() took " + std::to_string(t.open_ms()) + " ms (PLAN target < 5 ms)");

    // Touching one row must work without prefaulting the whole table.
    std::vector<float> row(cols, -1.0f);
    TEST_ASSERT(t.get_row(rows - 1, row.data(), row.size()), "last-row lookup failed");
    TEST_ASSERT(row[0] == 0.0f, "sparse row should read as zero");

    t.close();
    std::remove(p.c_str());
}

// --- SafetensorsLoader -------------------------------------------------------

void write_safetensors(const std::string &p, const std::string &header,
                       const std::vector<uint8_t> &data) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    TEST_ASSERT(out.good(), "cannot create " + p);
    uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
        char byte = static_cast<char>((n >> (8 * i)) & 0xFF);
        out.write(&byte, 1);
    }
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!data.empty()) {
        out.write(reinterpret_cast<const char *>(data.data()),
                  static_cast<std::streamsize>(data.size()));
    }
    out.close();
}

void test_safetensors() {
    std::cout << "\n[*] Testing .safetensors parsing..." << std::endl;

    // w: 2x3 F32 (24 bytes) at [0,24); h: 4 x F16 (8 bytes) at [24,32).
    const std::string header =
        R"({"__metadata__":{"format":"pt","producer":"eswin-test"},)"
        R"("w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
        R"("h":{"dtype":"F16","shape":[4],"data_offsets":[24,32]}})";

    std::vector<uint8_t> data(32, 0);
    const float wvals[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::memcpy(data.data(), wvals, sizeof(wvals));
    const uint16_t hvals[4] = {
        f32_to_f16_bits(0.5f), f32_to_f16_bits(-1.0f),
        f32_to_f16_bits(2.0f), f32_to_f16_bits(0.25f),
    };
    std::memcpy(data.data() + 24, hvals, sizeof(hvals));

    write_safetensors(path("t.safetensors"), header, data);

    SafetensorsLoader st;
    std::string err;
    TEST_ASSERT(st.open(path("t.safetensors"), err), "safetensors open failed: " + err);
    TEST_ASSERT(st.is_open(), "loader not open");
    TEST_ASSERT(st.tensor_count() == 2, "expected 2 tensors (metadata is not a tensor)");

    auto names = st.tensor_names();
    TEST_ASSERT(names.size() == 2 && names[0] == "h" && names[1] == "w",
                "tensor names wrong (expected alphabetical)");

    const SafetensorInfo *w = st.find("w");
    TEST_ASSERT(w != nullptr, "tensor 'w' not found");
    TEST_ASSERT(w->dtype == "F32", "w dtype wrong");
    TEST_ASSERT(w->shape.size() == 2 && w->shape[0] == 2 && w->shape[1] == 3, "w shape wrong");
    TEST_ASSERT(SafetensorsLoader::element_count(*w) == 6, "w element count wrong");
    TEST_ASSERT(w->offset == 0 && w->length == 24, "w offsets wrong");

    const SafetensorInfo *h = st.find("h");
    TEST_ASSERT(h != nullptr && h->dtype == "F16", "h missing or wrong dtype");
    TEST_ASSERT(h->offset == 24 && h->length == 8, "h offsets wrong");

    TEST_ASSERT(st.metadata().count("format") && st.metadata().at("format") == "pt",
                "metadata not parsed");
    TEST_ASSERT(st.find("nope") == nullptr, "unknown tensor should be null");

    // F32 conversion is a straight copy.
    std::vector<float> wf;
    TEST_ASSERT(st.to_f32("w", wf, err), "to_f32(w) failed: " + err);
    TEST_ASSERT(wf.size() == 6, "w float count wrong");
    for (int i = 0; i < 6; ++i) {
        TEST_ASSERT(wf[i] == wvals[i], "w value mismatch");
    }

    // F16 conversion decodes through the same primitive the embedding table uses.
    std::vector<float> hf;
    TEST_ASSERT(st.to_f32("h", hf, err), "to_f32(h) failed: " + err);
    const float hexpect[4] = {0.5f, -1.0f, 2.0f, 0.25f};
    for (int i = 0; i < 4; ++i) {
        TEST_ASSERT(std::fabs(hf[i] - hexpect[i]) < 1e-3f, "h value mismatch");
    }

    // Raw pointer access is zero-copy into the mapping.
    size_t nbytes = 0;
    const void *raw = st.data("w", &nbytes);
    TEST_ASSERT(raw != nullptr && nbytes == 24, "raw data accessor wrong");
    TEST_ASSERT(std::memcmp(raw, wvals, sizeof(wvals)) == 0, "raw bytes differ");

    st.close();
    TEST_ASSERT(!st.is_open(), "still open after close");
    std::cout << "    Header, shapes, offsets, metadata and F32/F16 decode verified."
              << std::endl;
}

void test_safetensors_errors() {
    std::cout << "\n[*] Testing .safetensors rejection of malformed files..." << std::endl;

    std::string err;

    // 1. data_offsets running past the end of the data section (truncated download).
    {
        const std::string hdr = R"({"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})";
        std::vector<uint8_t> short_data(12, 0);  // claims 24 bytes, provides 12
        write_safetensors(path("bad_trunc.safetensors"), hdr, short_data);
        SafetensorsLoader st;
        TEST_ASSERT(!st.open(path("bad_trunc.safetensors"), err), "truncated data accepted");
        TEST_ASSERT(err.find("truncated") != std::string::npos,
                    "unhelpful truncation error: " + err);
    }

    // 2. shape x dtype inconsistent with the declared byte length.
    {
        const std::string hdr = R"({"w":{"dtype":"F32","shape":[4,4],"data_offsets":[0,24]}})";
        std::vector<uint8_t> data(24, 0);
        write_safetensors(path("bad_shape.safetensors"), hdr, data);
        SafetensorsLoader st;
        TEST_ASSERT(!st.open(path("bad_shape.safetensors"), err), "shape mismatch accepted");
        TEST_ASSERT(err.find("implies") != std::string::npos,
                    "unhelpful shape error: " + err);
    }

    // 3. Unsupported dtype is named, not silently reinterpreted.
    {
        const std::string hdr = R"({"w":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}})";
        std::vector<uint8_t> data(2, 0);
        write_safetensors(path("bad_dtype.safetensors"), hdr, data);
        SafetensorsLoader st;
        TEST_ASSERT(!st.open(path("bad_dtype.safetensors"), err), "unknown dtype accepted");
        TEST_ASSERT(err.find("unsupported dtype") != std::string::npos,
                    "unhelpful dtype error: " + err);
    }

    // 4. Header length exceeding the file.
    {
        std::ofstream out(path("bad_header.safetensors"), std::ios::binary | std::ios::trunc);
        const uint64_t huge = 1000000;
        for (int i = 0; i < 8; ++i) {
            char byte = static_cast<char>((huge >> (8 * i)) & 0xFF);
            out.write(&byte, 1);
        }
        out.write("{}", 2);
        out.close();
        SafetensorsLoader st;
        TEST_ASSERT(!st.open(path("bad_header.safetensors"), err), "bad header len accepted");
        TEST_ASSERT(err.find("does not fit") != std::string::npos,
                    "unhelpful header error: " + err);
    }

    // 5. Missing file and a too-small file.
    {
        SafetensorsLoader st;
        TEST_ASSERT(!st.open(path("nope.safetensors"), err), "missing file accepted");
        write_bytes(path("tiny.safetensors"), "abc", 3);
        TEST_ASSERT(!st.open(path("tiny.safetensors"), err), "3-byte file accepted");
    }

    std::cout << "    Truncation, shape, dtype, header-length and missing-file cases rejected."
              << std::endl;
}

} // namespace

int main() {
    std::cout << "========================================================\n"
              << "  ESWIN Embedding & Weights Loader Test Suite (MS 8)    \n"
              << "========================================================\n\n";

    if (std::system(("mkdir -p " + kDir).c_str()) != 0) {
        std::cerr << "cannot create " << kDir << std::endl;
        return 1;
    }

    test_half_conversion();
    test_embedding_fp32();
    test_embedding_fp16_bf16();
    test_embedding_auto_and_errors();
    test_embedding_open_is_o1();
    test_safetensors();
    test_safetensors_errors();

    std::system(("rm -rf " + kDir).c_str());

    std::cout << "\n========================================================\n"
              << "  ALL MILESTONE 8 EMBEDDING LOADER TESTS PASSED!        \n"
              << "========================================================\n";
    return 0;
}
