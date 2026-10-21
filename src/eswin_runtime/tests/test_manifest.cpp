#include "../src/model/simple_json.hpp"
#include "../src/model/sha256.hpp"
#include "../src/model/model_manifest.hpp"
#include "../src/model/downloader.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <cstdlib>
#include <unistd.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\033[1;31m[FAIL]\033[0m " << msg << " (" << #cond << ") at " \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

static void test_json_parser() {
    std::cout << "[*] Testing Lightweight JSON Parser & Serializer..." << std::endl;

    std::string sample = R"({
        "name": "SmolLM-135M-int8",
        "tensor_parallel": 2,
        "active": true,
        "ratio": 1.25,
        "tags": ["llm", "npu", "dual-die"],
        "config": {
            "hidden_size": 576,
            "vocab_size": 49152
        }
    })";

    std::string err;
    eswin::JsonValue v = eswin::JsonValue::parse(sample, err);
    TEST_ASSERT(err.empty(), "JSON parse failed: " + err);
    TEST_ASSERT(v.is_object(), "Root is not object");
    TEST_ASSERT(v["name"].as_string() == "SmolLM-135M-int8", "name mismatch");
    TEST_ASSERT(v["tensor_parallel"].as_int() == 2, "tensor_parallel mismatch");
    TEST_ASSERT(v["active"].as_bool() == true, "active mismatch");
    TEST_ASSERT(v["ratio"].as_double() == 1.25, "ratio mismatch");
    TEST_ASSERT(v["tags"].is_array(), "tags is not array");
    TEST_ASSERT(v["tags"].as_array().size() == 3, "tags size mismatch");
    TEST_ASSERT(v["tags"][0].as_string() == "llm", "tags[0] mismatch");
    TEST_ASSERT(v["config"]["hidden_size"].as_int() == 576, "hidden_size mismatch");
    TEST_ASSERT(v["config"]["vocab_size"].as_int() == 49152, "vocab_size mismatch");

    // Serialization round-trip
    std::string dumped = v.dump(2);
    std::string err2;
    eswin::JsonValue v2 = eswin::JsonValue::parse(dumped, err2);
    TEST_ASSERT(err2.empty(), "Dumped JSON parse failed");
    TEST_ASSERT(v2["name"].as_string() == "SmolLM-135M-int8", "Roundtrip name mismatch");

    std::cout << "\033[1;32m[PASS]\033[0m JSON parser and serializer verified." << std::endl;
}

static void test_sha256() {
    std::cout << "[*] Testing NIST SHA-256 Reference Hashes..." << std::endl;

    // NIST Test Vector 1: Empty string
    std::string empty_hash = eswin::Sha256::hash_string("");
    TEST_ASSERT(empty_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "Empty string SHA-256 failed");

    // NIST Test Vector 2: "The quick brown fox jumps over the lazy dog"
    std::string fox_hash = eswin::Sha256::hash_string("The quick brown fox jumps over the lazy dog");
    TEST_ASSERT(fox_hash == "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
                "Quick brown fox SHA-256 failed");

    // File hashing test
    std::string tmp_path = "/tmp/eswin_test_sha.bin";
    {
        std::ofstream out(tmp_path, std::ios::binary);
        out << "The quick brown fox jumps over the lazy dog";
    }
    std::string file_hash = eswin::Sha256::hash_file(tmp_path);
    unlink(tmp_path.c_str());
    TEST_ASSERT(file_hash == fox_hash, "File hashing failed");

    std::cout << "\033[1;32m[PASS]\033[0m SHA-256 implementation verified against NIST standards." << std::endl;
}

static void test_manifest_verification() {
    std::cout << "[*] Testing ModelManifest Parsing and Integrity Checks..." << std::endl;

    std::string tmp_dir = "/tmp/eswin_manifest_test";
    std::system(("mkdir -p " + tmp_dir).c_str());

    std::string file0_path = tmp_dir + "/model_d0.model";
    std::string file1_path = tmp_dir + "/model_d1.model";

    {
        std::ofstream out0(file0_path, std::ios::binary);
        out0 << "DIE0_SYNTHETIC_WEIGHTS_4096_BYTES";
        std::ofstream out1(file1_path, std::ios::binary);
        out1 << "DIE1_SYNTHETIC_WEIGHTS_4096_BYTES";
    }

    std::string hash0 = eswin::Sha256::hash_file(file0_path);
    std::string hash1 = eswin::Sha256::hash_file(file1_path);

    eswin::ModelManifest manifest;
    manifest.name = "TestModel-DualDie";
    manifest.version = "1.0.0";
    manifest.tensor_parallel = 2;
    manifest.config.architecture = "smollm";
    manifest.config.vocab_size = 49152;
    manifest.config.hidden_size = 576;

    manifest.files.push_back({"model_d0.model", "weights_die0", hash0, 33, ""});
    manifest.files.push_back({"model_d1.model", "weights_die1", hash1, 33, ""});

    std::string err;
    TEST_ASSERT(manifest.verify_files(tmp_dir, err), "Integrity verification failed unexpectedly: " + err);
    std::cout << "    Verified valid manifest files successfully." << std::endl;

    // Now corrupt one file and assert verification fails!
    {
        std::ofstream out0(file0_path, std::ios::binary);
        out0 << "CORRUPTED_DATA_HERE";
    }
    TEST_ASSERT(!manifest.verify_files(tmp_dir, err), "Integrity verification should have failed on corrupted file!");
    std::cout << "    Successfully detected corrupted file: " << err << std::endl;

    // Clean up
    unlink(file0_path.c_str());
    unlink(file1_path.c_str());
    rmdir(tmp_dir.c_str());

    std::cout << "\033[1;32m[PASS]\033[0m ModelManifest integrity checks verified." << std::endl;
}

static void test_context_length_parsing() {
    std::cout << "[*] Testing Manifest Context Window (KV Cache Length) Parsing..." << std::endl;

    std::string err;

    // Absent -> documented default
    {
        eswin::ModelManifest m;
        std::string json = R"({"name":"NoCtx","config":{"architecture":"qwen2","vocab_size":151936}})";
        TEST_ASSERT(m.load_from_string(json, err), "Parse failed: " + err);
        TEST_ASSERT(m.config.context_length == 2048, "Default context_length should be 2048");
    }

    // Canonical open-package key
    {
        eswin::ModelManifest m;
        std::string json = R"({"name":"Ctx","config":{"context_length":8192}})";
        TEST_ASSERT(m.load_from_string(json, err), "Parse failed: " + err);
        TEST_ASSERT(m.config.context_length == 8192, "context_length mismatch");
    }

    // Vendor ENNP spelling ("token_num"), and the legacy aliases
    const char *aliases[] = {"max_seq_len", "kv_cache_len", "token_num"};
    for (const char *key : aliases) {
        eswin::ModelManifest m;
        std::string json = std::string(R"({"name":"A","config":{")") + key + R"(":1024}})";
        TEST_ASSERT(m.load_from_string(json, err), "Parse failed for alias: " + std::string(key));
        TEST_ASSERT(m.config.context_length == 1024,
                    std::string("alias not honoured: ") + key);
    }

    // Canonical key wins when several are present
    {
        eswin::ModelManifest m;
        std::string json = R"({"name":"P","config":{"token_num":1024,"context_length":4096}})";
        TEST_ASSERT(m.load_from_string(json, err), "Parse failed: " + err);
        TEST_ASSERT(m.config.context_length == 4096, "context_length should take precedence over token_num");
    }

    // Serialize round-trip keeps the value
    {
        eswin::ModelManifest m;
        m.name = "RoundTrip";
        m.config.context_length = 16384;
        std::string dumped = m.to_json_string(2);
        eswin::ModelManifest m2;
        TEST_ASSERT(m2.load_from_string(dumped, err), "Reparse failed: " + err);
        TEST_ASSERT(m2.config.context_length == 16384, "Round-trip lost context_length");
    }

    std::cout << "\033[1;32m[PASS]\033[0m Context window read from manifest (incl. vendor aliases)." << std::endl;
}

int main() {
    std::cout << "========================================================\n"
              << "  ESWIN Model Package Manifest & Downloader Test Suite  \n"
              << "========================================================\n\n";

    test_json_parser();
    test_sha256();
    test_manifest_verification();
    test_context_length_parsing();

    std::cout << "\n========================================================\n"
              << "\033[1;32m  ALL MILESTONE 3 MANIFEST TESTS PASSED!                \033[0m\n"
              << "========================================================\n";
    return 0;
}
