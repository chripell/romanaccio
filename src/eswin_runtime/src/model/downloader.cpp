#include "downloader.hpp"
#include "sha256.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace eswin {

static bool create_dir_recursive(const std::string &path) {
    if (path.empty()) return false;
    std::string current;
    for (size_t i = 0; i < path.size(); ++i) {
        current += path[i];
        if (path[i] == '/' || i == path.size() - 1) {
            struct stat st;
            if (stat(current.c_str(), &st) != 0) {
                if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }
    }
    return true;
}

std::string Downloader::get_default_cache_dir() {
    const char *env_cache = std::getenv("ESWIN_CACHE_DIR");
    if (env_cache && *env_cache) {
        return env_cache;
    }

    const char *home = std::getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.cache/eswin/models";
    }

    return "/tmp/eswin_models";
}

bool Downloader::download_file(
    const std::string &url,
    const std::string &dest_path,
    const std::string &expected_sha256,
    std::string &err,
    ProgressCallback progress_cb
) {
    // Check if file already exists with matching SHA-256
    struct stat st;
    if (stat(dest_path.c_str(), &st) == 0) {
        if (expected_sha256.empty()) {
            if (progress_cb) progress_cb(dest_path, 1.0f);
            return true;
        }
        std::string existing_hash = Sha256::hash_file(dest_path);
        if (existing_hash == expected_sha256) {
            if (progress_cb) progress_cb(dest_path, 1.0f);
            return true;
        }
    }

    if (url.empty()) {
        err = "No URL specified for missing file: " + dest_path;
        return false;
    }

    // Handle local file copying (file:// or local path)
    if (url.rfind("file://", 0) == 0 || url[0] == '/') {
        std::string src_path = (url.rfind("file://", 0) == 0) ? url.substr(7) : url;
        std::ifstream src(src_path, std::ios::binary);
        if (!src.is_open()) {
            err = "Source file not found: " + src_path;
            return false;
        }
        std::ofstream dst(dest_path, std::ios::binary);
        if (!dst.is_open()) {
            err = "Cannot open destination file for writing: " + dest_path;
            return false;
        }
        dst << src.rdbuf();
    } else {
        // Remote download using curl
        std::string cmd = "curl -sSfL -C - --connect-timeout 15 -o \"" + dest_path + "\" \"" + url + "\"";
        if (progress_cb) progress_cb(dest_path, 0.1f);
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            err = "Download failed (curl exit code " + std::to_string(ret) + ") for URL: " + url;
            return false;
        }
    }

    // Post-download SHA-256 verification
    if (!expected_sha256.empty()) {
        std::string actual_hash = Sha256::hash_file(dest_path);
        if (actual_hash != expected_sha256) {
            err = "SHA-256 verification failed for " + dest_path + ": expected " +
                  expected_sha256 + ", got " + actual_hash;
            unlink(dest_path.c_str());
            return false;
        }
    }

    if (progress_cb) progress_cb(dest_path, 1.0f);
    return true;
}

bool Downloader::ensure_model(
    const ModelManifest &manifest,
    const std::string &cache_dir,
    std::string &out_model_dir,
    std::string &err,
    ProgressCallback progress_cb
) {
    if (manifest.name.empty()) {
        err = "Manifest name cannot be empty";
        return false;
    }

    std::string base_cache = cache_dir.empty() ? get_default_cache_dir() : cache_dir;
    std::string target_dir = base_cache + "/" + manifest.name;

    if (!create_dir_recursive(target_dir)) {
        err = "Failed to create directory: " + target_dir;
        return false;
    }

    // Save manifest.json in the target directory
    std::string manifest_path = target_dir + "/manifest.json";
    std::string save_err;
    manifest.save_to_file(manifest_path, save_err);

    // Download/verify each component file
    for (const auto &f : manifest.files) {
        std::string dest_path = target_dir + "/" + f.name;
        if (!download_file(f.url, dest_path, f.sha256, err, progress_cb)) {
            return false;
        }
    }

    out_model_dir = target_dir;
    return true;
}

} // namespace eswin
