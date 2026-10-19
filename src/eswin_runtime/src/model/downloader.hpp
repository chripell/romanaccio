#ifndef ESWIN_DOWNLOADER_HPP
#define ESWIN_DOWNLOADER_HPP

#include "model_manifest.hpp"
#include <string>
#include <functional>

namespace eswin {

class Downloader {
public:
    using ProgressCallback = std::function<void(const std::string &filename, float progress)>;

    // Cache path resolution
    static std::string get_default_cache_dir();

    // Single file download and verification
    static bool download_file(
        const std::string &url,
        const std::string &dest_path,
        const std::string &expected_sha256,
        std::string &err,
        ProgressCallback progress_cb = nullptr
    );

    // Ensure all components of a model manifest are present and verified in cache
    static bool ensure_model(
        const ModelManifest &manifest,
        const std::string &cache_dir,
        std::string &out_model_dir,
        std::string &err,
        ProgressCallback progress_cb = nullptr
    );
};

} // namespace eswin

#endif // ESWIN_DOWNLOADER_HPP
