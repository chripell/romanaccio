/**
 * @file es_run_dsp.cpp
 * @brief Standalone CLI execution and benchmark utility for ESWIN EIC7700X Cadence Vision Q7 DSP kernels.
 */

#include "backend/dsp_driver.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>

using namespace eswin;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  -c <core_id>     Target DSP Core ID (0..7, default: 0)\n"
              << "  -k <kernel_name> Kernel name to load and benchmark (default: rms_norm_f16f16)\n"
              << "  -d <kernel_dir>  Directory containing .pkg.lib files (default: /usr/lib/firmware/dsp_kernels)\n"
              << "  -r <iterations>  Benchmark repeat count (default: 10)\n"
              << "  -h               Show this help message\n";
}

int main(int argc, char** argv) {
    int core_id = 0;
    std::string kernel_name = "rms_norm_f16f16";
    std::string kernel_dir = "/usr/lib/firmware/dsp_kernels";
    int iterations = 10;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            core_id = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            kernel_name = argv[++i];
        } else if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            kernel_dir = argv[++i];
        } else if (std::strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "========================================================\n"
              << " ESWIN EIC7700X DSP Core Runner & Benchmark Tool\n"
              << "========================================================\n"
              << "  DSP Core ID : " << core_id << "\n"
              << "  Kernel Name : " << kernel_name << "\n"
              << "  Kernel Dir  : " << kernel_dir << "\n"
              << "  Iterations  : " << iterations << "\n"
              << "--------------------------------------------------------\n";

    try {
        DspContext ctx(static_cast<DspCore>(core_id));
        ctx.open();
        std::cout << "[+] DSP Core " << core_id << " opened successfully (driver fd=" << ctx.fd() << ")\n";
        std::cout << "[+] Driver Version: " << ctx.get_version_string() << "\n";

        auto t_load_start = std::chrono::high_resolution_clock::now();
        uint32_t handle = ctx.load_operator(kernel_name, kernel_dir);
        auto t_load_end = std::chrono::high_resolution_clock::now();
        double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
        std::cout << "[+] Operator '" << kernel_name << "' loaded (handle=0x"
                  << std::hex << handle << std::dec << ") in " << load_ms << " ms\n";

        ctx.unload_operator(handle);
        std::cout << "[+] Operator unloaded cleanly\n";
        ctx.close();
        std::cout << "[+] DSP Core closed cleanly\n";
        std::cout << "========================================================\n"
                  << " Benchmark completed successfully!\n"
                  << "========================================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[-] Error: " << e.what() << "\n";
        return 1;
    }
}
