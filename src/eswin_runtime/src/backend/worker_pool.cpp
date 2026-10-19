/**
 * @file worker_pool.cpp
 * @brief Implementation of Multi-Process Dual-Die Isolation & NUMA Worker Daemon.
 */

#include "worker_pool.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <sched.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>

namespace eswin {

// =============================================================================
// WorkerChannel Implementation
// =============================================================================

WorkerChannel::WorkerChannel(uint16_t die_id)
    : m_die_id(die_id) {}

WorkerChannel::~WorkerChannel() {
    shutdown();
}

bool WorkerChannel::start() {
    int fds[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, fds) != 0) {
        std::cerr << "[WorkerChannel] Failed to create socketpair: " << std::strerror(errno) << std::endl;
        return false;
    }

    m_master_fd = fds[0];
    m_worker_fd = fds[1];

    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "[WorkerChannel] Failed to fork worker process: " << std::strerror(errno) << std::endl;
        close(m_master_fd);
        close(m_worker_fd);
        m_master_fd = -1;
        m_worker_fd = -1;
        return false;
    }

    if (pid == 0) {
        // Child process (Worker Daemon)
        close(m_master_fd);
        m_master_fd = -1;

        // 1. Set NUMA CPU affinity
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (m_die_id == 0) {
            // NUMA Node 0 (Cores 0-3)
            CPU_SET(0, &cpuset);
            CPU_SET(1, &cpuset);
            CPU_SET(2, &cpuset);
            CPU_SET(3, &cpuset);
        } else {
            // NUMA Node 1 (Cores 4-7)
            CPU_SET(4, &cpuset);
            CPU_SET(5, &cpuset);
            CPU_SET(6, &cpuset);
            CPU_SET(7, &cpuset);
        }
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

        // 2. Initialize Hardware Subsystem for this die
        auto &driver = EswinDriver::instance();
        bool hw = driver.initialize();
        if (hw) {
            driver.set_device(m_die_id);
        }

        // 3. Send initial Handshake Ready message to master
        IpcResponse ready_resp{};
        ready_resp.status = IpcStatus::Success;
        ready_resp.seq_id = 0;
        send(m_worker_fd, &ready_resp, sizeof(ready_resp), 0);

        // 4. Enter worker event loop
        run_worker_loop();

        if (hw) {
            driver.release_device(m_die_id);
            driver.shutdown();
        }

        close(m_worker_fd);
        _exit(0);
    }

    // Parent process (Master)
    close(m_worker_fd);
    m_worker_fd = -1;
    m_pid = pid;

    // Explicitly allow child worker processes to import DMA buffer handles via pidfd
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);

    // Wait for handshake with timeout (5000 ms)
    struct pollfd pfd{};
    pfd.fd = m_master_fd;
    pfd.events = POLLIN;

    int ret = poll(&pfd, 1, 5000);
    if (ret <= 0) {
        std::cerr << "[WorkerChannel] Timeout waiting for worker " << m_die_id << " to initialize." << std::endl;
        shutdown();
        return false;
    }

    IpcResponse handshake{};
    ssize_t n = recv(m_master_fd, &handshake, sizeof(handshake), 0);
    if (n != sizeof(handshake) || handshake.status != IpcStatus::Success) {
        std::cerr << "[WorkerChannel] Invalid handshake from worker " << m_die_id << std::endl;
        shutdown();
        return false;
    }

    m_is_alive = true;
    return true;
}

void WorkerChannel::shutdown() {
    if (!m_is_alive && m_pid <= 0) return;

    if (m_master_fd >= 0) {
        IpcRequest req{};
        req.cmd = IpcCmd::Shutdown;
        req.seq_id = ++m_seq_counter;
        send(m_master_fd, &req, sizeof(req), MSG_NOSIGNAL);

        // Read acknowledge with short deadline
        struct pollfd pfd{};
        pfd.fd = m_master_fd;
        pfd.events = POLLIN;
        if (poll(&pfd, 1, 1000) > 0) {
            IpcResponse resp{};
            recv(m_master_fd, &resp, sizeof(resp), 0);
        }

        close(m_master_fd);
        m_master_fd = -1;
    }

    if (m_pid > 0) {
        int status = 0;
        // Wait up to 1 second for graceful exit
        for (int i = 0; i < 10; ++i) {
            pid_t r = waitpid(m_pid, &status, WNOHANG);
            if (r == m_pid) {
                m_pid = -1;
                break;
            }
            usleep(100000);
        }

        if (m_pid > 0) {
            // Force terminate if hung
            kill(m_pid, SIGKILL);
            waitpid(m_pid, &status, 0);
            m_pid = -1;
        }
    }

    m_is_alive = false;
}

bool WorkerChannel::send_request(const IpcRequest &req, IpcResponse &resp) {
    if (!m_is_alive || m_master_fd < 0) return false;

    ssize_t sent = send(m_master_fd, &req, sizeof(req), MSG_NOSIGNAL);
    if (sent != sizeof(req)) {
        m_is_alive = false;
        return false;
    }

    // Wait for response (up to 30 second deadline)
    struct pollfd pfd{};
    pfd.fd = m_master_fd;
    pfd.events = POLLIN;

    int ret = poll(&pfd, 1, 30000);
    if (ret <= 0) {
        std::cerr << "[WorkerChannel] Timeout awaiting response from die " << m_die_id << std::endl;
        m_is_alive = false;
        return false;
    }

    ssize_t n = recv(m_master_fd, &resp, sizeof(resp), 0);
    if (n != sizeof(resp)) {
        m_is_alive = false;
        return false;
    }

    return (resp.status == IpcStatus::Success);
}

void WorkerChannel::run_worker_loop() {
    auto &driver = EswinDriver::instance();
    bool is_hw = driver.is_hardware_available();
    std::unordered_map<uint64_t, uint64_t> rcvd_fds;

    while (true) {
        IpcRequest req{};
        ssize_t n = recv(m_worker_fd, &req, sizeof(req), 0);
        if (n <= 0) {
            // Socket closed or EOF
            break;
        }

        IpcResponse resp{};
        resp.seq_id = req.seq_id;
        resp.status = IpcStatus::Success;

        switch (req.cmd) {
            case IpcCmd::Ping: {
                resp.status = IpcStatus::Success;
                break;
            }

            case IpcCmd::LoadModel: {
                if (is_hw) {
                    driver.set_device(m_die_id);
                    uint32_t model_id = 0;
                    int ret = driver.load_model(&model_id, req.model_path);
                    if (ret == 0) {
                        resp.model_id = model_id;
                        uint32_t num_in = 0, num_out = 0;
                        driver.get_num_input_tensors(model_id, &num_in);
                        driver.get_num_output_tensors(model_id, &num_out);
                        resp.num_inputs = static_cast<uint8_t>(num_in);
                        resp.num_outputs = static_cast<uint8_t>(num_out);

                        for (uint32_t i = 0; i < num_in && i < 10; ++i) {
                            NpuTensorDesc desc{};
                            driver.get_input_tensor_desc(model_id, i, &desc);
                            resp.input_sizes[i] = desc.buffer_size;
                        }

                        for (uint32_t i = 0; i < num_out && i < 10; ++i) {
                            NpuTensorDesc desc{};
                            driver.get_output_tensor_desc(model_id, i, &desc);
                            resp.output_sizes[i] = desc.buffer_size;
                        }

                        resp.status = IpcStatus::Success;
                    } else {
                        resp.status = IpcStatus::ErrorExecutionFailed;
                        resp.driver_error = ret;
                    }
                } else {
                    // Host fallback
                    resp.model_id = 100 + m_die_id;
                    resp.num_inputs = 1;
                    resp.num_outputs = 1;
                    resp.input_sizes[0] = 4096;
                    resp.output_sizes[0] = 4096;
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::UnloadModel: {
                if (is_hw) {
                    driver.set_device(m_die_id);
                    int ret = driver.unload_model(req.model_id);
                    resp.status = (ret == 0) ? IpcStatus::Success : IpcStatus::ErrorExecutionFailed;
                    resp.driver_error = ret;
                } else {
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::PrepareBuffer: {
                if (is_hw) {
                    driver.set_device(m_die_id);

                    uint64_t snd_fd = req.inputs[0].mem_fd;
                    uint64_t rcvd_fd = snd_fd;

                    pid_t snd_pid = static_cast<pid_t>(snd_fd >> 32);
                    if (snd_pid != 0 && snd_pid != getpid()) {
                        int r = driver.mem_rcv_fd(snd_fd, &rcvd_fd);
                        if (r != 0) {
                            std::cerr << "[Worker " << m_die_id << "] mem_rcv_fd failed for snd_fd 0x"
                                      << std::hex << snd_fd << std::dec << ": " << driver.decode_error(r) << std::endl;
                            resp.status = IpcStatus::ErrorExecutionFailed;
                            resp.driver_error = r;
                            break;
                        }
                    }

                    rcvd_fds[snd_fd] = rcvd_fd;

                    DevBuf dev_buf{};
                    dev_buf.mem_fd = rcvd_fd;
                    dev_buf.offset = req.inputs[0].offset;
                    dev_buf.size = req.inputs[0].size;
                    int ret = driver.prepare_dev_buf(&dev_buf);
                    if (ret != 0) {
                        std::cerr << "[Worker " << m_die_id << "] prepare_dev_buf failed on rcvd_fd 0x"
                                  << std::hex << rcvd_fd << std::dec << ": " << driver.decode_error(ret) << std::endl;
                    }
                    resp.status = (ret == 0) ? IpcStatus::Success : IpcStatus::ErrorExecutionFailed;
                    resp.driver_error = ret;
                } else {
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::UnprepareBuffer: {
                if (is_hw) {
                    driver.set_device(m_die_id);

                    uint64_t snd_fd = req.inputs[0].mem_fd;
                    uint64_t rcvd_fd = snd_fd;
                    auto it = rcvd_fds.find(snd_fd);
                    if (it != rcvd_fds.end()) {
                        rcvd_fd = it->second;
                    }

                    DevBuf dev_buf{};
                    dev_buf.mem_fd = rcvd_fd;
                    dev_buf.offset = req.inputs[0].offset;
                    dev_buf.size = req.inputs[0].size;
                    int ret = driver.unprepare_dev_buf(&dev_buf);

                    if (it != rcvd_fds.end() && it->second != snd_fd) {
                        driver.mem_free(it->second);
                        rcvd_fds.erase(it);
                    }

                    resp.status = (ret == 0) ? IpcStatus::Success : IpcStatus::ErrorExecutionFailed;
                    resp.driver_error = ret;
                } else {
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::SubmitTask: {
                auto t0 = std::chrono::high_resolution_clock::now();

                if (is_hw) {
                    driver.set_device(m_die_id);

                    NpuTask task{};
                    task.modelId = req.model_id;
                    task.inputFdNum = req.input_count;
                    task.outputFdNum = req.output_count;

                    for (uint8_t i = 0; i < req.input_count && i < 10; ++i) {
                        uint64_t snd_fd = req.inputs[i].mem_fd;
                        auto it = rcvd_fds.find(snd_fd);
                        task.inputFd[i].mem_fd = (it != rcvd_fds.end()) ? it->second : snd_fd;
                        task.inputFd[i].offset = req.inputs[i].offset;
                        task.inputFd[i].size   = req.inputs[i].size;
                        task.inputFd[i].reserved = 0;
                    }

                    for (uint8_t i = 0; i < req.output_count && i < 10; ++i) {
                        uint64_t snd_fd = req.outputs[i].mem_fd;
                        auto it = rcvd_fds.find(snd_fd);
                        task.outputFd[i].mem_fd = (it != rcvd_fds.end()) ? it->second : snd_fd;
                        task.outputFd[i].offset = req.outputs[i].offset;
                        task.outputFd[i].size   = req.outputs[i].size;
                        task.outputFd[i].reserved = 0;
                    }

                    int ret = driver.submit_task(&task, 1);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    resp.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    resp.status = (ret == 0) ? IpcStatus::Success : IpcStatus::ErrorExecutionFailed;
                    resp.driver_error = ret;
                } else {
                    // Host fallback simulation
                    auto t1 = std::chrono::high_resolution_clock::now();
                    resp.elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    resp.status = IpcStatus::Success;
                }
                break;
            }

            case IpcCmd::Shutdown: {
                if (is_hw) {
                    for (auto &pair : rcvd_fds) {
                        if (pair.second != pair.first) {
                            driver.mem_free(pair.second);
                        }
                    }
                    rcvd_fds.clear();
                }
                resp.status = IpcStatus::Success;
                send(m_worker_fd, &resp, sizeof(resp), 0);
                return;
            }

            default: {
                resp.status = IpcStatus::ErrorInvalidArg;
                break;
            }
        }

        send(m_worker_fd, &resp, sizeof(resp), 0);
    }
}

// =============================================================================
// WorkerPool Implementation
// =============================================================================

WorkerPool::WorkerPool() = default;

WorkerPool::~WorkerPool() {
    shutdown();
}

bool WorkerPool::start(uint16_t num_dies) {
    if (m_running) return true;

    auto &driver = EswinDriver::instance();
    driver.initialize();

    uint16_t available = driver.get_num_devices();
    if (available > 0 && num_dies > available) {
        num_dies = available;
    }
    if (num_dies == 0) num_dies = 1;

    m_channels.clear();
    for (uint16_t i = 0; i < num_dies; ++i) {
        auto ch = std::make_unique<WorkerChannel>(i);
        if (!ch->start()) {
            std::cerr << "[WorkerPool] Failed to launch worker daemon for die " << i << std::endl;
            shutdown();
            return false;
        }
        m_channels.push_back(std::move(ch));
    }

    m_running = true;
    return true;
}

void WorkerPool::shutdown() {
    for (auto &ch : m_channels) {
        if (ch) ch->shutdown();
    }
    m_channels.clear();
    m_running = false;
}

bool WorkerPool::load_model(uint16_t die_id, const std::string &path, ModelInfo &out_info) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::LoadModel;
    std::strncpy(req.model_path, path.c_str(), sizeof(req.model_path) - 1);

    IpcResponse resp{};
    if (m_channels[die_id]->send_request(req, resp)) {
        out_info.model_id = resp.model_id;
        out_info.num_inputs = resp.num_inputs;
        out_info.num_outputs = resp.num_outputs;
        for (int i = 0; i < 10; ++i) {
            out_info.input_sizes[i] = resp.input_sizes[i];
            out_info.output_sizes[i] = resp.output_sizes[i];
        }
        return true;
    }
    return false;
}

bool WorkerPool::load_model(uint16_t die_id, const std::string &path, uint32_t &out_model_id) {
    ModelInfo info;
    if (load_model(die_id, path, info)) {
        out_model_id = info.model_id;
        return true;
    }
    return false;
}

bool WorkerPool::unload_model(uint16_t die_id, uint32_t model_id) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::UnloadModel;
    req.model_id = model_id;

    IpcResponse resp{};
    return m_channels[die_id]->send_request(req, resp);
}

bool WorkerPool::prepare_buffer(uint16_t die_id, const DmaBuffer &buf) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::PrepareBuffer;
    req.inputs[0].mem_fd = buf.mem_fd();
    req.inputs[0].offset = 0;
    req.inputs[0].size   = buf.size();

    IpcResponse resp{};
    return m_channels[die_id]->send_request(req, resp);
}

bool WorkerPool::unprepare_buffer(uint16_t die_id, const DmaBuffer &buf) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::UnprepareBuffer;
    req.inputs[0].mem_fd = buf.mem_fd();
    req.inputs[0].offset = 0;
    req.inputs[0].size   = buf.size();

    IpcResponse resp{};
    return m_channels[die_id]->send_request(req, resp);
}

bool WorkerPool::submit_task(
    uint16_t die_id,
    uint32_t model_id,
    const std::vector<DmaBuffer*> &inputs,
    const std::vector<DmaBuffer*> &outputs,
    uint64_t *out_elapsed_ns
) {
    if (die_id >= m_channels.size() || !m_channels[die_id]) return false;

    IpcRequest req{};
    req.cmd = IpcCmd::SubmitTask;
    req.model_id = model_id;
    req.input_count = static_cast<uint8_t>(inputs.size());
    req.output_count = static_cast<uint8_t>(outputs.size());

    for (size_t i = 0; i < inputs.size() && i < 10; ++i) {
        if (inputs[i]) {
            req.inputs[i].mem_fd = inputs[i]->mem_fd();
            req.inputs[i].offset = 0;
            req.inputs[i].size   = inputs[i]->size();
        }
    }

    for (size_t i = 0; i < outputs.size() && i < 10; ++i) {
        if (outputs[i]) {
            req.outputs[i].mem_fd = outputs[i]->mem_fd();
            req.outputs[i].offset = 0;
            req.outputs[i].size   = outputs[i]->size();
        }
    }

    IpcResponse resp{};
    if (m_channels[die_id]->send_request(req, resp)) {
        if (out_elapsed_ns) {
            *out_elapsed_ns = resp.elapsed_ns;
        }
        return true;
    }
    return false;
}

} // namespace eswin
