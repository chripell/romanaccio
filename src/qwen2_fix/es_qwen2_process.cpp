#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <mutex>
#include <signal.h>
#include <queue>
#include <future>
#include <chrono>
#include <thread>
#include "half.h"
#include "es_qwen2_client.hpp"

static Qwen2Client gQwen2Client;
static int gMaskValue = 0;
static bool gInitFlag = false;
static bool gStartFlag = false;
static std::queue<int32_t> gTokenQueue;
static std::mutex gTokenMutex;

static constexpr ES_S32 begin_of_sentence = 151646;
static constexpr ES_S32 end_of_sentence = 151643;
static constexpr ES_S32 user_tok = 151644;
static constexpr ES_S32 assistant_tok = 151645;
static constexpr ES_S32 think_tok = 151648;
static constexpr ES_S32 kSuccess = 0;
static constexpr ES_S32 kFailed = 1;
static bool gIsDualDie = false;

#ifdef __cplusplus
extern "C" {
#endif

int graph_compute(void *input, int input_len, void *output, int output_len);
int set_token_ids(int token_id);
int load_model();

#ifdef __cplusplus
}
#endif

template <typename T>
void check(T result, char const* const func, ES_S32 const line)
{
    if (result) {
        fprintf(stderr, "%s(%d) error=%d \n", func, line, static_cast<unsigned ES_S32>(result));
        exit(EXIT_FAILURE);
    }
}

#define esCheckError(val) check((val), __func__, __LINE__)
#define DEEPSEEK_CONFIG_FILE "/usr/local/bin/config.json"

__attribute__((destructor))
void lib_deinit() {
    printf("Dynamic library unloaded: Cleaning up resources...\n");
    for (int32_t idx = 0; idx < gQwen2Client.getConfig().devPrivAttr.size(); idx++) {
        gQwen2Client.unloadModel(idx);
        gQwen2Client.deinitDevice(idx);
    }
}

static void convertF16ToF32(const uint16_t* outputMapAddr, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint16_t h = outputMapAddr[i];
        output[i] = half_float::detail::half2float(h);
    }
}

static int runInference(void* input, int input_len, int maskValue) {
    auto func = [funcName = __func__, input, input_len, maskValue](ES_S32 idx) {
        memcpy(gQwen2Client.getBlockTask()[idx].front().inputMapAddr[IN_TOKEN_IDX],
            input, input_len * sizeof(int16_t));

        for (int32_t modelIdx = 0; modelIdx < gQwen2Client.getBlockTask()[idx].size(); modelIdx++) {
            gQwen2Client.getBlockTask()[idx][modelIdx].task.outputFd[OUT_K_CACHE_IDX].offset =
                gQwen2Client.calcKCacheOutputAddrOffset(maskValue);
            gQwen2Client.getBlockTask()[idx][modelIdx].task.outputFd[OUT_V_CACHE_IDX].offset =
                gQwen2Client.calcVCacheOutputAddrOffset(maskValue);

            *static_cast<int16_t*>(gQwen2Client.getBlockTask()[idx][modelIdx].inputMapAddr[IN_MASK_IDX]) =
                maskValue;

            gQwen2Client.runModelAsync(gQwen2Client.getBlockTask()[idx][modelIdx],
                                    gQwen2Client.getStream()[idx]);
        }

        if (gQwen2Client.getConfig().logits) {
            gQwen2Client.runModelAsync(gQwen2Client.getLogitsTask()[idx],
                                    gQwen2Client.getStream()[idx]);
        }


        esCheckError(ES_NPU_SynchronizeStream(gQwen2Client.getStream()[idx]));
        esCheckError(ES_SYS_MemFlushCache(gQwen2Client.getLogitsTask()[idx].task.outputFd[0].memFd));

        return kSuccess;
    };

    if (gIsDualDie) {
        auto futureDie0 = std::async(std::launch::async, func, 0);
        auto futureDie1 = std::async(std::launch::async, func, 1);
        int result0 = futureDie0.get();
        int result1 = futureDie1.get();
    } else {
        func(0);
    }

    return kSuccess;
}

static inline void resetKVCache()
{
    for (int32_t idx = 0; idx < gQwen2Client.getConfig().devPrivAttr.size(); idx++) {
        for (int32_t i = 0; i < (int32_t)gQwen2Client.getBlockTask()[idx].size(); ++i) {
            memset(gQwen2Client.getBlockTask()[idx][i].inputMapAddr[IN_V_CACHE_IDX], 0x00,
                gQwen2Client.getBlockTask()[idx][i].task.inputFd[IN_V_CACHE_IDX].size);
            memset(gQwen2Client.getBlockTask()[idx][i].inputMapAddr[IN_K_CACHE_IDX], 0x00,
                gQwen2Client.getBlockTask()[idx][i].task.inputFd[IN_K_CACHE_IDX].size);
        }
    }
}

int set_token_ids(int token_id)
{
    // printf("--token_id:%d\n", token_id);
    {
        std::lock_guard<std::mutex> lock(gTokenMutex);
        gTokenQueue.push(token_id);
    }

    return kSuccess;
}

int load_model()
{
    if (!gInitFlag) {
        const std::string configFile = DEEPSEEK_CONFIG_FILE;
        gQwen2Client.init(configFile);
        gIsDualDie = gQwen2Client.getConfig().devPrivAttr.size() == 1 ? false : true;

        auto func = [](ES_S32 idx) {
            ES_S32 ret;
            esBindThread2Die(gQwen2Client.getConfig().devPrivAttr[idx].devId);
            ret = gQwen2Client.initDevice(idx);
            if (ES_SUCCESS != ret) {
                printf("init device(%d) error: 0x%x\n", gQwen2Client.getConfig().devPrivAttr[idx].devId, ret);
                return kFailed;
            }
            ret = gQwen2Client.loadModel(idx);
            if (ES_SUCCESS != ret) {
                printf("loadModel failed\n");
                return kFailed;
            }

            return kSuccess;
        };

        if (gIsDualDie) {
            auto threadDie0 = std::thread(func, 0);
            auto threadDie1 = std::thread(func, 1);
            if (threadDie0.joinable())
                threadDie0.join();
            if (threadDie1.joinable())
                threadDie1.join();
        } else {
            func(0);
        }

        gInitFlag = true;
    }

    return kSuccess;
}

int graph_compute(void *input, int input_len, void *output, int output_len) {
    if (!input || input_len <= 0 || !output || !output_len) {
        return kFailed;
    }

    int32_t tok;
    {
        std::lock_guard<std::mutex> lock(gTokenMutex);
        if (gTokenQueue.empty()) {
            return kFailed;
        }

        tok = gTokenQueue.front();
        gTokenQueue.pop();
        if (tok == begin_of_sentence) {
            gStartFlag = false;
            // printf("<｜begin▁of▁sentence｜>\n");
            resetKVCache();
            gMaskValue = 0;
        }
    }

    // printf("tok:%d, gMaskValue:%d\n", tok, gMaskValue);
    runInference(input, input_len, gMaskValue);
    if (gMaskValue >= (gQwen2Client.getConfig().responseMaxLen - 1)) {
        std::vector<float> logits(gQwen2Client.getConfig().vocabSizePerDie * 2, 0);
        logits[end_of_sentence] = 1000000.0;
        memcpy(output, logits.data(), logits.size() * sizeof(float));
        gMaskValue = 0;
        printf("Excceed maxium tokens. Reset maskValue:%d\n", gMaskValue);
        return kSuccess;
    }

    if (gStartFlag) {
        for (int32_t idx = 0; idx < gQwen2Client.getConfig().devPrivAttr.size(); idx++) {
            float* output_float = static_cast<float*>(output);
            convertF16ToF32((uint16_t*)gQwen2Client.getLogitsTask()[idx].outputMapAddr[0],
                            output_float + (idx * gQwen2Client.getConfig().vocabSizePerDie),
                            gQwen2Client.getConfig().vocabSizePerDie);
        }
    } else {
        if (tok == assistant_tok) {
            gStartFlag = true;
        }

        std::vector<float> logits(gQwen2Client.getConfig().vocabSizePerDie * 2, 0);
        logits[think_tok] = 1000000.0;
        memcpy(output, logits.data(), logits.size() * sizeof(float));
    }

    gMaskValue++;

    return kSuccess;
}