// Copyright © 2024 ESWIN. All rights reserved.
//
// Beijing ESWIN Computing Technology Co., Ltd and its affiliated companies ("ESWIN") retain
// all intellectual property and proprietary rights in and to this software. Except as expressly
// authorized by ESWIN, no part of the software may be released, copied, distributed, reproduced,
// modified, adapted, translated, or created derivative work of, in whole or in part.

#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <cstring>
#include <filesystem>
#include <pthread.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <functional>
#include <sys/stat.h>
#include <sys/mman.h>
#include <math.h>
#include <thread>
#include <chrono>
#include <limits>
#include <queue>
#include <string>
#include <cctype>
#include <unordered_set>
#include "es_qwen2_client.hpp"
#include "json.h"
#include "half.h"

namespace fs = std::filesystem;

static int32_t client_log_level = 0;

#define client_log(func, line, level, fmt, ...)                                                                \
    do {                                                                                                       \
        if (level <= client_log_level)                                                                         \
            printf("[ES][Qwen2][%s](%s:%d) " fmt, level == 0 ? "ERROR" : "DEBUG", func, line, ##__VA_ARGS__); \
    } while (0)

#define ES_LOG_DBG(format, ...) client_log(__func__, __LINE__, 2, format, ##__VA_ARGS__)
#define ES_LOG_MSG(format, ...) client_log(__func__, __LINE__, 1, format, ##__VA_ARGS__)
#define ES_LOG_ERR(format, ...) client_log(__func__, __LINE__, 0, format, ##__VA_ARGS__)
#define ES_CLIENT_WARN(format, ...) client_log(__func__, __LINE__, 0, "[WARN] " format, ##__VA_ARGS__)

template <typename T>
void check(T result, char const* const func, ES_S32 const line)
{
    if (result) {
        fprintf(stderr, "%s(%d) error=%d \n", func, line, static_cast<unsigned ES_S32>(result));
        exit(EXIT_FAILURE);
    }
}

#define esCheckError(val) check((val), __func__, __LINE__)

#define DUMP_TOKEN_PREFIX(deviceId, maskVal, layer) \
    ("device_" + std::to_string(deviceId) + "_token_" + std::to_string(maskVal) + "_layer_" + std::to_string(layer))

#define DMABUF_ALIGN_SIZE 0x200000UL  // 2M
#define ALIGNMENT_UP(size, align) (((size) + ((align)-1)) & (~((align)-1)))

std::atomic<bool> Qwen2Client::mExited{false};

/* 0: ready to processing
   1/2: processing
   -1: not ready to processing or processing
*/
std::atomic<int32_t> Qwen2Client::mCompletedNum(-1);
std::atomic<bool> Qwen2Client::mStopped(false);

void esBindThread2Die(int32_t dieId)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (dieId == 0) {
        CPU_SET(0, &cpuset);
        CPU_SET(1, &cpuset);
        CPU_SET(2, &cpuset);
        CPU_SET(3, &cpuset);
    } else {
        CPU_SET(4, &cpuset);
        CPU_SET(5, &cpuset);
        CPU_SET(6, &cpuset);
        CPU_SET(7, &cpuset);
    }

    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

Qwen2Client::Qwen2Client()
    : mWordEmbeddings(nullptr),
    mDeviceMask(0),
    mDeviceStatus(0),
    mPostCnt(0),
    mTokenNotifyCnt(0)
{
    ES_LOG_DBG("Qwen2 Client construction done.\n");
}

void Qwen2Client::start(const std::string& configPath)
{
    init(configPath);

    for (int32_t i = 0; i < mConfig.devPrivAttr.size(); i++) {
        mTaskThreadVec.push_back(std::thread(&Qwen2Client::taskThreadFunc, this, i));
    }

    {
        // wait load model complete
        std::unique_lock<std::mutex> syncLock(mTaskMutex);
        mCompleteCv.wait(syncLock, [this](){ return mLoadedModels.load() == mTotalModels.load() || Qwen2Client::mExited.load(); });
    }

    ES_LOG_DBG("Qwen2 Client set config done.\n");
}

void Qwen2Client::init(const std::string& configPath)
{
    if (loadConfig(configPath)) {
        throw std::runtime_error("Config file is not exist.");
    }

    if (ES_SUCCESS != ES_NPU_SetLogLevel(mConfig.logLevel)) {
        ES_LOG_ERR("Set log level: %d failed\n", mConfig.logLevel);
        throw std::runtime_error("Set log level failed.");
    }

    if (mConfig.modelName == "qwen2") {
        mTokenizer = std::make_shared<QwenTokenizer>(mConfig.tiktokenFile, mConfig.specialTokens);
    } else if (mConfig.modelName == "deepseek_r1_distill_qwen") {
        mTokenizer = std::make_shared<DeepSeekTokenizer>(mConfig.tiktokenFile, mConfig.specialTokens);
    } else {
        ES_LOG_ERR("Not support model: %s\n", mConfig.modelName.c_str());
        throw std::runtime_error("Not support model.");
    }

    if (!mTokenizer) {
        throw std::runtime_error("Construct QwenTokenizer failed.");
    }

    mContexts.resize(mConfig.devPrivAttr.size());
    mStreams.resize(mConfig.devPrivAttr.size());
    mLoadedModels.store(0);
    mTotalModels.store(0);
    for (auto &block : mConfig.devPrivAttr) {
        mTotalModels.fetch_add(block.blockModelFiles.size());
        mTotalModels.fetch_add(mConfig.logits ? 1 : 0);
    }

    mBlockTasks.resize(mConfig.devPrivAttr.size());
    for (int32_t i = 0; i < mBlockTasks.size(); i++) {
        mBlockTasks[i].resize(mConfig.devPrivAttr[i].blockModelFiles.size());
    }
    mLogitsTask.resize(mConfig.devPrivAttr.size());

    for (int32_t i = 0; i < mConfig.devPrivAttr.size(); i++) {
        mDeviceMask.fetch_or(0x1ULL << i);
        mTokenMask.emplace_back(std::make_shared<std::atomic<int32_t>>(0));
        mTokenScores.emplace_back(std::vector<TokenIdScore>(mConfig.vocabSize));
    }
}

Qwen2Client::~Qwen2Client() { }

void Qwen2Client::exitClient()
{
    if (Qwen2Client::mCompletedNum.load() != -1) {
        std::unique_lock<std::mutex> taskLock(mTaskMutex);
        Qwen2Client::mStopped.store(true);
        Qwen2Client::mExited.store(true);
        mTaskCv.notify_all();
    } else {
        std::unique_lock<std::mutex> taskLock(mTaskMutex);
        Qwen2Client::mExited.store(true);
        mTaskCv.notify_all();
    }

    for (auto &taskTread : mTaskThreadVec) {
        if (taskTread.joinable()) {
            taskTread.join();
        }
    }
}

ES_S32 Qwen2Client::releaseTaskBuffers(int32_t idx)
{
    MODEL_TASK_S *blockTask = &mBlockTasks[idx].front();
    if (blockTask->isPrepared) {
        esCheckError(ES_NPU_Unprepare(&blockTask->task.inputFd[IN_TOKEN_IDX]));
        esCheckError(ES_SYS_Munmap(blockTask->inputMapAddr[IN_TOKEN_IDX], blockTask->task.inputFd[IN_TOKEN_IDX].size));
        esCheckError(ES_SYS_MemFree(blockTask->task.inputFd[IN_TOKEN_IDX].memFd));

        esCheckError(ES_NPU_Unprepare(&blockTask->task.outputFd[OUT_HIDDEN_IDX]));
        esCheckError(ES_SYS_Munmap(blockTask->outputMapAddr[OUT_HIDDEN_IDX], blockTask->task.outputFd[OUT_HIDDEN_IDX].size));
        esCheckError(ES_SYS_MemFree(blockTask->task.outputFd[OUT_HIDDEN_IDX].memFd));
    } else {
        return ES_SUCCESS;
    }

    for (auto& blockTask : mBlockTasks[idx]) {
        if (blockTask.isPrepared) {
            esCheckError(ES_NPU_Unprepare(&blockTask.task.inputFd[IN_K_CACHE_IDX]));
            esCheckError(ES_SYS_Munmap(blockTask.inputMapAddr[IN_K_CACHE_IDX], blockTask.task.inputFd[IN_K_CACHE_IDX].size));
            esCheckError(ES_SYS_MemFree(blockTask.task.inputFd[IN_K_CACHE_IDX].memFd));

            esCheckError(ES_NPU_Unprepare(&blockTask.task.inputFd[IN_V_CACHE_IDX]));
            esCheckError(ES_SYS_Munmap(blockTask.inputMapAddr[IN_V_CACHE_IDX], blockTask.task.inputFd[IN_V_CACHE_IDX].size));
            esCheckError(ES_SYS_MemFree(blockTask.task.inputFd[IN_V_CACHE_IDX].memFd));

            esCheckError(ES_NPU_Unprepare(&blockTask.task.inputFd[IN_MASK_IDX]));
            esCheckError(ES_SYS_Munmap(blockTask.inputMapAddr[IN_MASK_IDX], blockTask.task.inputFd[IN_MASK_IDX].size));
            esCheckError(ES_SYS_MemFree(blockTask.task.inputFd[IN_MASK_IDX].memFd));
        } else {
            return ES_SUCCESS;
        }
    }

    if ((mConfig.logits) && (mBlockTasks[idx].back().isPrepared)) {
        esCheckError(ES_NPU_Unprepare(&mLogitsTask[idx].task.outputFd[0]));
        esCheckError(ES_SYS_Munmap(mLogitsTask[idx].outputMapAddr[0], mLogitsTask[idx].task.outputFd[0].size));
        esCheckError(ES_SYS_MemFree(mLogitsTask[idx].task.outputFd[0].memFd));
    }

    return ES_SUCCESS;
}

ES_S32 Qwen2Client::dumpData(const void* address, const size_t size, const std::string& file)
{
    std::string outputPath = std::string("dump/") + file;
    std::ofstream outputFile(outputPath, std::ios::binary);
    if (!outputFile.is_open()) {
        ES_LOG_ERR("Failed to open file: %s\n", outputPath.c_str());
        return ES_NPU_ERROR_MODEL_NOT_FOUND;
    }

    outputFile.write((const char*)address, size);
    outputFile.close();

    ES_LOG_DBG("\n===> Data dumped to file: %s\n", outputPath.c_str());
    return ES_SUCCESS;
}

ES_S32 Qwen2Client::loadConfig(const std::string& configPath)
{
    std::ifstream configFile(configPath, std::ifstream::binary);
    if (!configFile.is_open()) {
        return ES_NPU_ERROR_MODEL_NOT_FOUND;
    }

    Json::Value root;
    configFile >> root;

    const Json::Value modelName = root["model_name"];
    mConfig.modelName = modelName.asString();

    const Json::Value modelDir = root["model_dir"];
    mConfig.modelDir = modelDir.asString();

    const Json::Value logits = root["logits"];
    mConfig.logits = logits.asBool();

    mConfig.tiktokenFile = root["tokenizer_file"].asString();
    mConfig.embeddingFile = root["embedding_file"].asString();

    const Json::Value modelList = root["model_list"];
    for (const auto& elem : modelList) {
        DevicePrivAttr privAttr;
        privAttr.devId = elem["device_id"].asInt();
        privAttr.vocabOffset = elem["vocab_offset"].asInt();
        privAttr.dataDump = elem["data_dump"].asInt();
        privAttr.dataTokenInterval = CLIENT_INTERVAL_S(elem["data_dump_token"][0].asInt(), elem["data_dump_token"][1].asInt());
        privAttr.perfEnable = elem["perf_enable"].asInt();
        privAttr.opDump = elem["op_dump"].asInt();
        privAttr.opTokenInterval = CLIENT_INTERVAL_S(elem["op_dump_token"][0].asInt(), elem["op_dump_token"][1].asInt());
        privAttr.opLayerInterval = CLIENT_INTERVAL_S(elem["op_dump_layer"][0].asInt(), elem["op_dump_layer"][1].asInt());
        privAttr.opListInterval = CLIENT_INTERVAL_S(elem["op_dump_list"][0].asInt(), elem["op_dump_list"][1].asInt());
        privAttr.opType = elem["op_dump_type"].asInt();

        std::vector<std::string> modelNameList;
        for (const auto& model : elem["block_models"]) {
            privAttr.blockModelFiles.push_back(model.asString());
        }

        if (privAttr.blockModelFiles.empty()) {
            ES_LOG_ERR("There is no block model(device: %d) in %s\n", elem["device_id"].asInt(), mConfig.modelDir.c_str());
            return ES_NPU_ERROR_MODEL_NOT_FOUND;
        }

        if (mConfig.logits) {
            privAttr.logitsModelFile = elem["logits_models"].asString();
        }
        mConfig.devPrivAttr.emplace_back(privAttr);
    }

    mConfig.vocabSize = root["vocab_size"].asInt();
    if (mConfig.devPrivAttr.size() == 1) {
        mConfig.vocabSizePerDie = mConfig.vocabSize;
    } else {
        mConfig.vocabSizePerDie = mConfig.devPrivAttr[1].vocabOffset;
    }
    mConfig.numHead = root["num_head"].asInt();
    mConfig.embeddingSize = root["embedding_size"].asInt();
    mConfig.embeddingDim = root["embedding_dim"].asInt();
    mConfig.embeddingDim1 = root["embedding_dim1"].asInt();
    mConfig.embeddingDim2 = root["embedding_dim2"].asInt();
    mConfig.tokenNum = root["token_num"].asInt();
    mConfig.tokenN1 = root["token_n1"].asInt();
    mConfig.tokenN2 = root["token_n2"].asInt();
    mConfig.precision = root["precision"].asInt();

    mConfig.language = root["language"].asString();

    const Json::Value specialTokens = root["special_tokens"];
    for (const auto& token : specialTokens) {
        mConfig.specialTokens.push_back(token.asString());
    }

    const Json::Value eosList = root["eos_list"];
    for (const auto& eos : eosList) {
        mConfig.eosList.push_back(eos.asString());
    }

    mConfig.responseMaxLen = root["response_max_len"].asInt();
    mConfig.responseLineLen = root["response_line_len"].asInt();
    mConfig.logLevel = root["log_level"].asInt();
    mConfig.randomCnt = root["random_cnt"].asInt();
    mConfig.repeatLen = root["repeat_len"].asInt();
    mConfig.repeatCnt = root["repeat_cnt"].asInt();
    mConfig.modelTest = root["model_test"].asInt();
    mConfig.repetitionPenalty = root["repetition_penalty"].asFloat();
    if (root.isMember("repeat_window")) {
        mConfig.repeatWindow = root["repeat_window"].asInt();
    } else {
        mConfig.repeatWindow = 64;
    }
    mConfig.topK = root["top_k"].asInt();
    mConfig.doSample = root["do_sample"].asBool();
    mConfig.topP = root["top_p"].asFloat();
    mConfig.temperature = root["temperature"].asFloat();

    /* doSample: topK > 1; topK no more than output size
       Not doSample: set topK 1, just using argmax */
    if (mConfig.doSample) {
        if (mConfig.topK == 1) {
            ES_LOG_MSG("when doSample, topK must greater than 1 but sets 1, make doSample false\n");
            mConfig.doSample = false;
        } else if (mConfig.topK > mConfig.vocabSizePerDie) {
            mConfig.topK = mConfig.vocabSizePerDie;
        }
    } else {
        if (mConfig.topK != 1) {
            ES_LOG_MSG("When not doSample, topK must equal 1 but sets not 1, make topK equal 1\n");
            mConfig.topK = 1;
        }
    }

    const Json::Value& prompts = root["prompts"];
    for (Json::ValueConstIterator it = prompts.begin(); it != prompts.end(); ++it) {
        int32_t key = std::stoi(it.key().asString());
        mConfig.prompts[key] = it->asString();
    }

    const Json::Value& rolePrompts = root["role_prompts"];
    for (Json::ValueConstIterator it = rolePrompts.begin(); it != rolePrompts.end(); ++it) {
        int32_t key = std::stoi(it.key().asString());
        mConfig.rolePrompts[key] = it->asString();
    }

    client_log_level = mConfig.logLevel;

    return ES_SUCCESS;
}

ES_S32 Qwen2Client::loadEmbedding()
{
    if (mWordEmbeddings != nullptr) {
        delete[] mWordEmbeddings;
        mWordEmbeddings = nullptr;
    }

    try {
        mWordEmbeddings = new fp16_t[(size_t)mConfig.vocabSize * mConfig.embeddingSize];
    } catch (const std::bad_alloc& e) {
        ES_LOG_ERR("alloc memory for word embedding failed: %s\n", e.what());
        return ES_FAILURE;
    }

    std::ifstream file(mConfig.embeddingFile, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        ES_LOG_ERR("Failed to open embedding file: %s\n", mConfig.embeddingFile.c_str());
        delete[] mWordEmbeddings;
        mWordEmbeddings = nullptr;
        return ES_FAILURE;
    }

    std::streampos fileSize = file.tellg();
    std::streamsize expectedSize = (std::streamsize)mConfig.vocabSize * mConfig.embeddingSize * sizeof(fp16_t);
    if (fileSize != expectedSize) {
        ES_LOG_ERR("Invalid embedding file size: %lld, expected: %lld\n", (long long)fileSize, (long long)expectedSize);
        delete[] mWordEmbeddings;
        mWordEmbeddings = nullptr;
        return ES_FAILURE;
    }

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(mWordEmbeddings), fileSize);

    if (!file) {
        ES_LOG_ERR("Error reading embedding file: %s\n", mConfig.embeddingFile.c_str());
        delete[] mWordEmbeddings;
        mWordEmbeddings = nullptr;
        return ES_FAILURE;
    }

    file.close();

    return ES_SUCCESS;
}

void Qwen2Client::displayLoadModelProgress(int32_t current, int32_t total, double costTime)
{
    float progress = (float)current / total * 100.0f;
    std::string progressBar;

    progressBar = "\rLoading models: [";
    int pos = 50 * progress / 100.0;

    for (int i = 0; i < 50; ++i) {
        if (i < pos)
            progressBar += "=";
        else if (i == pos)
            progressBar += ">";
        else
            progressBar += " ";
    }

    progressBar += "] ";

    std::ostringstream progressFmt;
    progressFmt << std::setw(6) << std::setfill(' ') << std::fixed << std::setprecision(2) << progress;
    progressBar += progressFmt.str() + "%" + " ( " + std::to_string(costTime) + " seconds )";

    std::cout << progressBar;

    std::cout.flush();
}

ES_S32 Qwen2Client::initDevice(int32_t idx)
{
    uint16_t deviceNum = 0;

    esCheckError(ES_NPU_GetNumDevices(&deviceNum));
    if (mConfig.devPrivAttr[idx].devId >= deviceNum) {
        ES_LOG_ERR("deviceId(%d) is not less than deviceNum(%d)\n", mConfig.devPrivAttr[idx].devId, deviceNum);
        return ES_NPU_ERROR_DEV_NOT_FOUND;
    }

    ES_LOG_DBG("open device:%u\n", mConfig.devPrivAttr[idx].devId);
    esCheckError(ES_NPU_SetDevice(mConfig.devPrivAttr[idx].devId));

    esCheckError(ES_NPU_CreateContext(&mContexts[idx], mConfig.devPrivAttr[idx].devId));

    esCheckError(ES_NPU_CreateStream(&mStreams[idx]));

    return ES_SUCCESS;
}

ES_S32 Qwen2Client::deinitDevice(int32_t idx)
{
    esCheckError(ES_NPU_DestroyStream(mStreams[idx]));
    esCheckError(ES_NPU_DestroyContext(mContexts[idx]));
    esCheckError(ES_NPU_ReleaseDevice(mConfig.devPrivAttr[idx].devId));
    return ES_SUCCESS;
}

ES_S32 Qwen2Client::loadModel(int32_t idx)
{
    uint32_t modelId;

    auto start = std::chrono::high_resolution_clock::now();
    // load block models
    for (int32_t num = 0; num < mConfig.devPrivAttr[idx].blockModelFiles.size(); num++) {
        if (Qwen2Client::mExited.load()) {
            return ES_SUCCESS;
        }
        std::string model = mConfig.modelDir + "/" + mConfig.devPrivAttr[idx].blockModelFiles[num];

        esCheckError(ES_NPU_LoadModelFromFile(&modelId, model.c_str()));
        ES_LOG_DBG("load %d model:%s, modelId=%u\n", num, model.c_str(), modelId);

        esCheckError(prepareBlockTaskBuffers(idx, num, modelId));

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        displayLoadModelProgress(mLoadedModels.load() + 1, mTotalModels.load(), elapsed.count());
        mLoadedModels.fetch_add(1);
    }

    // load logits model
    if (mConfig.logits) {
        std::string model = mConfig.modelDir + "/" + mConfig.devPrivAttr[idx].logitsModelFile;
        esCheckError(ES_NPU_LoadModelFromFile(&modelId, model.c_str()));
        ES_LOG_DBG("load logits model:%s, modelId=%u\n", model.c_str(), modelId);

        esCheckError(prepareLogitsTaskBuffers(idx, modelId));

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        displayLoadModelProgress(mLoadedModels.load() + 1, mTotalModels.load(), elapsed.count());
        mLoadedModels.fetch_add(1);
    }

    return ES_SUCCESS;
}

ES_S32 Qwen2Client::unloadModel(int32_t idx)
{
    for (auto& blockTask : mBlockTasks[idx]) {
        if (blockTask.isPrepared) {
            esCheckError(ES_NPU_UnloadModel(blockTask.task.modelId));
        } else {
            break;
        }
    }

    if ((mConfig.logits) && (mBlockTasks[idx].back().isPrepared)) {
        esCheckError(ES_NPU_UnloadModel(mLogitsTask[idx].task.modelId));
    }

    releaseTaskBuffers(idx);

    if (idx == 0) {
        delete[] mWordEmbeddings;
        mWordEmbeddings = nullptr;
    }

    return ES_SUCCESS;
}

ES_S32 Qwen2Client::prepareTensorBuffer(NPU_TENSOR_S &tensor, ES_DEV_BUF_S &devBuf, void **pAddr, uint16_t devId, bool cache)
{
    ES_U64 memFd;
    SYS_CACHE_MODE_E cacheMode = cache ? SYS_CACHE_MODE_CACHED : SYS_CACHE_MODE_NOCACHE;
    const char *zone = (devId == 0) ? "mmz_nid_0_part_0" : "mmz_nid_1_part_0";
    ES_U64 bufferSize;

    bufferSize = (cache) ? tensor.bufferSize : ALIGNMENT_UP(tensor.bufferSize, DMABUF_ALIGN_SIZE);
    esCheckError(ES_SYS_MemAlloc(&memFd, cacheMode, "qwen2_demo", zone, bufferSize));
    *pAddr = ES_SYS_Mmap(memFd, tensor.bufferSize, cacheMode);
    if (!*pAddr) {
        ES_LOG_ERR("ES_SYS_Mmap failed!\n");
        return ES_FAILURE;
    }

    (void)memset(*pAddr, 0x00, tensor.bufferSize);
    if (cache) {
        esCheckError(ES_SYS_MemFlushCache(memFd));
    }

    devBuf.size = tensor.bufferSize;
    devBuf.offset = 0;
    devBuf.memFd = memFd;

    return ES_SUCCESS;
}

ES_S32 Qwen2Client::prepareKVCacheBuffers(uint32_t modelId, MODEL_TASK_S &modelTask, uint16_t devId)
{
    void* pData;
    ES_U64 memFd;
    ES_U64 kCacheSize = mConfig.precision * mConfig.numHead * mConfig.embeddingDim1 * mConfig.tokenNum * mConfig.embeddingDim2;
    ES_U64 vCacheSize = mConfig.precision * mConfig.numHead * mConfig.tokenN1 * mConfig.embeddingDim * mConfig.tokenN2;
    ES_U64 maskSize = MASK_BUFFER_SIZE;
    NPU_TENSOR_S tensor;

    // Prepare K cache buffer
    esCheckError(ES_NPU_GetInputTensorDesc(modelId, IN_K_CACHE_IDX, &tensor));
    if (tensor.bufferSize != kCacheSize) {
        ES_LOG_ERR("k cache size is not valid, %d!\n", tensor.bufferSize);
        return ES_FAILURE;
    }
    esCheckError(prepareTensorBuffer(tensor, modelTask.task.inputFd[IN_K_CACHE_IDX],
                                     &modelTask.inputMapAddr[IN_K_CACHE_IDX], devId, false));

    modelTask.task.outputFd[OUT_K_CACHE_IDX] = modelTask.task.inputFd[IN_K_CACHE_IDX];
    modelTask.outputMapAddr[OUT_K_CACHE_IDX] = modelTask.inputMapAddr[IN_K_CACHE_IDX];
    esCheckError(ES_NPU_Prepare(&modelTask.task.inputFd[IN_K_CACHE_IDX]));

    // Prepare V cache buffer
    esCheckError(ES_NPU_GetInputTensorDesc(modelId, IN_V_CACHE_IDX, &tensor));
    if (tensor.bufferSize != vCacheSize) {
        ES_LOG_ERR("v cache size is not valid, %d!\n", tensor.bufferSize);
        return ES_FAILURE;
    }
    esCheckError(prepareTensorBuffer(tensor, modelTask.task.inputFd[IN_V_CACHE_IDX],
                                     &modelTask.inputMapAddr[IN_V_CACHE_IDX], devId, false));

    modelTask.task.outputFd[OUT_V_CACHE_IDX] = modelTask.task.inputFd[IN_V_CACHE_IDX];
    modelTask.outputMapAddr[OUT_V_CACHE_IDX] = modelTask.inputMapAddr[IN_V_CACHE_IDX];
    esCheckError(ES_NPU_Prepare(&modelTask.task.inputFd[IN_V_CACHE_IDX]));

    ES_LOG_DBG("prepare KVCache buffers done.\n");

    return ES_SUCCESS;
}

ES_S32 Qwen2Client::prepareBlockTaskBuffers(int32_t idx, int32_t modelIdx, uint32_t modelId)
{
    MODEL_TASK_S &modelTask = mBlockTasks[idx][modelIdx];
    NPU_TENSOR_S tensor;
    int32_t numInputTensors, numOutputTensors;

    memset(&modelTask, 0, sizeof(MODEL_TASK_S));
    esCheckError(ES_NPU_GetNumInputTensors(modelId, &numInputTensors));
    if (numInputTensors != IN_MAX_IDX) {
        ES_LOG_ERR("Input tensor num is invalid from model, numInputTensors=%d\n", numInputTensors);
        return ES_NPU_ERROR_INVALID_SIZE;
    }
    modelTask.task.inputFdNum = numInputTensors;

    esCheckError(ES_NPU_GetNumOutputTensors(modelId, &numOutputTensors));
    if (numOutputTensors != OUT_MAX_IDX) {
        ES_LOG_ERR("Output tensor num is invalid from model, numOutputTensors=%d\n", numOutputTensors);
        return ES_NPU_ERROR_INVALID_SIZE;
    }
    modelTask.task.outputFdNum = numOutputTensors;

    // prepare KV Cache
    esCheckError(prepareKVCacheBuffers(modelId, modelTask, mConfig.devPrivAttr[idx].devId));

    // prepare mask buffer
    esCheckError(ES_NPU_GetInputTensorDesc(modelId, IN_MASK_IDX, &tensor));
    esCheckError(prepareTensorBuffer(tensor, modelTask.task.inputFd[IN_MASK_IDX],
                                     &modelTask.inputMapAddr[IN_MASK_IDX], mConfig.devPrivAttr[idx].devId, false));
    esCheckError(ES_NPU_Prepare(&modelTask.task.inputFd[IN_MASK_IDX]));

    // prepare token buffer
    if (modelIdx == 0) {
        esCheckError(ES_NPU_GetInputTensorDesc(modelId, IN_TOKEN_IDX, &tensor));
        esCheckError(prepareTensorBuffer(tensor, modelTask.task.inputFd[IN_TOKEN_IDX],
                                         &modelTask.inputMapAddr[IN_TOKEN_IDX], mConfig.devPrivAttr[idx].devId, false));
        esCheckError(ES_NPU_Prepare(&modelTask.task.inputFd[IN_TOKEN_IDX]));

        esCheckError(ES_NPU_GetOutputTensorDesc(modelId, OUT_HIDDEN_IDX, &tensor));
        esCheckError(prepareTensorBuffer(tensor, modelTask.task.outputFd[OUT_HIDDEN_IDX],
                                         &modelTask.outputMapAddr[OUT_HIDDEN_IDX], mConfig.devPrivAttr[idx].devId, false));
        esCheckError(ES_NPU_Prepare(&modelTask.task.outputFd[OUT_HIDDEN_IDX]));

        // actually output size
        modelTask.task.outputFd[OUT_HIDDEN_IDX].size = modelTask.task.inputFd[IN_TOKEN_IDX].size;
    } else {
        modelTask.task.inputFd[IN_TOKEN_IDX] = mBlockTasks[idx][modelIdx - 1].task.outputFd[OUT_HIDDEN_IDX];
        modelTask.inputMapAddr[IN_TOKEN_IDX] = mBlockTasks[idx][modelIdx - 1].outputMapAddr[OUT_HIDDEN_IDX];

        modelTask.task.outputFd[OUT_HIDDEN_IDX] = mBlockTasks[idx][modelIdx - 1].task.inputFd[IN_TOKEN_IDX];
        modelTask.outputMapAddr[OUT_HIDDEN_IDX] = mBlockTasks[idx][modelIdx - 1].inputMapAddr[IN_TOKEN_IDX];
    }

    modelTask.task.modelId = modelId;
    modelTask.isPrepared = true;

    return ES_SUCCESS;
}

ES_S32 Qwen2Client::prepareLogitsTaskBuffers(int32_t idx, uint32_t modelId)
{
    NPU_TENSOR_S tensor;
    int32_t numInputTensors, numOutputTensors;

    memset(&mLogitsTask[idx], 0x00, sizeof(MODEL_TASK_S));
    esCheckError(ES_NPU_GetNumInputTensors(modelId, &numInputTensors));
    if (numInputTensors > 1) {
        ES_LOG_ERR("logits task numInput tensor can not align: %d\n", numInputTensors);
    }

    mLogitsTask[idx].task.inputFdNum = numInputTensors;
    mLogitsTask[idx].task.inputFd[0] = mBlockTasks[idx].back().task.outputFd[OUT_HIDDEN_IDX];
    mLogitsTask[idx].inputMapAddr[0] = mBlockTasks[idx].back().outputMapAddr[OUT_HIDDEN_IDX];

    esCheckError(ES_NPU_GetNumOutputTensors(modelId, &numOutputTensors));
    mLogitsTask[idx].task.outputFdNum = numOutputTensors;
    for (int32_t i = 0; i < numOutputTensors; i++) {
        esCheckError(ES_NPU_GetOutputTensorDesc(modelId, i, &tensor));
        // actually output size
        tensor.bufferSize = mConfig.vocabSizePerDie * sizeof(fp16_t);
        esCheckError(prepareTensorBuffer(tensor, mLogitsTask[idx].task.outputFd[i], &mLogitsTask[idx].outputMapAddr[i], mConfig.devPrivAttr[idx].devId, true));
        esCheckError(ES_NPU_Prepare(&mLogitsTask[idx].task.outputFd[i]));
    }

    mLogitsTask[idx].task.modelId = modelId;

    return ES_SUCCESS;
}

ES_U64 Qwen2Client::calcKCacheOutputAddrOffset(int32_t num)
{
    ES_U64 offset;

    return mConfig.precision * mConfig.embeddingDim2 * num;
}

ES_U64 Qwen2Client::calcVCacheOutputAddrOffset(int32_t num)
{
    int32_t dim0 = num / mConfig.tokenN2;
    int32_t dim1 = num % mConfig.tokenN2;

    return mConfig.precision * (mConfig.tokenN2 * mConfig.embeddingDim * dim0 + dim1);
}

void Qwen2Client::runModel(MODEL_TASK_S& modelTask)
{
    // Submit model task
    esCheckError(ES_NPU_Submit(&modelTask.task, 1));
}

void Qwen2Client::runModelAsync(MODEL_TASK_S& modelTask, npu_stream stream)
{
    // Submit model task async
    esCheckError(ES_NPU_SubmitAsync(&modelTask.task, 1, stream));
}

static int32_t argmax(int16_t *data, int32_t size, int16_t *max)
{
    int32_t pos = 0;
    *max = data[0];

    for (int32_t i = 1; i < size; i++) {
        if (data[i] > *max) {
            pos = i;
            *max = data[i];
        }
    }

    return pos;
}

prob_data_t Qwen2Client::getTopK(int32_t idx, int32_t wordPosInVocab, const int32_t& maskVal, const std::vector<int32_t>& outputIds)
{
    prob_data_t topK;
    std::chrono::_V2::system_clock::time_point startTime, modelStartTime, lmStartTime, postStartTime, endTime;

    if (mConfig.devPrivAttr[idx].perfEnable) {
        startTime = std::chrono::high_resolution_clock::now();
    }

    // Get embeddings for next word with bounds validation
    if (mWordEmbeddings == nullptr) {
        ES_LOG_ERR("mWordEmbeddings is NULL! Cannot retrieve embedding.\n");
        return 0;
    }
    if (wordPosInVocab < 0 || wordPosInVocab >= mConfig.vocabSize) {
        ES_LOG_ERR("wordPosInVocab (%d) out of range (0..%d), clamping to 0\n", wordPosInVocab, mConfig.vocabSize - 1);
        wordPosInVocab = 0;
    }
    int32_t wordStart = wordPosInVocab * mConfig.embeddingSize;
    int32_t wordEnd = wordStart + mConfig.embeddingSize;
    std::vector<fp16_t> wordEmbedding(mWordEmbeddings + wordStart, mWordEmbeddings + wordEnd);

    // prepare input token
    memcpy(mBlockTasks[idx].front().inputMapAddr[IN_TOKEN_IDX],
           wordEmbedding.data(), wordEmbedding.size() * sizeof(fp16_t));

    if (mConfig.devPrivAttr[idx].dataDump && checkInterval(maskVal, mConfig.devPrivAttr[idx].dataTokenInterval)) {
        std::string fileName = "token_" + std::to_string(maskVal) + std::string("_wordembedding.data");
        dumpData(mBlockTasks[idx].front().inputMapAddr[IN_TOKEN_IDX], mBlockTasks[idx].front().task.inputFd[IN_TOKEN_IDX].size, fileName);
    }

    if (mConfig.devPrivAttr[idx].perfEnable) {
        modelStartTime = std::chrono::high_resolution_clock::now();
    }

    // run block models
    for (int32_t modelIdx = 0; modelIdx < mBlockTasks[idx].size(); modelIdx++) {
        // adjust output KVCache offset
        mBlockTasks[idx][modelIdx].task.outputFd[OUT_K_CACHE_IDX].offset = calcKCacheOutputAddrOffset(maskVal);
        mBlockTasks[idx][modelIdx].task.outputFd[OUT_V_CACHE_IDX].offset = calcVCacheOutputAddrOffset(maskVal);
        // fill mask
        *static_cast<int16_t*>(mBlockTasks[idx][modelIdx].inputMapAddr[IN_MASK_IDX]) = maskVal;

        ES_LOG_DBG("start runmodel mask=%d, modelIdx=%d, modelId: %d\n", maskVal, modelIdx, mBlockTasks[idx][modelIdx].task.modelId);
        if (mConfig.devPrivAttr[idx].opDump &&
            checkInterval(maskVal, mConfig.devPrivAttr[idx].opTokenInterval) &&
            checkInterval(modelIdx, mConfig.devPrivAttr[idx].opLayerInterval)) {
            NPU_DUMP_CONF_S dumpConf;
            std::string dumpPath = "/tmp/" + std::to_string(idx) + "_" + std::to_string(maskVal) + "_" + std::to_string(modelIdx);
            char dumpStr[64] = {0};

            snprintf(dumpStr, 64, "%d-%d", mConfig.devPrivAttr[idx].opListInterval.min, mConfig.devPrivAttr[idx].opListInterval.max);
            dumpConf.subModelId = mConfig.devPrivAttr[idx].opType;
            dumpConf.dumpList = dumpStr;

            esCheckError(ES_NPU_ConfigDumpInfo(mBlockTasks[idx][modelIdx].task.modelId, dumpPath.c_str(), &dumpConf, 1));
            esCheckError(ES_NPU_EnableDump(mBlockTasks[idx][modelIdx].task.modelId));
        }

        // run block model
        if (mConfig.devPrivAttr[idx].opDump || mConfig.devPrivAttr[idx].dataDump) {
            runModel(mBlockTasks[idx][modelIdx]);
            if (mConfig.devPrivAttr[idx].opDump &&
                checkInterval(maskVal, mConfig.devPrivAttr[idx].opTokenInterval) &&
                checkInterval(modelIdx, mConfig.devPrivAttr[idx].opLayerInterval)) {
                esCheckError(ES_NPU_DisableDump(mBlockTasks[idx][modelIdx].task.modelId));
            }
            if (mConfig.devPrivAttr[idx].dataDump && checkInterval(maskVal, mConfig.devPrivAttr[idx].dataTokenInterval)) {
                std::string fileName = DUMP_TOKEN_PREFIX(idx, maskVal, modelIdx) + std::string("_token.data");
                dumpData(mBlockTasks[idx][modelIdx].outputMapAddr[OUT_HIDDEN_IDX],
                        mBlockTasks[idx][modelIdx].task.outputFd[OUT_HIDDEN_IDX].size, fileName);
                fileName = DUMP_TOKEN_PREFIX(idx, maskVal, modelIdx) + std::string("_kcache.data");
                dumpData(mBlockTasks[idx][modelIdx].outputMapAddr[OUT_K_CACHE_IDX],
                        mBlockTasks[idx][modelIdx].task.outputFd[OUT_K_CACHE_IDX].size, fileName);
                fileName = DUMP_TOKEN_PREFIX(idx, maskVal, modelIdx) + std::string("_vcache.data");
                dumpData(mBlockTasks[idx][modelIdx].outputMapAddr[OUT_V_CACHE_IDX],
                        mBlockTasks[idx][modelIdx].task.outputFd[OUT_V_CACHE_IDX].size, fileName);
            }
        } else {
            runModelAsync(mBlockTasks[idx][modelIdx], mStreams[idx]);
        }
    }

    if (mConfig.devPrivAttr[idx].perfEnable) {
        esCheckError(ES_NPU_SynchronizeStream(mStreams[idx]));
        lmStartTime = std::chrono::high_resolution_clock::now();
    }

    // run logits model
    if (mConfig.logits) {
        if (mConfig.devPrivAttr[idx].dataDump && checkInterval(maskVal, mConfig.devPrivAttr[idx].dataTokenInterval)) {
            runModel(mLogitsTask[idx]);
            std::string fileName = "device" + std::to_string(idx) + "_token_" + std::to_string(maskVal) + std::string("_logits_out.data");
            dumpData(mLogitsTask[idx].outputMapAddr[0], mLogitsTask[idx].task.outputFd[0].size, fileName);
        } else {
            runModelAsync(mLogitsTask[idx], mStreams[idx]);
        }
    }
    esCheckError(ES_NPU_SynchronizeStream(mStreams[idx]));

    if (mConfig.devPrivAttr[idx].perfEnable) {
        postStartTime = std::chrono::high_resolution_clock::now();
    }

    // doSample
    uint16_t *logitData = (uint16_t*)mLogitsTask[idx].outputMapAddr[0];
    if (mConfig.doSample) {
        // logits to mTokenScores
        for (int32_t i = 0; i < mConfig.vocabSizePerDie; i++) {
            mTokenScores[idx][i] = TokenIdScore(i, half_float::detail::half2float(logitData[i]));
        }

        // repetition penalty
        if (mConfig.repetitionPenalty != 1.0f) {
            repetitionPenalty(idx, mTokenScores[idx], outputIds, mConfig.repetitionPenalty);
        }

        // adjust logits distribution
        if (mConfig.doSample && (mConfig.temperature != 1.0f) && (mConfig.temperature != 0.0f)) {
            float invTemp = 1.0 / mConfig.temperature;
            for (int32_t i = 0; i < mConfig.vocabSizePerDie; i++) {
                mTokenScores[idx][i].score *= invTemp;
            }
        }

        // select topK
        selectTopK(mTokenScores[idx], mConfig.topK, mConfig.vocabSizePerDie);
    } else {
        // repetition penalty
        if (mConfig.repetitionPenalty != 1.0f) {
            repetitionPenalty(idx, logitData, outputIds, mConfig.repetitionPenalty);
        }

        // topK=1
        int16_t maxVal;
        mTokenScores[idx][0].id = argmax((int16_t*)mLogitsTask[idx].outputMapAddr[0], mConfig.vocabSizePerDie, &maxVal);
        mTokenScores[idx][0].score = half_float::detail::half2float(maxVal);
    }

    esCheckError(ES_SYS_MemFlushCache(mLogitsTask[idx].task.outputFd[0].memFd));

    {
        // notify the status to other thread.
        std::unique_lock<std::mutex> postSynclock(mPostSyncMutex);
        mDeviceStatus.fetch_or(0x1ULL << idx);
        mPostSyncCv.notify_all();
    }

    {
        std::unique_lock<std::mutex> postSynclock(mPostSyncMutex);
        mPostSyncCv.wait(postSynclock, [this]() { return !(mDeviceStatus.load() ^ mDeviceMask.load()) || Qwen2Client::mStopped.load(); });
        mPostCnt.fetch_add(1);
        // clear device status when all device finished
        if (mPostCnt.load() == mConfig.devPrivAttr.size()) {
            mPostCnt.store(0);
            mDeviceStatus.store(0);
        }
    }

    // only die0 calc postprocess
    if (idx == 0) {
        if (mConfig.doSample) {
            // merge 2 die topKs and select new topK
            for (int32_t i = 0; i < mConfig.devPrivAttr.size(); i++) {
                for (int32_t j = 0; j < mConfig.topK; j++) {
                    mTokenScores[idx][i * mConfig.topK + j] = mTokenScores[i][j];
                    mTokenScores[idx][i * mConfig.topK + j].id += mConfig.devPrivAttr[i].vocabOffset;
                }
            }
            selectTopK(mTokenScores[idx], mConfig.topK, mConfig.topK * mConfig.devPrivAttr.size());

            // softmax(mConfig.topK), then based on topP to get candidateNum
            int32_t candidateNum = mConfig.topK;
            if (mConfig.topP > 0 && mConfig.topP < 1.0f) {
                softmaxInPlace(mTokenScores[idx], mConfig.topK);
                candidateNum = samplingTopP(mTokenScores[idx], mConfig.topP, mConfig.topK);
            }

            // softmax(candidateNum)
            softmaxInPlace(mTokenScores[idx], candidateNum);
            std::vector<float> probs(candidateNum);
            for (int32_t i = 0; i < candidateNum; i++) {
                probs[i] = mTokenScores[idx][i].score;
            }

            // random get tokenId by probs[candidateNum]
            thread_local std::random_device rd;
            thread_local std::mt19937 gen(rd());
            std::discrete_distribution<> dist(probs.data(), probs.data() + candidateNum);
            topK = mTokenScores[idx][dist(gen)].id;
        } else {
            // topK=1
            auto maxIt = std::max_element(mTokenScores.begin(), mTokenScores.end(), [](std::vector<TokenIdScore> &t1, std::vector<TokenIdScore> &t2){
                return t1[0].score < t2[0].score;
            });
            int32_t maxIndex = std::distance(mTokenScores.begin(), maxIt);
            topK = (*maxIt)[0].id + mConfig.devPrivAttr[maxIndex].vocabOffset;
        }
        mTokenId = topK;

        // sync to other die thread
        {
            std::unique_lock<std::mutex> tokenSynclock(mTokenIdSyncMutex);
            mTokenIdDone.store(true);
            mTokenIdSyncCv.notify_all();
        }
    } else {
        // sync from die0 thread
        std::unique_lock<std::mutex> tokenSynclock(mTokenIdSyncMutex);
        mTokenIdSyncCv.wait(tokenSynclock, [this]() { return mTokenIdDone.load(); });
        mTokenNotifyCnt.fetch_add(1);
        if (mTokenNotifyCnt.load() == (mConfig.devPrivAttr.size() - 1)) {
            mTokenNotifyCnt.store(0);
            mTokenIdDone.store(false);
        }
        topK = mTokenId;
    }

    if (mConfig.devPrivAttr[idx].perfEnable) {
        endTime = std::chrono::high_resolution_clock::now();
        // corresponding: Preprocess, block models, lm model, Postprocess, Total time
        double t[5];
        t[0] = std::chrono::duration_cast<std::chrono::microseconds>(modelStartTime - startTime).count() / 1000.0;
        t[1] = std::chrono::duration_cast<std::chrono::microseconds>(lmStartTime - modelStartTime).count() / 1000.0;
        t[2] = std::chrono::duration_cast<std::chrono::microseconds>(postStartTime - lmStartTime).count() / 1000.0;
        t[3] = std::chrono::duration_cast<std::chrono::microseconds>(endTime - postStartTime).count() / 1000.0;
        t[4] = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000.0;

        static double sum[5] = {0.0f};
        const char* name[5] = {"Preprocess", "block models", "lm model", "Postprocess", "Total"};
        printf("(Avg time) dev_idx: %d, token: %d", idx, maskVal);
        for (int i = 0; i < 5; i++) {
            sum[i] += t[i];
            printf(", %s: %.2fms", name[i], sum[i] / maskVal);
        }
        printf("\n");
    }

    ES_LOG_MSG("mask=%d topK=%hu\n", maskVal, topK);

    // Return topK value
    return topK;
}

std::vector<std::string> Qwen2Client::splitUtf8(const std::string& str)
{
    std::vector<std::string> result;
    std::string charBuffer;

    const size_t UTF8_1BYTE = 0x80;
    const size_t UTF8_2BYTES_LEAD = 0x6;
    const size_t UTF8_3BYTES_LEAD = 0xe;
    const size_t UTF8_4BYTES_LEAD = 0x1e;

    for (std::size_t i = 0; i < str.size();) {
        charBuffer.clear();
        unsigned char lead = str[i];
        size_t charLen = 0;

        if (lead < UTF8_1BYTE) {
            charLen = 1;
        } else if ((lead >> 5) == UTF8_2BYTES_LEAD) {
            charLen = 2;
        } else if ((lead >> 4) == UTF8_3BYTES_LEAD) {
            charLen = 3;
        } else if ((lead >> 3) == UTF8_4BYTES_LEAD) {
            charLen = 4;
        } else {
            ++i;
            continue;
        }

        charBuffer = str.substr(i, charLen);
        result.push_back(charBuffer);
        ES_LOG_MSG("split word:%s\n", charBuffer.c_str());
        i += charLen;
    }

    return result;
}

std::vector<std::string> Qwen2Client::splitEn(const std::string& input)
{
    std::vector<std::string> tokens;
    std::stringstream ss(input);
    std::string token;
    while (ss >> token) {
        std::string word;
        std::string punctuation;
        for (char c : token) {
            if (ispunct(static_cast<unsigned char>(c))) {
                if (!word.empty()) {
                    tokens.push_back(word);
                    word.clear();
                }
                punctuation += c;
                tokens.push_back(punctuation);
                punctuation.clear();
            } else {
                word += c;
            }
        }
        if (!word.empty()) {
            tokens.push_back(word);
        }
    }

    for (auto& wordStr : tokens) {
        ES_LOG_MSG("wordStr=%s\n", wordStr.c_str());
    }

    return tokens;
}

std::vector<std::string> Qwen2Client::splitSentence(const std::string& input)
{
    if (mConfig.language == "zh") {
        return splitUtf8(input);
    } else {
        return splitEn(input);
    }
}

bool Qwen2Client::checkDuplicateWords(const std::string& sentence)
{
    std::vector<std::string> words = splitSentence(sentence);
    if (words.size() < mConfig.repeatLen * 2) {
        return false;
    }

    std::unordered_map<std::string, int32_t> sequenceCounts;

    for (size_t i = 0; i <= words.size() - mConfig.repeatLen; ++i) {
        std::string subString;
        for (size_t j = i; j < i + mConfig.repeatLen; ++j) {
            subString += words[j];
        }
        sequenceCounts[subString]++;

        if (sequenceCounts[subString] == mConfig.repeatCnt) {
            ES_LOG_MSG("\nduplicate str:%s\n", subString.c_str());
            return true;
        }
    }

    return false;
}

bool Qwen2Client::checkTokenRepetition(const std::vector<int32_t>& outputIds)
{
    if (outputIds.size() < 4) {
        return false;
    }

    // 1. Single token repeated consecutively (e.g. 4 times: "Wait", "Wait", "Wait", "Wait")
    const size_t singleRepeatThreshold = 4;
    if (outputIds.size() >= singleRepeatThreshold) {
        int32_t lastToken = outputIds.back();
        bool allSame = true;
        for (size_t k = outputIds.size() - singleRepeatThreshold; k < outputIds.size(); ++k) {
            if (outputIds[k] != lastToken) {
                allSame = false;
                break;
            }
        }
        if (allSame) {
            ES_CLIENT_WARN("Single token repetition loop detected (token %d repeated %zu times)\n", lastToken, singleRepeatThreshold);
            return true;
        }
    }

    // 2. N-gram pattern repetition (periods 2 to 16, repeated 3 times)
    // E.g. [A, B, A, B, A, B] or [Wait, -, Wait, -, Wait, -]
    const size_t minRepeats = 3;
    const size_t maxPeriod = 16;
    for (size_t period = 2; period <= maxPeriod; ++period) {
        if (outputIds.size() >= period * minRepeats) {
            bool isCycle = true;
            size_t baseOffset = outputIds.size() - period;
            for (size_t r = 1; r < minRepeats; ++r) {
                size_t prevOffset = baseOffset - r * period;
                for (size_t p = 0; p < period; ++p) {
                    if (outputIds[prevOffset + p] != outputIds[baseOffset + p]) {
                        isCycle = false;
                        break;
                    }
                }
                if (!isCycle) break;
            }
            if (isCycle) {
                ES_CLIENT_WARN("Cycle repetition loop detected (period=%zu, repeated %zu times)\n", period, minRepeats);
                return true;
            }
        }
    }

    return false;
}

ES_S32 Qwen2Client::taskThreadFunc(int32_t idx)
{
    ES_S32 ret;
    bool master = (idx == 0) ? true : false;

    esBindThread2Die(mConfig.devPrivAttr[idx].devId);

    ret = initDevice(idx);
    if (ES_SUCCESS != ret) {
        ES_LOG_ERR("init device(%d) error: 0x%x\n", mConfig.devPrivAttr[idx].devId, ret);
        return ES_FAILURE;
    }

    ret = loadModel(idx);
    if (ES_SUCCESS != ret) {
        ES_LOG_ERR("Load model error on device %d\n", mConfig.devPrivAttr[idx].devId);
        Qwen2Client::mExited.store(true);
        std::unique_lock<std::mutex> syncLock(mTaskMutex);
        mCompleteCv.notify_all();
        return ES_FAILURE;
    }

    if (master) {
        if (loadEmbedding() != ES_SUCCESS) {
            ES_LOG_ERR("Load embedding failed: %s\n", mConfig.embeddingFile.c_str());
            Qwen2Client::mExited.store(true);
            std::unique_lock<std::mutex> syncLock(mTaskMutex);
            mCompleteCv.notify_all();
            return ES_FAILURE;
        }
    }

    if (mLoadedModels.load() == mTotalModels.load() || Qwen2Client::mExited.load()) {
        std::unique_lock<std::mutex> syncLock(mTaskMutex);
        mCompleteCv.notify_one();
    }

    while (!Qwen2Client::mExited.load()) {
        {
            std::unique_lock<std::mutex> syncLock(mTaskMutex);
            mTaskCv.wait(syncLock, []() { return Qwen2Client::mExited.load() || (Qwen2Client::mCompletedNum.load() == 0); });
            if (Qwen2Client::mExited.load()) {
                break;
            }
        }

        mTokenMask[idx]->store(0);
        for (int32_t i = 0; i < mBlockTasks[idx].size(); i++) {
            memset(mBlockTasks[idx][i].inputMapAddr[IN_V_CACHE_IDX], 0x00, mBlockTasks[idx][i].task.inputFd[IN_V_CACHE_IDX].size);
            memset(mBlockTasks[idx][i].inputMapAddr[IN_K_CACHE_IDX], 0x00, mBlockTasks[idx][i].task.inputFd[IN_K_CACHE_IDX].size);
        }

        std::string response = generate(idx, master);

        {
            std::unique_lock<std::mutex> syncLock(mTaskMutex);
            Qwen2Client::mCompletedNum.fetch_add(1);
            if (Qwen2Client::mCompletedNum.load() == mConfig.devPrivAttr.size()) {
                if (Qwen2Client::mStopped.load()) {
                    Qwen2Client::mStopped.store(false);
                }
                mCompleteCv.notify_all();
            }
        }
    }

    unloadModel(idx);
    deinitDevice(idx);

    return ES_SUCCESS;
}

inline bool Qwen2Client::checkInterval(int32_t value, const CLIENT_INTERVAL_S &interval)
{
    return (value >= interval.min && value <= interval.max);
}

std::string Qwen2Client::generate(int32_t idx, bool master)
{
    std::string newSentence = "";

    std::vector<int32_t> outputIds(mSentencePosInVocab);
    int32_t wordPosInVocab = mSentencePosInVocab[0];
    std::string inputWord;
    bool endFlag = false;
    int32_t lineLength = 0;
    int32_t i;

    auto startTime = std::chrono::high_resolution_clock::now();

    ES_LOG_MSG("mSentencePosInVocab size=%u\n", mSentencePosInVocab.size());
    if (mSentencePosInVocab.size() == 0) {
        ES_LOG_ERR("master: %d, input sentence is empty, no response\n", master);
        return newSentence;
    }

    int32_t maxAllowedLen = std::min(mConfig.responseMaxLen, mConfig.tokenNum - 1);
    bool inThinking = false;
    int thinkingCount = 0;
    bool thinkingLimitReached = false;
    for (i = 0; i < maxAllowedLen; ++i) {
        inputWord = mTokenizer->decode(std::vector<int32_t>(1, wordPosInVocab));

        // stop to generate
        if (i >= mSentencePosInVocab.size()) {
            if (wordPosInVocab == 151643 || wordPosInVocab == 151645) { // <|endoftext|> or <|im_end|>
                ES_LOG_MSG("Encountered EOS token ID %d, generation finished.\n", wordPosInVocab);
                endFlag = true;
            }
            if (!endFlag) {
                for (const auto& endStr : mConfig.eosList) {
                    if (endStr == inputWord) {
                        ES_LOG_MSG("遇到结束词%s, 生成过程结束。", endStr.c_str());
                        endFlag = true;
                        break;
                    }
                }
            }
            if (endFlag) {
                // add the word count of eos
                mTokenMask[idx]->fetch_add(1);
                Qwen2Client::mStopped.store(true);
                {
                    std::unique_lock<std::mutex> postSynclock(mPostSyncMutex);
                    mPostSyncCv.notify_all();
                }
                {
                    std::unique_lock<std::mutex> tokenSynclock(mTokenIdSyncMutex);
                    mTokenIdDone.store(true);
                    mTokenIdSyncCv.notify_all();
                }
                break;
            }
        }

        // Skip the original input sentence and print response one by one.
        if (i >= mSentencePosInVocab.size()) {
            outputIds.emplace_back(wordPosInVocab);
            if (master) {
                bool isThinkStart = (wordPosInVocab == 151648 || inputWord == "<think>" || inputWord.find("<think>") != std::string::npos);
                bool isThinkEnd   = (wordPosInVocab == 151649 || inputWord == "</think>" || inputWord.find("</think>") != std::string::npos);

                if (isThinkStart) {
                    inThinking = true;
                    thinkingCount = 0;
                    thinkingLimitReached = false;
                    if (mConfig.showTokens) {
                        std::cout << "\033[90m[#" << wordPosInVocab << ":<think>]\033[0m" << std::flush;
                    }
                    if (mConfig.showThinking) {
                        std::cout << "\033[90m\n[Thinking: \033[0m" << std::flush;
                    } else {
                        std::cout << "\033[90m[Thinking...]\033[0m" << std::flush;
                    }
                } else if (isThinkEnd) {
                    if (inThinking) {
                        inThinking = false;
                        if (mConfig.showThinking) {
                            std::cout << "\033[90m]\n\n\033[0m" << std::flush;
                        } else {
                            std::cout << "\r              \r" << std::flush;
                        }
                    }
                    if (mConfig.showTokens) {
                        std::cout << "\033[90m[#" << wordPosInVocab << ":</think>]\033[0m\n\n" << std::flush;
                    }
                } else if (inThinking) {
                    thinkingCount++;
                    if (mConfig.maxThinkingTokens > 0 && thinkingCount >= mConfig.maxThinkingTokens && !thinkingLimitReached) {
                        thinkingLimitReached = true;
                        inThinking = false;
                        if (mConfig.showThinking) {
                            std::cout << "\033[33m\n[Thinking limit of " << mConfig.maxThinkingTokens
                                      << " tokens reached, concluding reasoning]\n\n\033[0m" << std::flush;
                        } else {
                            std::cout << "\033[33m\n[Thinking limit reached]\n\033[0m" << std::flush;
                        }
                    } else if (mConfig.showThinking) {
                        if (mConfig.showTokens) {
                            std::cout << "\033[90m[#" << wordPosInVocab << ":\"" << inputWord << "\"]\033[0m" << std::flush;
                        } else {
                            std::cout << "\033[90m" << inputWord << "\033[0m" << std::flush;
                        }
                    }
                } else {
                    if (mConfig.showTokens) {
                        std::cout << "[#" << wordPosInVocab << ":\"" << inputWord << "\"]" << std::flush;
                    } else {
                        std::cout << inputWord << std::flush;
                    }
                }
            }

            // Check for duplicate / repetition loop in output tokens
            if (checkTokenRepetition(outputIds)) {
                ES_CLIENT_WARN("Repetition loop detected! Stopping generation early.\n");
                mTokenMask[idx]->fetch_add(1);
                Qwen2Client::mStopped.store(true);
                {
                    std::unique_lock<std::mutex> postSynclock(mPostSyncMutex);
                    mPostSyncCv.notify_all();
                }
                {
                    std::unique_lock<std::mutex> tokenSynclock(mTokenIdSyncMutex);
                    mTokenIdDone.store(true);
                    mTokenIdSyncCv.notify_all();
                }
                break;
            }
        }

        if (lineLength >= mConfig.responseLineLen) {
            if (mConfig.language == "zh") {
                std::cout << std::endl;
            }
            lineLength = 0;
        }
        lineLength += inputWord.size();
        newSentence += inputWord;

        ES_LOG_MSG("Input sentence_pos(%d) vocab_pos(%d): %s\n", i, wordPosInVocab, inputWord.c_str());

        // Prevent KV Cache buffer overflow
        if (i >= mConfig.tokenNum - 1) {
            ES_CLIENT_WARN("Reached KV cache capacity limit (%d), stopping.\n", mConfig.tokenNum);
            Qwen2Client::mStopped.store(true);
            {
                std::unique_lock<std::mutex> postSynclock(mPostSyncMutex);
                mPostSyncCv.notify_all();
            }
            {
                std::unique_lock<std::mutex> tokenSynclock(mTokenIdSyncMutex);
                mTokenIdDone.store(true);
                mTokenIdSyncCv.notify_all();
            }
            break;
        }

        // Get next word token with top_k value, skip the original input sentence.
        wordPosInVocab = getTopK(idx, wordPosInVocab, i, outputIds);

        if (i < mSentencePosInVocab.size() - 1) {
            wordPosInVocab = mSentencePosInVocab[i + 1];
        }
        ES_LOG_MSG("Output mask(%d) sentence_pos(%d) vocab_pos(%d)\n", i, i + 1, wordPosInVocab);

        mTokenMask[idx]->fetch_add(1);

        // check user stop generate or quit it.
        if (Qwen2Client::mStopped.load()) {
            break;
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000.0;
    double tps = 1000 / (totalTime / mTokenMask[idx]->load());

    if (master) {
        std::cout << std::endl <<
                     "--------------------------------------------------------" << std::endl <<
                     "Throughput: " << tps << "tokens/s" << std::endl <<
                     "--------------------------------------------------------" << std::endl;
    }

    return newSentence;
}

std::map<int32_t, std::string> Qwen2Client::getOptionalPrompts()
{
    std::map<int32_t, std::string> randomMap;

    if (mConfig.randomCnt >= mConfig.prompts.size()) {
        return mConfig.prompts;
    }

    std::vector<std::map<int32_t, std::string>::const_iterator> iters;
    for (auto it = mConfig.prompts.begin(); it != mConfig.prompts.end(); ++it) {
        iters.push_back(it);
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(iters.begin(), iters.end(), g);

    for (int32_t i = 0; i < mConfig.randomCnt; ++i) {
        randomMap[i] = iters[i]->second;
    }

    return randomMap;
}

std::map<int32_t, std::string> Qwen2Client::getOptionalRolePrompts()
{
    return mConfig.rolePrompts;
}

std::string Qwen2Client::genResponse(const std::string& rolePrompt, const std::string& prompts)
{
    std::vector<ChatMessage> messages;

    messages.emplace_back(ChatMessage(ChatMessage::ROLE_SYSTEM, rolePrompt));
    messages.emplace_back(ChatMessage(ChatMessage::ROLE_USER, prompts));

    auto startTime = std::chrono::high_resolution_clock::now();

    mSentencePosInVocab = mTokenizer->encode_messages(messages, mConfig.responseMaxLen);
    {
        std::unique_lock<std::mutex> syncLock(mTaskMutex);
        Qwen2Client::mCompletedNum.store(0);
        Qwen2Client::mStopped.store(false);
        mTaskCv.notify_all();
    }

    {
        std::unique_lock<std::mutex> syncLock(mTaskMutex);
        mCompleteCv.wait(syncLock, [this]() { return (Qwen2Client::mCompletedNum.load() == mConfig.devPrivAttr.size()) || Qwen2Client::mExited.load(); });
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

    if (mConfig.devPrivAttr[0].perfEnable) {
        std::cout << "Times:" << (duration / 1000.0 / 1000.0) << "(s)" << std::endl;
    }

    return "";
}

void Qwen2Client::repetitionPenalty(int32_t idx, uint16_t* logitData, const std::vector<int32_t>& outputIds, float penalty)
{
    if (penalty != 1.0f) {
        // Use sliding window for repetition penalty to avoid penalizing all common grammatical tokens
        int32_t windowSize = mConfig.repeatWindow > 0 ? mConfig.repeatWindow : 64;
        size_t startPos = outputIds.size() > (size_t)windowSize ? outputIds.size() - windowSize : 0;
        std::unordered_set<int> uniqueOutputIds(outputIds.begin() + startPos, outputIds.end());

        int32_t vocabOffset = mConfig.devPrivAttr[idx].vocabOffset;
        int32_t vocabEnd = vocabOffset + mConfig.vocabSizePerDie;

        for (auto ids : uniqueOutputIds) {
            // Out of range check: ids must be in [vocabOffset, vocabEnd)
            if (ids < vocabOffset || ids >= vocabEnd) {
                continue;
            }

            int32_t localIdx = ids - vocabOffset;
            float tmp = half_float::detail::half2float(logitData[localIdx]);
            if (tmp > 0) {
                tmp /= penalty;
            } else {
                tmp *= penalty;
            }
            logitData[localIdx] = half_float::detail::float2half<std::round_to_nearest>(tmp);
        }
    }
}

void Qwen2Client::repetitionPenalty(int32_t idx, std::vector<TokenIdScore> &tokenScores, const std::vector<int32_t>& outputIds, float penalty)
{
    if (penalty != 1.0f) {
        // Use sliding window for repetition penalty
        int32_t windowSize = mConfig.repeatWindow > 0 ? mConfig.repeatWindow : 64;
        size_t startPos = outputIds.size() > (size_t)windowSize ? outputIds.size() - windowSize : 0;
        std::unordered_set<int> uniqueOutputIds(outputIds.begin() + startPos, outputIds.end());

        int32_t vocabOffset = mConfig.devPrivAttr[idx].vocabOffset;
        int32_t vocabEnd = vocabOffset + mConfig.vocabSizePerDie;

        for (auto ids : uniqueOutputIds) {
            // Out of range check: ids must be in [vocabOffset, vocabEnd)
            if (ids < vocabOffset || ids >= vocabEnd) {
                continue;
            }

            int32_t localIdx = ids - vocabOffset;
            float tmp = half_float::detail::half2float(tokenScores[localIdx].score);
            if (tmp > 0) {
                tmp /= penalty;
            } else {
                tmp *= penalty;
            }
            tokenScores[localIdx].score = half_float::detail::float2half<std::round_to_nearest>(tmp);
        }
    }
}

void Qwen2Client::selectTopK(std::vector<TokenIdScore> &tokenScores, int32_t topK, int32_t numTokens)
{
    if (numTokens > topK) {
        std::nth_element(tokenScores.data(), tokenScores.data() + topK, tokenScores.data() + numTokens, std::greater<TokenIdScore>());
    }
}

void Qwen2Client::softmaxInPlace(std::vector<TokenIdScore> &tokenScores, int32_t numTokens)
{
    float maxScore = std::max_element(tokenScores.data(), tokenScores.data() + numTokens)->score;
    float sum = 0.f;
    for (int32_t i = 0; i < numTokens; i++) {
        float s = std::exp(tokenScores[i].score - maxScore);
        tokenScores[i].score = s;
        sum += s;
    }
    float invSum = 1.f / sum;
    for (int32_t i = 0; i < numTokens; i++) {
        tokenScores[i].score *= invSum;
    }
}

int32_t Qwen2Client::samplingTopP(std::vector<TokenIdScore> &tokenScores, float topP, int32_t numTokens)
{
    TokenIdScore *first = tokenScores.data();
    TokenIdScore *last = tokenScores.data() + numTokens;
    while (first + 1 < last) {
        float pivotScore = (last - 1)->score;
        TokenIdScore *mid =
            std::partition(first, last - 1, [pivotScore](const TokenIdScore &x) { return x.score > pivotScore; });
        std::swap(*mid, *(last - 1));

        float prefixSum =
            std::accumulate(first, mid, 0.f, [](float sum, const TokenIdScore &x) { return sum + x.score; });
        if (prefixSum >= topP) {
            last = mid;
        } else if (prefixSum + mid->score < topP) {
            first = mid + 1;
            topP -= prefixSum + mid->score;
        } else {
            return mid + 1 - first;
        }
    }

    return last - first;
}
