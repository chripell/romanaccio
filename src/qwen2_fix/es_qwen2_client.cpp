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

    initTokenFlags();
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
    esCheckError(ES_SYS_MemFlushCache(mLogitsTask[idx].task.outputFd[0].memFd));

    if (mConfig.devPrivAttr[idx].perfEnable) {
        postStartTime = std::chrono::high_resolution_clock::now();
    }

    // Unified float32 logit sanitization, token guard masking, and repetition penalty
    // for BOTH sampling (doSample=true) and greedy (doSample=false) modes
    uint16_t *logitData = (uint16_t*)mLogitsTask[idx].outputMapAddr[0];
    int32_t activeTopK = mConfig.doSample ? std::max(1, mConfig.topK) : 1;

    for (int32_t i = 0; i < mConfig.vocabSizePerDie; i++) {
        float val = half_float::detail::half2float(logitData[i]);
        if (std::isnan(val) || std::isinf(val)) {
            val = -1e9f;
        }
        mTokenScores[idx][i] = TokenIdScore(i, val);
    }

    // Apply token sanitizer, <think>/</think> state guards, and frequency repetition penalty
    repetitionPenalty(idx, mTokenScores[idx], outputIds, mConfig.repetitionPenalty);

    if (mConfig.doSample && (mConfig.temperature != 1.0f) && (mConfig.temperature > 0.0f)) {
        float invTemp = 1.0f / mConfig.temperature;
        for (int32_t i = 0; i < mConfig.vocabSizePerDie; i++) {
            if (mTokenScores[idx][i].score > -1e8f) {
                mTokenScores[idx][i].score *= invTemp;
            }
        }
    }

    // Select sorted topK per die (topK=1 for greedy mode)
    selectTopK(mTokenScores[idx], activeTopK, mConfig.vocabSizePerDie);

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
            // merge 2 die topKs and select new topK (sorted descending)
            for (int32_t i = 0; i < mConfig.devPrivAttr.size(); i++) {
                for (int32_t j = 0; j < activeTopK; j++) {
                    mTokenScores[idx][i * activeTopK + j] = mTokenScores[i][j];
                    mTokenScores[idx][i * activeTopK + j].id += mConfig.devPrivAttr[i].vocabOffset;
                }
            }
            selectTopK(mTokenScores[idx], activeTopK, activeTopK * mConfig.devPrivAttr.size());

            // Convert topK logits to probabilities ONCE via softmax
            softmaxInPlace(mTokenScores[idx], activeTopK);

            // Apply Top-P (nucleus) filtering on sorted probabilities
            int32_t candidateNum = activeTopK;
            if (mConfig.topP > 0.0f && mConfig.topP < 1.0f) {
                candidateNum = samplingTopP(mTokenScores[idx], mConfig.topP, activeTopK);
            }

            // Apply Min-P relative threshold (drop noisy tail tokens < 15% of top-1 probability)
            float minProbThreshold = mTokenScores[idx][0].score * 0.15f;
            while (candidateNum > 1 && mTokenScores[idx][candidateNum - 1].score < minProbThreshold) {
                candidateNum--;
            }

            // Pass true probabilities directly to discrete_distribution (do NOT call softmaxInPlace a second time!)
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
    bool thinkingDone = false;
    int forceThinkEndStep = 0; // 1 = feed newline, 2 = feed </think>, 3 = feed \n\n
    for (i = 0; i < maxAllowedLen; ++i) {
        if (i >= (int32_t)mSentencePosInVocab.size()) {
            // Multi-step clean conclusion of capped <think> block:
            // Ensures KV-cache sees `.\n</think>\n\n` on BOTH dies before generating the final answer.
            if (forceThinkEndStep == 1) {
                forceThinkEndStep = 2;
                wordPosInVocab = 198; // "\n"
                outputIds.emplace_back(wordPosInVocab);
                inputWord = "\n";
                goto feed_token_to_npu;
            } else if (forceThinkEndStep == 2) {
                forceThinkEndStep = 3;
                thinkingLimitReached = true;
                inThinking = false;
                thinkingDone = true;
                wordPosInVocab = 151649; // "</think>"
                if (master) {
                    if (mConfig.showThinking) {
                        std::cout << "\033[33m\n[Thinking limit of " << mConfig.maxThinkingTokens
                                  << " tokens reached, concluding reasoning]\033[0m\n\n" << std::flush;
                    } else {
                        std::cout << "\r              \r\033[33m[Thinking limit reached]\033[0m\n\n" << std::flush;
                    }
                    if (mConfig.showTokens) {
                        std::cout << "\033[90m[#151649:</think>][#271:\"\\n\\n\"]\033[0m\n\n" << std::flush;
                    }
                }
                outputIds.emplace_back(wordPosInVocab);
                inputWord = "</think>";
                goto feed_token_to_npu;
            } else if (forceThinkEndStep == 3) {
                forceThinkEndStep = 0;
                wordPosInVocab = 271; // "\n\n"
            }
        }

        inputWord = mTokenizer->decode(std::vector<int32_t>(1, wordPosInVocab));

        // stop to generate
        if (i >= mSentencePosInVocab.size()) {
            if (wordPosInVocab == 151643 || wordPosInVocab == 151645 || wordPosInVocab == 151647) { // <|endoftext|> or <|im_end|> or <|EOT|>
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

            // Track thinking state across BOTH threads so Die 0 and Die 1 stay in lockstep
            bool isThinkStart = (!thinkingDone && !inThinking &&
                                 (wordPosInVocab == 151648 || inputWord == "<think>" || inputWord.find("<think>") != std::string::npos));
            bool isThinkEnd   = (wordPosInVocab == 151649 || inputWord == "</think>" || inputWord.find("</think>") != std::string::npos);

            if (isThinkStart) {
                inThinking = true;
                thinkingCount = 0;
                thinkingLimitReached = false;
                if (master) {
                    if (mConfig.showTokens) {
                        std::cout << "\033[90m[#" << wordPosInVocab << ":<think>]\033[0m" << std::flush;
                    }
                    if (mConfig.showThinking) {
                        std::cout << "\033[90m\n[Thinking: \033[0m" << std::flush;
                    } else {
                        std::cout << "\033[90m[Thinking...]\033[0m" << std::flush;
                    }
                }
            } else if (isThinkEnd) {
                if (inThinking) {
                    inThinking = false;
                    thinkingDone = true;
                    if (master) {
                        if (mConfig.showThinking) {
                            std::cout << "\033[90m]\n\n\033[0m" << std::flush;
                        } else {
                            std::cout << "\r              \r" << std::flush;
                        }
                    }
                }
                if (master && mConfig.showTokens) {
                    std::cout << "\033[90m[#" << wordPosInVocab << ":</think>]\033[0m\n\n" << std::flush;
                }
            } else if (inThinking) {
                thinkingCount++;
                if (master && mConfig.showThinking) {
                    if (mConfig.showTokens) {
                        std::cout << "\033[90m[#" << wordPosInVocab << ":\"" << inputWord << "\"]\033[0m" << std::flush;
                    } else {
                        std::cout << "\033[90m" << inputWord << "\033[0m" << std::flush;
                    }
                }
                if (mConfig.maxThinkingTokens > 0 && thinkingCount >= mConfig.maxThinkingTokens && !thinkingLimitReached) {
                    // Conclude reasoning at the end of the current sentence/line (or within +32 tokens max)
                    bool atSentenceBoundary = (inputWord.find('.') != std::string::npos ||
                                               inputWord.find('!') != std::string::npos ||
                                               inputWord.find('?') != std::string::npos ||
                                               inputWord.find('\n') != std::string::npos);
                    bool hardCapReached = (thinkingCount >= mConfig.maxThinkingTokens + 32);
                    if (atSentenceBoundary || hardCapReached) {
                        bool endsWithNewline = (!inputWord.empty() && inputWord.back() == '\n');
                        forceThinkEndStep = endsWithNewline ? 2 : 1;
                    }
                }
            } else {
                thinkingDone = true;
                // Suppress redundant leading whitespace right after forced </think>\n\n
                bool skipPrint = (outputIds.size() >= mSentencePosInVocab.size() + 2 &&
                                  outputIds[outputIds.size() - 2] == 151649 &&
                                  wordPosInVocab == 271);
                if (master && !skipPrint) {
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

feed_token_to_npu:

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

static constexpr uint8_t TOK_FLAG_FORBIDDEN      = 0x01;
static constexpr uint8_t TOK_FLAG_CJK_OR_EMOJI   = 0x02;
static constexpr uint8_t TOK_FLAG_MATH_OR_QUOTE  = 0x04;
static constexpr uint8_t TOK_FLAG_EXEMPT_PENALTY = 0x08;
static constexpr uint8_t TOK_FLAG_HAS_NEWLINE    = 0x10;
static constexpr uint8_t TOK_FLAG_HAS_BOLD       = 0x20;
static constexpr uint8_t TOK_FLAG_CLOSING_BOLD   = 0x40;
static constexpr uint8_t TOK_FLAG_SENTENCE_END   = 0x80;

static bool isValidUtf8AndClassify(const std::string& s, bool& hasCjkOrEmoji)
{
    hasCjkOrEmoji = false;
    size_t i = 0;
    while (i < s.size()) {
        uint8_t b0 = static_cast<uint8_t>(s[i]);
        if (b0 < 0x80) {
            if (b0 < 0x20 && b0 != '\n' && b0 != '\r' && b0 != '\t') {
                return false;
            }
            i += 1;
        } else if ((b0 >> 5) == 0x06) {
            if (i + 1 >= s.size()) return false;
            uint8_t b1 = static_cast<uint8_t>(s[i + 1]);
            if ((b1 & 0xC0) != 0x80 || b0 < 0xC2) return false;
            // Allow Latin-1 / Latin-Extended-A (0xC2..0xC5), flag other scripts
            if (b0 > 0xC5) hasCjkOrEmoji = true;
            i += 2;
        } else if ((b0 >> 4) == 0x0E) {
            if (i + 2 >= s.size()) return false;
            uint8_t b1 = static_cast<uint8_t>(s[i + 1]);
            uint8_t b2 = static_cast<uint8_t>(s[i + 2]);
            if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) return false;
            // Reject U+FFFD replacement character (EF BF BD)
            if (b0 == 0xEF && b1 == 0xBF && b2 == 0xBD) return false;
            // Allow common English typography in General Punctuation (E2 80 93..A6: – — ‘ ’ “ ” • …)
            bool isGeneralPunct = (b0 == 0xE2 && b1 == 0x80 &&
                                   (b2 == 0x93 || b2 == 0x94 || b2 == 0x98 || b2 == 0x99 ||
                                    b2 == 0x9C || b2 == 0x9D || b2 == 0xA2 || b2 == 0xA6));
            if (!isGeneralPunct) {
                hasCjkOrEmoji = true;
            }
            i += 3;
        } else if ((b0 >> 3) == 0x1E) {
            if (i + 3 >= s.size()) return false;
            uint8_t b1 = static_cast<uint8_t>(s[i + 1]);
            uint8_t b2 = static_cast<uint8_t>(s[i + 2]);
            uint8_t b3 = static_cast<uint8_t>(s[i + 3]);
            if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80 || b0 > 0xF4) return false;
            hasCjkOrEmoji = true;
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

void Qwen2Client::initTokenFlags()
{
    mTokenFlags.assign(mConfig.vocabSize, 0);
    mBoldTokens.clear();
    mClosingBoldTokens.clear();
    mNoSpaceBoldTokens.clear();
    mOpenParenTokens.clear();
    mCloseParenTokens.clear();
    if (!mTokenizer) return;

    static const std::unordered_set<std::string> exemptWords = {
        "the", "The", "a", "A", "an", "An", "and", "or", "of", "to", "in", "In",
        "is", "are", "was", "were", "be", "been", "being", "that", "which", "who",
        "for", "with", "on", "as", "by", "at", "from", "this", "This", "it", "It",
        "they", "They", "their", "Their", "them", "have", "has", "had", "can", "could",
        "will", "would", "may", "might", "while", "While", "both", "Both", "more", "most",
        "than", "not", "but", "But", "also", "such", "other", "some", "into", "between",
        "these", "those", "its", "one", "two"
    };

    static const std::unordered_set<std::string> exemptPunct = {
        ".", ",", ":", ";", "!", "?", "-", "--", "—", "–", " -",
        "\n", "\n\n", ".\n", ".\n\n", ":\n", ":\n\n",
        "**", " **", ":**", "**\n", "**\n\n", "**:", "**:\n",
        "*", " *", "###", "####", "##", " ###", " ####"
    };

    static const std::vector<std::string> glitchSubstrings = {
        "${", "}$", "^{", "_{", "(($", "(${", "=?", "!!", "..", ",,", "*(", "*)",
        "—\"", ".\"", "\".", "\",", ".'", "'.", "\"]", "].", "))))", "(((",
        "():", "->", "<-", "=>", "<=", ">=", "!=", "==", "::", "//", "/*", "*/",
        "||", "&&", "<<", ">>", "()", "[]", "{}", "(>", "<)", "(-", "-)", "(+", "+)",
        "[[", "]]", "([", "])", "(?", "?)", "):", "?:", ":?", "&:", "-lg",
        ".!", "!.", "?!", "!?", "):.", ").?", "):,", ".*", ":*"
    };

    int32_t baseLimit = std::min(mConfig.vocabSize, 151643);
    for (int32_t id = 0; id < baseLimit; ++id) {
        std::string s = mTokenizer->decode(std::vector<int32_t>(1, id));
        if (s.empty()) {
            mTokenFlags[id] |= TOK_FLAG_FORBIDDEN;
            continue;
        }

        bool hasCjkOrEmoji = false;
        if (!isValidUtf8AndClassify(s, hasCjkOrEmoji)) {
            mTokenFlags[id] |= TOK_FLAG_FORBIDDEN;
            continue;
        }
        if (hasCjkOrEmoji) {
            mTokenFlags[id] |= TOK_FLAG_CJK_OR_EMOJI;
        }

        for (const auto& sub : glitchSubstrings) {
            if (s.find(sub) != std::string::npos) {
                mTokenFlags[id] |= TOK_FLAG_FORBIDDEN;
                break;
            }
        }

        if (s.find('$') != std::string::npos || s.find('^') != std::string::npos ||
            s.find('{') != std::string::npos || s.find('}') != std::string::npos ||
            s.find('[') != std::string::npos || s.find(']') != std::string::npos ||
            s.find('\\') != std::string::npos || s.find('"') != std::string::npos ||
            s.find('<') != std::string::npos || s.find('>') != std::string::npos ||
            s.find('=') != std::string::npos || s.find('|') != std::string::npos ||
            s.find('~') != std::string::npos || s.find('`') != std::string::npos ||
            s.find('_') != std::string::npos || s.find('&') != std::string::npos ||
            s.find('@') != std::string::npos) {
            mTokenFlags[id] |= TOK_FLAG_MATH_OR_QUOTE;
        }

        if (s.find('(') != std::string::npos) {
            mOpenParenTokens.push_back(id);
            if (s[0] == '(') {
                // Tokens starting with '(' without a leading space (e.g. "(er", "(x)") are code/word-suffix fragments
                mTokenFlags[id] |= TOK_FLAG_MATH_OR_QUOTE;
            }
        }
        if (s.find(')') != std::string::npos) {
            mCloseParenTokens.push_back(id);
        }

        if (s.find('\n') != std::string::npos) {
            mTokenFlags[id] |= TOK_FLAG_HAS_NEWLINE;
            mTokenFlags[id] |= TOK_FLAG_SENTENCE_END;
        } else if (s.find('.') != std::string::npos || s.find('!') != std::string::npos ||
                   s.find('?') != std::string::npos || s.find(';') != std::string::npos) {
            mTokenFlags[id] |= TOK_FLAG_SENTENCE_END;
        }

        if (s.find("**") != std::string::npos) {
            mTokenFlags[id] |= TOK_FLAG_HAS_BOLD;
            mBoldTokens.push_back(id);
            if (!s.empty() && (s[0] == ':' || (s.size() >= 3 && s.substr(0, 2) == "**" && (s[2] == ':' || s[2] == '\n' || s[2] == '.' || s[2] == ',')))) {
                mTokenFlags[id] |= TOK_FLAG_CLOSING_BOLD;
                mClosingBoldTokens.push_back(id);
            } else if (s == "**") {
                mNoSpaceBoldTokens.push_back(id);
            }
        }

        // Check if token is a whitelisted punctuation/markdown token, pure number, or common function word
        bool allDigits = true;
        std::string trimmed;
        for (unsigned char c : s) {
            if (!std::isspace(c)) trimmed.push_back(static_cast<char>(c));
            if (!std::isdigit(c) && !std::isspace(c) && c != '.') {
                allDigits = false;
            }
        }
        if (exemptPunct.count(s) > 0 || (!trimmed.empty() && allDigits) || exemptWords.count(trimmed) > 0) {
            mTokenFlags[id] |= TOK_FLAG_EXEMPT_PENALTY;
        }
    }

    // Special control & padding tokens (151643 .. vocabSize - 1)
    // 151643 = EOS (<|endoftext|> / <｜end▁of▁sentence｜>)
    // 151644 = <｜User｜> (forbidden)
    // 151645 = <｜Assistant｜> in DeepSeek (forbidden) or <|im_end|> in Qwen2 (EOS)
    // 151646 = <｜begin▁of▁sentence｜> (forbidden)
    // 151647 = <|EOT|> (EOS)
    // 151648 = <think>, 151649 = </think> (handled by state machine)
    if (151644 < mConfig.vocabSize) mTokenFlags[151644] |= TOK_FLAG_FORBIDDEN;
    if (mConfig.modelName == "deepseek_r1_distill_qwen" && 151645 < mConfig.vocabSize) {
        mTokenFlags[151645] |= TOK_FLAG_FORBIDDEN;
    }
    if (151646 < mConfig.vocabSize) mTokenFlags[151646] |= TOK_FLAG_FORBIDDEN;
    for (int32_t id = 151650; id < mConfig.vocabSize; ++id) {
        mTokenFlags[id] |= TOK_FLAG_FORBIDDEN;
    }
}

void Qwen2Client::analyzePromptFlags(const std::string& prompt)
{
    bool hasCjk = false;
    isValidUtf8AndClassify(prompt, hasCjk);
    mPromptAllowCjk = hasCjk || (mConfig.language == "zh");

    std::string lower = prompt;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    mPromptAllowMath = false;
    static const std::vector<std::string> mathOrCodeKeywords = {
        "code", "program", "python", "c++", "java", "rust", "script", "function",
        "json", "equation", "solve", "calculate", "math", "formula", "latex",
        "+", "=", "*", "/", "^", "$", "{", "}", "quote"
    };
    for (const auto& kw : mathOrCodeKeywords) {
        if (lower.find(kw) != std::string::npos) {
            mPromptAllowMath = true;
            break;
        }
    }
}

std::string Qwen2Client::genResponse(const std::string& rolePrompt, const std::string& prompts)
{
    std::vector<ChatMessage> messages;

    messages.emplace_back(ChatMessage(ChatMessage::ROLE_SYSTEM, rolePrompt));
    messages.emplace_back(ChatMessage(ChatMessage::ROLE_USER, prompts));

    auto startTime = std::chrono::high_resolution_clock::now();

    analyzePromptFlags(prompts);
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
    (void)idx;
    (void)logitData;
    (void)outputIds;
    (void)penalty;
}

void Qwen2Client::repetitionPenalty(int32_t idx, std::vector<TokenIdScore> &tokenScores, const std::vector<int32_t>& outputIds, float penalty)
{
    int32_t vocabOffset = mConfig.devPrivAttr[idx].vocabOffset;
    int32_t vocabEnd = vocabOffset + mConfig.vocabSizePerDie;
    size_t promptLen = mSentencePosInVocab.size();

    auto maskGlobalToken = [&](int32_t globalId) {
        if (globalId >= vocabOffset && globalId < vocabEnd) {
            tokenScores[globalId - vocabOffset].score = -1e9f;
        }
    };

    // 1. Apply precomputed vocabulary sanitizer flags (forbidden glitch tokens, CJK/emoji leakage, math/quote symbols)
    uint8_t disallowMask = TOK_FLAG_FORBIDDEN;
    if (!mPromptAllowCjk)  disallowMask |= TOK_FLAG_CJK_OR_EMOJI;
    if (!mPromptAllowMath) disallowMask |= TOK_FLAG_MATH_OR_QUOTE;

    if (mTokenFlags.size() >= (size_t)vocabEnd) {
        for (int32_t localIdx = 0; localIdx < mConfig.vocabSizePerDie; ++localIdx) {
            if (mTokenFlags[vocabOffset + localIdx] & disallowMask) {
                tokenScores[localIdx].score = -1e9f;
            }
        }
    }

    // 2. Enforce strict <think> (151648) and </think> (151649) state transitions
    size_t genLen = outputIds.size() > promptLen ? (outputIds.size() - promptLen) : 0;
    bool isFirstGenToken = (genLen == 0);
    bool sawThinkStart = false;
    bool sawThinkEnd = false;
    size_t thinkStartPos = promptLen;
    for (size_t k = promptLen; k < outputIds.size(); ++k) {
        if (outputIds[k] == 151648 && !sawThinkStart) {
            sawThinkStart = true;
            thinkStartPos = k;
        }
        if (outputIds[k] == 151649) {
            sawThinkEnd = true;
        }
    }
    bool currentlyInThink = (sawThinkStart && !sawThinkEnd);

    // <think> (151648) is ONLY valid as the very first generated token (and only if maxThinkingTokens > 0)
    if (!isFirstGenToken || mConfig.maxThinkingTokens <= 0) {
        maskGlobalToken(151648);
    }

    if (!currentlyInThink) {
        // Never allow </think> outside an active <think> block
        maskGlobalToken(151649);
    } else {
        // While inside <think>, require closing with </think> before emitting EOS
        maskGlobalToken(151643);
        maskGlobalToken(151645);
        maskGlobalToken(151647);

        size_t thinkingTokensSoFar = outputIds.size() > (thinkStartPos + 1) ? (outputIds.size() - (thinkStartPos + 1)) : 0;
        if (thinkingTokensSoFar == 0) {
            // Immediately after <think> (151648), prevent the `<think>\n\n</think>` bypass sequence
            // by masking "\n\n" (271) and "</think>" (151649) so the model emits "\n" (198) and thinks.
            maskGlobalToken(271);
            maskGlobalToken(151649);
        } else if (mConfig.maxThinkingTokens > 0 && (int32_t)thinkingTokensSoFar < std::min(24, mConfig.maxThinkingTokens)) {
            // Require at least 24 tokens of chain-of-thought reasoning before allowing </think>
            maskGlobalToken(151649);
        }
    }

    // 3. Frequency-based repetition penalty on content tokens only (never penalize punctuation/markdown/stopwords)
    if (penalty > 1.0f && genLen > 0) {
        int32_t windowSize = mConfig.repeatWindow > 0 ? mConfig.repeatWindow : 64;
        size_t windowStart = outputIds.size() > (size_t)windowSize ? outputIds.size() - windowSize : 0;
        size_t startPos = std::max(promptLen, windowStart);

        std::unordered_map<int32_t, int32_t> tokenCounts;
        for (size_t k = startPos; k < outputIds.size(); ++k) {
            tokenCounts[outputIds[k]]++;
        }

        int32_t prevToken = outputIds.back();
        float stepPenalty = (penalty - 1.0f) * 3.5f; // e.g. 0.35 logit points per extra repeat for penalty=1.10

        for (const auto& kv : tokenCounts) {
            int32_t id = kv.first;
            int32_t count = kv.second;
            if (id < vocabOffset || id >= vocabEnd) continue;

            // Never penalize structural/markdown/punctuation/function-word tokens unless immediately repeating
            bool isExempt = (id < (int32_t)mTokenFlags.size()) && ((mTokenFlags[id] & TOK_FLAG_EXEMPT_PENALTY) != 0);
            if (isExempt) {
                if (id == prevToken && id != 198 && id != 271) {
                    tokenScores[id - vocabOffset].score -= 1.5f;
                }
                continue;
            }

            // Immediate back-to-back repeat of a content word (e.g. "the the", "I I")
            if (id == prevToken) {
                tokenScores[id - vocabOffset].score -= 4.0f;
            } else if (count >= 2) {
                // Gentle additive frequency penalty starting from 2nd occurrence in window
                tokenScores[id - vocabOffset].score -= stepPenalty * static_cast<float>(count - 1);
            }
        }
    }

    // 4. Per-line Markdown `**` pairing & comma-splice run-on guard
    if (genLen > 0) {
        size_t lineStart = promptLen;
        size_t sentenceStart = promptLen;
        for (size_t k = promptLen; k < outputIds.size(); ++k) {
            int32_t id = outputIds[k];
            if (id == 151648 || id == 151649) {
                lineStart = k + 1;
                sentenceStart = k + 1;
                continue;
            }
            if (id >= 0 && id < (int32_t)mTokenFlags.size()) {
                uint8_t flags = mTokenFlags[id];
                if (flags & TOK_FLAG_HAS_NEWLINE) {
                    lineStart = k + 1;
                    sentenceStart = k + 1;
                } else if (flags & TOK_FLAG_SENTENCE_END) {
                    sentenceStart = k + 1;
                }
            }
        }

        int32_t boldCountOnLine = 0;
        int32_t openParenCountOnLine = 0;
        int32_t closeParenCountOnLine = 0;
        for (size_t k = lineStart; k < outputIds.size(); ++k) {
            int32_t id = outputIds[k];
            if (id >= 0 && id < (int32_t)mTokenFlags.size() && (mTokenFlags[id] & TOK_FLAG_HAS_BOLD)) {
                boldCountOnLine++;
            }
            if (id == 7 || id == 320) { // "(" or " ("
                openParenCountOnLine++;
            }
            if (id == 8 || id == 568 || id == 701 || id == 2590) { // ")", ").", "),", "):\n"
                closeParenCountOnLine++;
            }
        }

        int32_t prevToken = outputIds.back();
        if (currentlyInThink || prevToken == 11) {
            // Never emit markdown bold `**` inside <think> or immediately after a comma `,` (#11 -> `,**` glitch)
            for (int32_t bId : mBoldTokens) {
                maskGlobalToken(bId);
            }
        } else if (boldCountOnLine == 0) {
            // No bold span is open on this line: disallow closing-only bold tokens (`:**`, `**\n`, `**:`)
            for (int32_t bId : mClosingBoldTokens) {
                maskGlobalToken(bId);
            }
            // Only allow opening a bold span at the beginning of a line (bullet/numbered header, <= 3 tokens from lineStart)
            if (outputIds.size() - lineStart > 3) {
                for (int32_t bId : mBoldTokens) {
                    maskGlobalToken(bId);
                }
            }
        } else if (boldCountOnLine >= 2) {
            // A bold header (e.g. `- **Behavior**:`) has already opened and closed on this line;
            // mask further `**` tokens until the next newline to prevent self-attention bold contagion
            for (int32_t bId : mBoldTokens) {
                maskGlobalToken(bId);
            }
        }

        // Prevent parenthetical citation contagion (`(?,`, `([`) in prose mode:
        // Allow at most 1 parenthetical per line, and never immediately after `(`
        if (!mPromptAllowMath && (openParenCountOnLine >= 1 || prevToken == 7 || prevToken == 320)) {
            for (int32_t pId : mOpenParenTokens) {
                maskGlobalToken(pId);
            }
        }
        // Disallow unmatched closing parenthesis `)` when no `(` is currently open on this line
        if (!mPromptAllowMath && openParenCountOnLine <= closeParenCountOnLine) {
            for (int32_t cId : mCloseParenTokens) {
                maskGlobalToken(cId);
            }
        }
        // Immediately after `(` or ` (`, never allow `?`, `:`, `,`, `.`, `;`, `!`
        if (prevToken == 7 || prevToken == 320) {
            maskGlobalToken(30);  // "?"
            maskGlobalToken(949); // " ?"
            maskGlobalToken(25);  // ":"
            maskGlobalToken(11);  // ","
            maskGlobalToken(13);  // "."
            maskGlobalToken(26);  // ";"
            maskGlobalToken(0);   // "!"
        }

        // Prevent endless comma-splice run-on sentences (where the model chains 20+ clauses with `,`)
        int32_t commasInSentence = 0;
        for (size_t k = sentenceStart; k < outputIds.size(); ++k) {
            if (outputIds[k] == 11) {
                commasInSentence++;
            }
        }
        if (commasInSentence >= 3 && 11 >= vocabOffset && 11 < vocabEnd) {
            if (commasInSentence >= 6) {
                maskGlobalToken(11);
            } else {
                tokenScores[11 - vocabOffset].score -= 2.5f * static_cast<float>(commasInSentence - 2);
            }
        }
    }

    // 5. Proactive anti-stutter & cyclic n-gram loop suppression on generated tokens
    if (genLen >= 2) {
        int32_t last1 = outputIds.back();
        int32_t last2 = outputIds[outputIds.size() - 2];
        if (last1 == last2) {
            maskGlobalToken(last1);
        }
    }
    for (size_t period = 2; period <= 12; ++period) {
        if (genLen >= 2 * period) {
            bool repeating = true;
            size_t base = outputIds.size() - period;
            for (size_t p = 0; p < period; ++p) {
                if (outputIds[base - period + p] != outputIds[base + p]) {
                    repeating = false;
                    break;
                }
            }
            if (repeating) {
                // Demote the token that would start the 3rd consecutive repeat of this cycle
                maskGlobalToken(outputIds[base]);
            }
        }
    }
}

void Qwen2Client::selectTopK(std::vector<TokenIdScore> &tokenScores, int32_t topK, int32_t numTokens)
{
    if (numTokens > topK) {
        std::partial_sort(tokenScores.data(), tokenScores.data() + topK, tokenScores.data() + numTokens, std::greater<TokenIdScore>());
    } else {
        std::sort(tokenScores.data(), tokenScores.data() + numTokens, std::greater<TokenIdScore>());
    }
}

void Qwen2Client::softmaxInPlace(std::vector<TokenIdScore> &tokenScores, int32_t numTokens)
{
    if (numTokens <= 0) return;
    float maxScore = std::max_element(tokenScores.data(), tokenScores.data() + numTokens)->score;
    float sum = 0.f;
    for (int32_t i = 0; i < numTokens; i++) {
        float s = std::exp(tokenScores[i].score - maxScore);
        tokenScores[i].score = s;
        sum += s;
    }
    if (sum > 0.f) {
        float invSum = 1.f / sum;
        for (int32_t i = 0; i < numTokens; i++) {
            tokenScores[i].score *= invSum;
        }
    }
}

int32_t Qwen2Client::samplingTopP(std::vector<TokenIdScore> &tokenScores, float topP, int32_t numTokens)
{
    // tokenScores[0 .. numTokens-1] is already sorted in descending order by selectTopK
    float cumSum = 0.0f;
    for (int32_t i = 0; i < numTokens; i++) {
        cumSum += tokenScores[i].score;
        if (cumSum >= topP) {
            return i + 1;
        }
    }
    return numTokens;
}
