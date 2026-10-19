// Copyright © 2024 ESWIN. All rights reserved.
//
// Beijing ESWIN Computing Technology Co., Ltd and its affiliated companies ("ESWIN") retain
// all intellectual property and proprietary rights in and to this software. Except as expressly
// authorized by ESWIN, no part of the software may be released, copied, distributed, reproduced,
// modified, adapted, translated, or created derivative work of, in whole or in part.

#ifndef __ES_QWEN2_CLIENT_H__
#define __ES_QWEN2_CLIENT_H__

#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <list>
#include <memory>
#include <utility>
#include <cstdint>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "es_npu_interface.h"
#include "es_sys_memory.h"
#include "llm_tokenizer.hpp"

typedef uint16_t fp16_t;
typedef int32_t prob_data_t;

#define QWEN2_CONFIG_FILE "/opt/eswin/sample-code/npu_sample/qwen_sample/src/qwen2_7b/config.json"
#define MASK_BUFFER_SIZE (sizeof(int16_t))

struct CLIENT_INTERVAL_S {
    CLIENT_INTERVAL_S() : min(0), max(0) {}
    CLIENT_INTERVAL_S(int32_t mi, int32_t ma) : min(mi), max(ma) {}
    int32_t min;
    int32_t max;
};

typedef struct {
    uint16_t devId;
    std::vector<std::string> blockModelFiles;
    std::string logitsModelFile;
    int32_t vocabOffset;
    int32_t dataDump;
    CLIENT_INTERVAL_S dataTokenInterval;
    int32_t opDump;
    int32_t opType;
    CLIENT_INTERVAL_S opTokenInterval;
    CLIENT_INTERVAL_S opLayerInterval;
    CLIENT_INTERVAL_S opListInterval;
    int32_t perfEnable;
} DevicePrivAttr;

typedef struct {
    std::string modelName;
    std::string modelDir;
    bool logits;
    std::string vocabFile;
    std::string vocabJson;
    std::string mergesFile;
    std::string tiktokenFile;
    std::string embeddingFile;
    int32_t vocabSize;
    int32_t vocabSizePerDie;
    int32_t embeddingSize;
    int32_t numHead;
    int32_t embeddingDim;
    int32_t embeddingDim1;
    int32_t embeddingDim2;
    int32_t tokenNum;
    int32_t tokenN1;
    int32_t tokenN2;
    int32_t precision;
    std::string language;
    std::vector<std::string> specialTokens;
    std::vector<std::string> eosList;
    int32_t responseMaxLen;
    int32_t responseLineLen;
    std::map<int32_t, std::string> prompts;
    std::map<int32_t, std::string> rolePrompts;
    int32_t randomCnt;
    int32_t repeatLen;
    int32_t repeatCnt;
    int32_t logLevel;
    int32_t modelTest;
    float repetitionPenalty;
    int32_t repeatWindow;
    int32_t topK;
    bool doSample;
    float topP;
    float temperature;
    std::vector<DevicePrivAttr> devPrivAttr;
} CLIENT_CONFIG_S;

typedef enum {
    IN_TOKEN_IDX = 0,
    IN_MASK_IDX,
    IN_V_CACHE_IDX,
    IN_K_CACHE_IDX,
    IN_MAX_IDX
} INPUT_TENSOR_E;

typedef enum {
    OUT_HIDDEN_IDX = 0,
    OUT_V_CACHE_IDX,
    OUT_K_CACHE_IDX,
    OUT_MAX_IDX
} OUTPUT_TENSOR_E;

typedef struct {
    NPU_TASK_S task;
    void* inputMapAddr[IN_MAX_IDX];
    void* outputMapAddr[OUT_MAX_IDX];
    bool isPrepared;
} MODEL_TASK_S;

void esBindThread2Die(int32_t dieId);

struct TokenIdScore {
    int id;
    float score;

    TokenIdScore() = default;
    TokenIdScore(int id, float score) : id(id), score(score) {}

    auto operator<(const TokenIdScore &other) const -> bool { return score < other.score; }
    auto operator>(const TokenIdScore &other) const -> bool { return score > other.score; }

    friend auto operator<<(std::ostream &os, const TokenIdScore &self) -> std::ostream & {
        return os << "TokenIdScore(id=" << self.id << ", score=" << self.score << ")";
    }
};

class Qwen2Client
{
 public:
    explicit Qwen2Client();
    ~Qwen2Client();

    void start(const std::string& configPath);
    void init(const std::string& configPath);
    ES_S32 loadModel(int32_t idx);
    ES_S32 unloadModel(int32_t idx);
    ES_S32 initDevice(int32_t idx);
    ES_S32 deinitDevice(int32_t idx);
    void runModelAsync(MODEL_TASK_S& modelTask, npu_stream stream);
    void runModel(MODEL_TASK_S& modelTask);

    std::string genResponse(const std::string& rolesPrompt, const std::string& prompts);
    std::map<int32_t, std::string> getOptionalPrompts();
    std::map<int32_t, std::string> getOptionalRolePrompts();
    void exitClient();

    static std::atomic<int32_t> mCompletedNum;
    static std::atomic<bool> mExited;
    static std::atomic<bool> mStopped;

    ES_S32 prepareBlockTaskBuffers(int32_t idx, int32_t modelIdx, uint32_t modelId);
    ES_S32 prepareLogitsTaskBuffers(int32_t idx, uint32_t modelId);
    ES_U64 calcKCacheOutputAddrOffset(int32_t num);
    ES_U64 calcVCacheOutputAddrOffset(int32_t num);

    std::vector<std::vector<MODEL_TASK_S>> & getBlockTask() { return mBlockTasks; };
    std::vector<MODEL_TASK_S> & getLogitsTask() { return mLogitsTask; };
    std::vector<npu_stream> & getStream() { return mStreams; };
    CLIENT_CONFIG_S & getConfig() { return mConfig; }
    fp16_t *getWordEmbeddings() { return mWordEmbeddings; };
    std::shared_ptr<QwenTokenizer> getTokenizer() { return mTokenizer; };
 private:
    // Tokenization and embedding function for Chinese
    std::vector<std::string> splitUtf8(const std::string& str);
    std::vector<std::string> splitEn(const std::string& input);
    std::vector<std::string> splitSentence(const std::string& input);

    // Utils
    bool checkDuplicateWords(const std::string& sentence);
    bool checkTokenRepetition(const std::vector<int32_t>& outputIds);
    ES_S32 dumpData(const void* address, const size_t size, const std::string& file);
    void displayLoadModelProgress(int32_t current, int32_t total, double costTime);

    // Model loading and initialization
    ES_S32 loadConfig(const std::string& configPath);
    ES_S32 loadEmbedding();
    ES_S32 prepareTensorBuffer(NPU_TENSOR_S &tensor, ES_DEV_BUF_S &devBuf, void **pAddr, uint16_t devId, bool cache);
    ES_S32 prepareKVCacheBuffers(uint32_t modelId, MODEL_TASK_S &modelTask, uint16_t devId);
    prob_data_t getTopK(int32_t idx, int32_t wordPosInVocab, const int32_t& maskVal, const std::vector<int32_t>& outputIds);
    std::string generate(int32_t idx, bool master);

    ES_S32 taskThreadFunc(int32_t idx);

    // Model release functions
    ES_S32 releaseTaskBuffers(int32_t idx);

    inline bool checkInterval(int32_t value, const CLIENT_INTERVAL_S &interval);
    void repetitionPenalty(int32_t idx, uint16_t* logitData, const std::vector<int32_t>& outputIds, float penalty);
    void repetitionPenalty(int32_t idx, std::vector<TokenIdScore> &tokenScores, const std::vector<int32_t>& outputIds, float penalty);
    void selectTopK(std::vector<TokenIdScore> &tokenScores, int32_t topK, int32_t numTokens);
    void softmaxInPlace(std::vector<TokenIdScore> &tokenScores, int32_t numTokens);
    int32_t samplingTopP(std::vector<TokenIdScore> &tokenScores, float topP, int32_t numTokens);

    CLIENT_CONFIG_S mConfig;
    std::vector<npu_context> mContexts;
    std::vector<npu_stream> mStreams;
    std::vector<std::vector<MODEL_TASK_S>> mBlockTasks;
    std::vector<MODEL_TASK_S> mLogitsTask;
    std::shared_ptr<QwenTokenizer> mTokenizer;
    int32_t mWordEmbeddingSize;
    fp16_t *mWordEmbeddings;
    std::vector<std::thread> mTaskThreadVec;
    std::mutex mTaskMutex;
    std::condition_variable mTaskCv;
    std::condition_variable mCompleteCv;
    std::atomic<int32_t> mLoadedModels;
    std::atomic<int32_t> mTotalModels;
    std::vector<int32_t> mSentencePosInVocab;

    // token mask for all device.
    std::vector<std::shared_ptr<std::atomic<int32_t>>> mTokenMask;
    std::vector<std::vector<TokenIdScore>> mTokenScores;

    // sync for post process
    std::mutex mPostSyncMutex;
    std::condition_variable mPostSyncCv;
    std::atomic<uint64_t> mDeviceMask;
    std::atomic<uint64_t> mDeviceStatus;
    std::atomic<int32_t> mPostCnt;

    // sync for tokenId
    std::mutex mTokenIdSyncMutex;
    std::condition_variable mTokenIdSyncCv;
    std::atomic<bool> mTokenIdDone;
    std::atomic<int32_t> mTokenNotifyCnt;
    int32_t mTokenId;
};

#endif