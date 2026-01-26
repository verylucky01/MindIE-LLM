/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef OCK_ENDPOINT_HEALTH_CHECKER_WRAPPER_H
#define OCK_ENDPOINT_HEALTH_CHECKER_WRAPPER_H

#include <vector>
#include <shared_mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <set>
#include <unordered_map>
#include <mutex>
#include <memory>
#include "log.h"
#include "concurrent_deque.h"
#include "infer_instances.h"
#include "config_manager.h"
#include "simulate_task_runner.h"

namespace mindie_llm {

enum ServiceStatus : uint32_t {
    SERVICE_READY = 0,
    SERVICE_NORMAL = 1,
    SERVICE_ABNORMAL = 2,
    SERVICE_PAUSE = 3,
    SERVICE_INIT = 4,
    SERVICE_BUSY = 5
};

enum EndpointStatusCode : uint32_t {
    STATUS_CODE_INIT = 0b111,
    STATUS_CODE_NORMAL = 0b000,
    STATUS_CODE_PAUSE = 0b100,
    STATUS_CODE_ABNORMAL = 0b001,
    STATUS_CODE_READY = 0b010,
    STATUS_CODE_ABNORMAL_PAUSE = 0b101,
    STATUS_CODE_ABNORMAL_READY = 0b011
};

struct ErrorItem {
    struct Addition {
        std::string deviceIP;
        int deviceID;
        Addition() : deviceIP(""), deviceID(-1) {}
        Addition(const std::string &deviceIP, const int &deviceID) : deviceIP(deviceIP), deviceID(deviceID) {}
    };

    int64_t timestamp;
    std::string errCode;
    std::string createdBy;
    Addition addition;
    ErrorItem()
        : timestamp(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count()),
          errCode(""), createdBy(""), addition() {}
    ErrorItem(const std::string &errCode, const std::string &createdBy, const std::string &deviceIP,
              const int &deviceID, const std::chrono::time_point<std::chrono::system_clock> &timestamp)
        : timestamp(std::chrono::duration_cast<std::chrono::milliseconds>(timestamp.time_since_epoch()).count()),
          errCode(errCode), createdBy(createdBy), addition(deviceIP, deviceID) {}
};

class HealthChecker {
public:
    static HealthChecker &GetInstance();
    ~HealthChecker();
    ServiceStatus GetServiceStatus();
    EndpointStatusCode GetEndpointStatusCode();
    void GetStatusAndErrorList(ServiceStatus &status, std::vector<ErrorItem> &errorList);
    void UpdateNpuDeviceIds(const std::set<int> &npuDeviceIds);
    void UpdateStatusByCode(const EndpointStatusCode &code);
    void EnqueueErrorMessage(
        const std::string &errCode, const std::string &createdBy, const std::string &deviceIP = "",
        const int &deviceID = -1,
        const std::chrono::time_point<std::chrono::system_clock> &timestamp = std::chrono::system_clock::now());
    void PrintNpuDeviceIds();
    std::string CodeToString(const EndpointStatusCode &code) const;
    std::string StatusToString(const ServiceStatus &status) const;
    bool Start();
    void Stop();
    SimulateResult RunHttpTimedHealthCheck(uint32_t waitTime);
    HealthChecker(const HealthChecker &) = delete;
    HealthChecker &operator=(const HealthChecker &) = delete;
    HealthChecker(HealthChecker &&) = delete;
    HealthChecker &operator=(HealthChecker &&) = delete;

private:
    std::atomic<ServiceStatus> mServiceStatus;
    std::atomic<EndpointStatusCode>
        mEndpointStatusCode; // Status code bits: [PAUSE(bit 2), READY(bit 1), ABNORMAL(bit 0)]
    int mChipPerCard = 1;    // A2: 1, A3: 2
    mindie_llm::ConcurrentDeque<ErrorItem> mErrorList;
    std::set<int> mNpuDeviceCardIds;
    std::string mEngineName;
    std::shared_mutex mNpuDevicesMutex;
    std::thread mCheckerThread;
    std::atomic<bool> mRunning;
    std::unordered_map<int, std::vector<int>> statusTransferMap;
    static constexpr int checkIntervalSeconds = 5;
    static constexpr int maxErrorListSize = 100;

    void CheckServiceStatus();
    void GetChipPerCard();
    void UpdateErrorList(
        const std::string &errCode, const std::string &createdBy, const std::string &deviceIP, const int &deviceID,
        const std::chrono::time_point<std::chrono::system_clock> &timestamp = std::chrono::system_clock::now());

    bool CheckErrorListEmpty();
    // npu monitor for self health check
    bool CheckModelInstanceStarted() const;
    std::vector<int> GetAicoreUsages(int npuId) const;
    std::string ExecuteCommand(const std::string &cmd) const;
    std::vector<int> ParseAicoreUsage(const std::string &output) const;
    bool CheckAllNpuAicoreUsage();
    bool CheckVirtualInfer() const;
    bool IsValidStatusTransition(const EndpointStatusCode &from, const EndpointStatusCode &to);

private:
    HealthChecker();
};

} // namespace mindie_llm

#endif