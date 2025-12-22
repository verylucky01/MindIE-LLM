/**
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
 
#ifndef PD_MSG_RECEIVER_H
#define PD_MSG_RECEIVER_H

#include <memory>
#include <string>
#include <functional>
#include <vector>

#include <grpcpp/server.h>
#include <grpcpp/server_context.h>
#include <grpcpp/grpcpp.h>
#include "prefillAndDecodeCommunication.grpc.pb.h"

using GetDecodeRequestFunc = std::function<void(const prefillAndDecodeCommunication::DecodeParameters& request,
                                                prefillAndDecodeCommunication::DecodeRequestResponse& response)>;

using GetRequestIDFunc = std::function<void(const std::string& requestID)>;

using GetDeviceListFunc = std::function<bool(const std::vector<std::string>& deviceIp)>;

namespace mindie_llm {
        class DecodeRequestReceiver : public prefillAndDecodeCommunication::DecodeService::Service {
        public:
            explicit DecodeRequestReceiver(std::string localAddr): localAddr_(localAddr) {}

            grpc::Status DecodeRequestChannel(grpc::ServerContext* context,
                                              const prefillAndDecodeCommunication::DecodeParameters* request,
                                              prefillAndDecodeCommunication::DecodeRequestResponse* response) override;

            bool RegisterMsgHandler(GetDecodeRequestFunc callBack);

        private:
            bool isValidRequest(const prefillAndDecodeCommunication::DecodeParameters* request,
                                prefillAndDecodeCommunication::DecodeRequestResponse* response, std::string& errMsg);

            std::string localAddr_;

            GetDecodeRequestFunc getDecodeRequestFunc_{nullptr};
        };

        class KvReleaseReceiver : public prefillAndDecodeCommunication::PrefillService::Service {
        public:
            explicit KvReleaseReceiver(std::string localAddr): localAddr_(localAddr) {}

            grpc::Status ReleaseKVCacheChannel(grpc::ServerContext* context,
                                          const prefillAndDecodeCommunication::RequestId* request,
                                          google::protobuf::Empty* response) override;

            bool RegisterMsgHandler(GetRequestIDFunc callBack);

        private:
            bool isValidRequest(const prefillAndDecodeCommunication::RequestId* request);

            std::string localAddr_;

            GetRequestIDFunc getRequestIDFunc_{nullptr};
        };

        class ForceReleaseLinkReceiver : public prefillAndDecodeCommunication::ForcePReleaseService::Service {
        public:
            explicit ForceReleaseLinkReceiver(std::string localAddr): localAddr_(localAddr) {}

            grpc::Status ForceReleaseLinkChannel(grpc::ServerContext* context,
                                            const prefillAndDecodeCommunication::DeviceList* request,
                                            google::protobuf::Empty* response) override;

            bool RegisterMsgHandler(GetDeviceListFunc callBack);

        private:
            bool isValidRequest(const prefillAndDecodeCommunication::DeviceList* request);

            std::string localAddr_;

            GetDeviceListFunc getDeviceListFunc_{nullptr};
        };
} // namespace mindie_llm

#endif // PD_MSG_RECEIVER_H
