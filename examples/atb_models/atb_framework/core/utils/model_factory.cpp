/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#include "atb_speed/utils/model_factory.h"
#include "system_log.h"

namespace atb_speed {
bool ModelFactory::Register(const std::string &modelName, CreateModelFuncPtr createModel)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        if (it->second == nullptr) {
            LOG_ERROR_MODEL << "Find modelName error: " << modelName;
            return false;
        }
        return false;
    }
    ModelFactory::GetRegistryMap()[modelName] = createModel;
    return true;
}

std::shared_ptr<atb_speed::Model> ModelFactory::CreateInstance(const std::string &modelName, const std::string &param)
{
    auto it = ModelFactory::GetRegistryMap().find(modelName);
    if (it != ModelFactory::GetRegistryMap().end()) {
        LOG_DEBUG_MODEL << "Find model: " << modelName;
        return it->second(param);
    }
    LOG_WARN_MODEL << "ModelName: " << modelName << " not find in model factory map";
    return nullptr;
}

std::unordered_map<std::string, CreateModelFuncPtr> &ModelFactory::GetRegistryMap()
{
    static std::unordered_map<std::string, CreateModelFuncPtr> modelRegistryMap;
    return modelRegistryMap;
}
} // namespace atb_speed