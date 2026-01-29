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
#include "safe_path.h"

#include <regex>
#include <unistd.h>
#include <pwd.h>
#include <grp.h>
#include <iostream>
#include <sys/stat.h>

#include "string_utils.h"
#include "safe_envvar.h"

namespace mindie_llm {

static constexpr size_t MAX_PATH_LENGTH = 4096;
static constexpr size_t MAX_LAST_NAME_LENGTH = 255;
static constexpr size_t MAX_DIR_DEPTH = 32;

static const std::map<std::string, int> modeMap = {
    {"r",   R_OK},            // 只读
    {"r+",  R_OK | W_OK},     // 读写
    {"w",   W_OK},            // 只写
    {"w+",  R_OK | W_OK},     // 读写
    {"a",   W_OK},            // 追加写
    {"a+",  R_OK | W_OK},     // 读写追加
    {"x",   X_OK}             // 只可执行
};

Result ChangePermission(const std::string& path, const fs::perms& permission)
{
    if (!fs::exists(path) || fs::is_symlink(path)) {
        return Result::Error(ResultCode::RISK_ALERT, "Path does not exist or is a symlink: " + path);
    }
    try {
        fs::permissions(path, permission);
    } catch (const fs::filesystem_error &e) {
        return Result::Error(ResultCode::IO_FAILURE, "Failed to set permissions for " + path + ": " + e.what());
    }
    return Result::OK();
}

Result MakeDirs(const std::string& pathStr)
{
    if (fs::exists(pathStr)) {
        return Result::OK();
    }
    fs::path path(pathStr);
    size_t depth = static_cast<size_t>(std::distance(path.begin(), path.end()));
    if (depth > MAX_DIR_DEPTH) {
        return Result::Error(ResultCode::RISK_ALERT,
            "Exceeded max directory depth: " + std::to_string(MAX_DIR_DEPTH));
    }
    try {
        fs::create_directories(path);
        ChangePermission(path, PERM_750);
    } catch (const fs::filesystem_error &e) {
        return Result::Error(ResultCode::IO_FAILURE, "Failed to create directories for " + pathStr + ": " + e.what());
    }
    return Result::OK();
}

SafePath::SafePath(std::string path,
                   PathType pathType,
                   std::string mode,
                   uint64_t sizeLimitation,
                   std::string suffix)
    : path_(std::move(path)),
      pathType_(pathType),
      mode_(std::move(mode)),
      sizeLimitation_(sizeLimitation),
      suffix_(suffix) {}

Result SafePath::Check(std::string& checkedPath, bool pathExist, SoftLinkLevel softLinkLevel)
{
    Result r = NormalizePath();
    if (!r.IsOk()) {
        return r;
    }
    r = pathExist ? CheckPathWhenExist(softLinkLevel) : CheckPathWhenNotExist(softLinkLevel);
    if (!r.IsOk()) {
        return r;
    }
    checkedPath = path_;
    return r;
}

Result SafePath::NormalizePath()
{
    if (!path_.empty() && path_[0] == '~') {
        std::string home;
        Result r = EnvVar::GetInstance().Get("HOME", "/root/", home);
        if (!r.IsOk()) {
            return r;
        }
        if (path_.size() == 1) {
            path_ = home;
        } else if (path_[1] == '/') {
            path_ = home + path_.substr(1);
        }
    }
    fs::path abs = fs::absolute(path_);
    fs::path existing;
    for (auto it = abs.begin(); it != abs.end(); ++it) {
        fs::path trial = existing / *it;
        if (fs::exists(trial)) {
            existing = trial;
        } else {
            break;
        }
    }
    fs::path result;
    try {
        if (!existing.empty()) {
            result = fs::canonical(existing);
        } else {
            result = fs::absolute(abs.root_path());
        }
    } catch (...) {
        result = fs::absolute(abs.root_path());
    }
    if (existing != abs) {
        fs::path suffix;
        auto abs_it = abs.begin();
        auto ex_it  = existing.begin();
        while (abs_it != abs.end() && ex_it != existing.end() && *abs_it == *ex_it) {
            ++abs_it;
            ++ex_it;
        }
        for (; abs_it != abs.end(); ++abs_it) {
            suffix /= *abs_it;
        }
        result /= suffix;
        fs::path normalized;
        for (auto &part : result) {
            std::string s = part.string();
            if (s == ".") {
                continue;
            }
            if (s == "..") {
                if (!normalized.empty()) {
                    normalized = normalized.parent_path();
                }
            } else {
                normalized /= part;
            }
        }
        result = normalized;
    }
    path_ = result.string();
    return Result::OK();
}

Result SafePath::CheckPathWhenExist(SoftLinkLevel softLinkLevel)
{
    std::string resolvedPath;
    Result r = CheckPathExist(path_);
    if (!r.IsOk()) {
        return r;
    }
    r = CheckSoftLink(path_, softLinkLevel, resolvedPath);
    if (!r.IsOk()) {
        return r;
    }
    path_ = resolvedPath;
    r = CheckSpecialChars();
    if (!r.IsOk()) {
        return r;
    }
    r = CheckPathLength();
    if (!r.IsOk()) {
        return r;
    }
    if (pathType_ == PathType::FILE) {
        r = IsFile();
        if (!r.IsOk()) {
            return r;
        }
        r = CheckFileSuffix();
        if (!r.IsOk()) {
            return r;
        }
        r = CheckFileSize();
        if (!r.IsOk()) {
            return r;
        }
    } else if (pathType_ == PathType::DIR) {
        r = IsDir();
        if (!r.IsOk()) {
            return r;
        }
        r = CheckDirSize();
        if (!r.IsOk()) {
            return r;
        }
        if (!IsSuffix(path_, "/")) {
            path_ += "/";
        }
    }
    return CheckPermission(path_);
}

Result SafePath::CheckPathWhenNotExist(SoftLinkLevel softLinkLevel)
{
    std::string resolvedPath;
    fs::path parentDir = fs::path(path_).parent_path();

    Result r = CheckPathExist(parentDir.string());
    if (!r.IsOk()) {
        return r;
    }
    r = CheckSoftLink(parentDir.string(), softLinkLevel, resolvedPath);
    if (!r.IsOk()) {
        return r;
    }
    parentDir = resolvedPath;
    r = CheckSpecialChars();
    if (!r.IsOk()) {
        return r;
    }
    r = CheckPathLength();
    if (!r.IsOk()) {
        return r;
    }
    return CheckPermission(parentDir);
}

Result SafePath::IsFile()
{
    if (!fs::is_regular_file(path_)) {
        return Result::Error(ResultCode::TYPE_MISMATCH, "The path is not file: " + path_);
    }
    return Result::OK();
}

Result SafePath::IsDir()
{
    if (!fs::is_directory(path_)) {
        return Result::Error(ResultCode::TYPE_MISMATCH, "The path is not directory: " + path_);
    }
    return Result::OK();
}

Result SafePath::CheckPathExist(const std::string& path) const
{
    if (!fs::exists(path)) {
        return Result::Error(ResultCode::NONE_ARGUMENT, "Path not found: " + path);
    }
    return Result::OK();
}

Result SafePath::CheckSoftLink(const std::string& path, SoftLinkLevel level, std::string& resolvedPath) const
{
    if (!fs::is_symlink(path)) {
        resolvedPath = path;
        return Result::OK();
    }
    if (level == SoftLinkLevel::STRICT) {
        return Result::Error(ResultCode::RISK_ALERT, "Found symlink path: " + path);
    }
    resolvedPath = fs::read_symlink(path).string();
    return Result::OK();
}

Result SafePath::CheckWriterForGroupOthers(const std::string& path) const
{
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) {
        return Result::Error(ResultCode::NO_PERMISSION, "Cannot stat path: " + path);
    }
    if ((st.st_mode & static_cast<mode_t>(S_IWGRP)) != 0 || (st.st_mode & static_cast<mode_t>(S_IWOTH)) != 0) {
        return Result::Error(ResultCode::RISK_ALERT, "Path writable by group or others: " + path);
    }
    return Result::OK();
}

Result SafePath::CheckMode(const std::string& path) const
{
    auto it = modeMap.find(mode_);
    if (it == modeMap.end()) {
        std::string keys = GetKeysFromMap(modeMap, ",");
        return Result::Error(ResultCode::INVALID_ARGUMENT,
            "Unsupported mode: " + mode_ + ". Only supported modes are: " + keys);
    }
    int accessMode = it->second;
    if (access(path.c_str(), accessMode) != 0) {
        return Result::Error(ResultCode::NO_PERMISSION,
            "Insufficient permissions for mode '" + mode_ + "' on path: " + path);
    }
    return Result::OK();
}

Result SafePath::CheckPermission(const std::string& path) const
{
    uid_t uid = geteuid();
    if (uid != 0) {
        // 不是 root 用户才检查写权限
        Result r = CheckWriterForGroupOthers(path);
        if (!r.IsOk()) {
            return r;
        }
    }
    Result r = CheckMode(path);
    if (!r.IsOk()) {
        return r;
    }
    return Result::OK();
}

Result SafePath::CheckSpecialChars()
{
    const std::regex VALID_PATH_PATTERN(R"(^(?!.*\.\.)[a-zA-Z0-9_./-]+$)");
    if (!std::regex_match(path_, VALID_PATH_PATTERN)) {
        return Result::Error(ResultCode::INVALID_ARGUMENT, "Path contains special characters: " + path_);
    }
    return Result::OK();
}

Result SafePath::CheckPathLength()
{
    if (path_.length() > MAX_PATH_LENGTH) {
        return Result::Error(ResultCode::RISK_ALERT,
            "Path length exceeds maximum limit: " + std::to_string(MAX_PATH_LENGTH));
    }
    size_t depth = 0;
    for (auto& subName : fs::path(path_)) {
        ++depth;
        if (depth > MAX_DIR_DEPTH) {
            return Result::Error(ResultCode::RISK_ALERT,
                "Exceeded max directory depth: " + std::to_string(MAX_DIR_DEPTH));
        }
        if (subName.string().length() > MAX_LAST_NAME_LENGTH) {
            return Result::Error(ResultCode::RISK_ALERT,
                "Directory/file name exceeds maximum length limit: " + std::to_string(MAX_LAST_NAME_LENGTH));
        }
    }
    return Result::OK();
}

Result SafePath::CheckFileSuffix()
{
    if (!suffix_.empty() && !IsSuffix(path_, suffix_)) {
        return Result::Error(ResultCode::INVALID_ARGUMENT, path_ + " is not a " + suffix_ + " file.");
    }
    return Result::OK();
}

Result SafePath::CheckFileSize()
{
    if (sizeLimitation_ == 0) {
        return Result::OK();
    }
    auto size = fs::file_size(path_);
    if (size > sizeLimitation_) {
        return Result::Error(ResultCode::RISK_ALERT, "File size exceeds limit: " + std::to_string(sizeLimitation_));
    }
    return Result::OK();
}

Result SafePath::CheckDirSize()
{
    if (sizeLimitation_ == 0) {
        return Result::OK();
    }
    size_t totalSize = 0;
    for (auto& p : fs::recursive_directory_iterator(path_)) {
        if (fs::is_regular_file(p.path())) {
            totalSize += fs::file_size(p.path());
            if (totalSize > sizeLimitation_) {
                return Result::Error(ResultCode::RISK_ALERT, "Directory size exceeds limit");
            }
        }
    }
    return Result::OK();
}

} // namespace mindie_llm
