#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
logging
"""
import logging
import time
import os
import argparse
import threading
import re
from logging import StreamHandler
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta

from mindie_llm.utils.log.utils import update_log_file_param
from .. import file_utils
from ..env import ENV
from .error_code import ErrorCode

MAX_LOG_DIR_PERM = 0o750
MAX_OPEN_LOG_FILE_PERM = 0o640
MAX_CLOSE_LOG_FILE_PERM = 0o440

MAX_KEY_LENGTH = 4096 * 10

MAX_MSG_LEN = 4096
EXTRA = 'extra'

COMPONENT = "LLMMODELS"
TRUE = "TRUE"
FALSE = "FALSE"

MINDIE = "mindie"
LOG = "log"
DEBUG_PATH = "debug"

SPECIAL_CHARS = [
    '\n', '\r', '\f',
    '\t', '\v', '\b',
    "//", "\\", "&",
    '\u000A', '\u000D', '\u000C',
    '\u000B', '\u0008', '\u007F',
]

log_lock = threading.Lock()


class CustomLogger(logging.Logger):
    def error(self, msg, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ErrorCode):
            error_code = args[0]
            kwargs[EXTRA] = kwargs.get(EXTRA, {})
            kwargs[EXTRA]['error_code'] = error_code
            args = args[1:]

        super().error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ErrorCode):
            error_code = args[0]
            kwargs[EXTRA] = kwargs.get(EXTRA, {})
            kwargs[EXTRA]['error_code'] = error_code
            args = args[1:]

        super().critical(msg, *args, **kwargs)


class ErrorCodeFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)
        logging.addLevelName(logging.WARNING, 'WARN')
        self.error_fmt_verbose = logging.Formatter(
        '[%(asctime)s] [%(process)d] [%(thread)d] [llmmodels] [%(levelname)s] '
        '[%(filename)s-%(lineno)d] : [%(error_code)s] %(message)s'
        )
        self.default_fmt_verbose = logging.Formatter(
        '[%(asctime)s] [%(process)d] [%(thread)d] [llmmodels] [%(levelname)s] '
        '[%(filename)s-%(lineno)d] : %(message)s'
        )
        self.error_fmt = logging.Formatter(
        '[%(asctime)s] : [%(levelname)s] [%(error_code)s] %(message)s'
        )
        self.default_fmt = logging.Formatter(
        '[%(asctime)s] : [%(levelname)s] %(message)s'
        )

    def format(self, record):
        log_verbose_list = ['0', '1', FALSE, TRUE]
        log_verbose = ENV.atb_llm_log_verbose
        if len(log_verbose) >= MAX_KEY_LENGTH:
            log_verbose = log_verbose[:MAX_KEY_LENGTH]
        log_verbose = standard_env(log_verbose)
        if log_verbose.upper() not in log_verbose_list:
            log_verbose = "1"
        if str(log_verbose).upper() in ["1", TRUE]:
            if record.levelno >= logging.ERROR and hasattr(record, 'error_code'):
                formatter = self.error_fmt_verbose
            else:
                formatter = self.default_fmt_verbose   
        else:
            if record.levelno >= logging.ERROR and hasattr(record, 'error_code'):
                formatter = self.error_fmt
            else:
                formatter = self.default_fmt

        return formatter.format(record)


class SafeRotatingFileHandler(RotatingFileHandler):
    def close(self):
        """
        Override:
            Close the stream and set the file permissions.
        """
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                StreamHandler.close(self)
        finally:
            file_utils.safe_chmod(self.baseFilename, MAX_CLOSE_LOG_FILE_PERM)
            self.release()
    
    def doRollover(self):
        """
        Override:
            Do a rollover and modify the permissions of old files.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d" % (self.baseFilename, i + 1))
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
                    file_utils.safe_chmod(dfn, MAX_CLOSE_LOG_FILE_PERM)
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if os.path.exists(dfn):
                os.remove(dfn)
            self.rotate(self.baseFilename, dfn)
            file_utils.safe_chmod(dfn, MAX_CLOSE_LOG_FILE_PERM)
        if not self.delay:
            self.stream = self._open()

    def _open(self):
        """
        Override:
            Open the current base file with the (original) mode and encoding.
            Modify the permissions of current files.
            Return the resulting stream.
        """
        if os.path.exists(self.baseFilename):
            file_utils.safe_chmod(self.baseFilename, MAX_OPEN_LOG_FILE_PERM)
        ret = self._builtin_open(self.baseFilename, self.mode, encoding=self.encoding, errors=self.errors)
        file_utils.safe_chmod(self.baseFilename, MAX_OPEN_LOG_FILE_PERM)
        return ret


def standard_env(env_variable):
    for component in env_variable.split(";")[::-1]:
        split_list = component.split(":")
        if len(split_list) == 1:
            env_variable = split_list[0].strip(" ")
            break
        elif len(split_list) == 2 and split_list[0].strip(" ").upper() == COMPONENT:
            env_variable = split_list[1].strip(" ")
            break
        else:
            env_variable = ""
    return env_variable


def makedir_and_change_permissions(path, mode=MAX_LOG_DIR_PERM):
    parts = path.strip(os.sep).split(os.sep)    
    current_path = os.sep
    
    for part in parts:
        current_path = os.path.join(current_path, part)
        if not os.path.exists(current_path):
            os.makedirs(current_path, mode, exist_ok=True)


def get_log_rotate_config() -> tuple[int, int]:
    """获取日志配置参数
    
    Returns:
        tuple[int, int]: (最大文件大小, 最大文件数量)
    """
    # 1. 获取并处理日志轮转配置
    log_rotate_config = standard_env(ENV.log_file_rotate)
    max_size, max_files = update_log_file_param(log_rotate_config)
    
    # 2. 处理旧版环境变量
    if ENV.atb_llm_log_maxsize:
        log_deprecation_warning()
        if ENV.atb_llm_log_maxsize < 0 or ENV.atb_llm_log_maxsize > 524288000:   # 500MB
            raise ValueError(
                "PYTHON_LOG_MAXSIZE should be a number in the range of 0 to 524288000 (500MB).\n"
            )
        
        max_size = ENV.atb_llm_log_maxsize
    
    return max_size, max_files


def log_deprecation_warning() -> None:
    """记录环境变量弃用警告"""
    logging.warning(
        "Note: The old environment variable PYTHON_LOG_MAXSIZE will be deprecated on 2026/12/31.\n"
        "Please use the new environment variable MINDIE_LOG_ROTATE instead.\n"
        "Usage: export MINDIE_LOG_ROTATE=\"-fs 20 -r 10\"\n"
        "Where:\n"
        "  -fs: maximum size of each log file in MB (range: [1, 500])\n"
        "  -r : maximum number of log files per process (range: [1, 64])"
    )


def init_logger(logger_ins: CustomLogger, file_name: str, stream=None):
    """
    日志初始化
    :param logger:
    :param file_name:
    :return:
    """
    pid = os.getpid()
    milliseconds = str(time.time() * 1000)
    process_datetime = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M%S") + milliseconds[0:3]

    # LOG_LEVEL校验
    log_level_list = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"]
    log_level = ENV.atb_llm_log_level
    if len(log_level) >= MAX_KEY_LENGTH:
        log_level = log_level[:MAX_KEY_LENGTH]
    log_level = standard_env(log_level)
    if log_level.upper() not in log_level_list:
        log_level = "INFO"
    
    logger_ins.setLevel(logging.getLevelName(log_level.upper()))

    # LOG_PATH校验
    ## 根目录校验
    home_dir = os.path.expanduser("~")
    ## 路径长度校验
    if len(home_dir) >= MAX_KEY_LENGTH:
        home_dir = home_dir[:MAX_KEY_LENGTH]

    ## 存在性校验
    if not file_utils.is_path_exists(home_dir):
        raise FileNotFoundError(f"Home directory {home_dir} does not exist or access denied! " 
                            "Please manually set the log storage path "
                            f"or change the home directory {home_dir} permission.")
    ## 路径标准化&权限校验
    home_dir = file_utils.standardize_path(home_dir)
    file_utils.check_path_permission(home_dir)
    
    log_file_path = file_name
    if len(log_file_path) >= MAX_KEY_LENGTH:
        log_file_path = log_file_path[:MAX_KEY_LENGTH]
    log_file_path = standard_env(log_file_path)

    if len(log_file_path) == 0:
        log_file_path = os.path.join(home_dir, MINDIE, LOG, DEBUG_PATH)
    elif not log_file_path.startswith("/"):
        log_file_path = os.path.join(home_dir, MINDIE, LOG, log_file_path, DEBUG_PATH)
    else:
        log_file_path = os.path.join(log_file_path, LOG, DEBUG_PATH)
    llm_log_file_path = os.path.join(log_file_path, f"mindie-llmmodels_{pid}_{process_datetime}.log")
   
    # LOG_TO_FILE校验
    log_to_file_list = ['0', '1', FALSE, TRUE]
    log_to_file = ENV.atb_llm_log_to_file
    if len(log_to_file) >= MAX_KEY_LENGTH:
        log_to_file = log_to_file[:MAX_KEY_LENGTH]
    log_to_file = standard_env(log_to_file)
    if log_to_file.upper() not in log_to_file_list:
        log_to_file = '1'

    if str(log_to_file).upper() in ["1", TRUE]:
        makedir_and_change_permissions(log_file_path)
        log_file_path = file_utils.standardize_path(log_file_path)
        file_utils.check_path_permission(log_file_path)

        prepare_log_path(llm_log_file_path)

        # 获取日志轮转配置
        log_file_maxsize, log_file_maxnum = get_log_rotate_config()

        # 创建日志记录器，指明日志保存路径,每个日志的大小，保存日志的上限
        file_handle = SafeRotatingFileHandler(
            filename=llm_log_file_path,
            maxBytes=log_file_maxsize,
            backupCount=log_file_maxnum,
            delay=True)
        
        # 为全局的日志工具对象添加日志记录器
        file_formatter = ErrorCodeFormatter()
        file_handle.setFormatter(file_formatter)
        logger_ins.addHandler(file_handle)

    #LOG_TO_STDOUT校验
    log_to_stdout_list = ['0', '1', FALSE, TRUE]
    log_to_stdout = ENV.atb_llm_log_to_stdout
    if len(log_to_stdout) >= MAX_KEY_LENGTH:
        log_to_stdout = log_to_stdout[:MAX_KEY_LENGTH]
    log_to_stdout = standard_env(log_to_stdout)

    if log_to_stdout.upper() not in log_to_stdout_list:
        log_to_stdout = '0'

    if str(log_to_stdout).upper() in ["1", TRUE]: 
        # 添加控制台输出日志
        console_formatter = ErrorCodeFormatter()
        console_handle = logging.StreamHandler(stream)
        console_handle.setFormatter(console_formatter)
        if hasattr(console_handle, 'encoding'):
            console_handle.encoding = 'utf-8'

        logger_ins.addHandler(console_handle)

    logger_ins.propagate = False
    return logger_ins


def prepare_log_path(input_path):
    file_path = file_utils.standardize_path(input_path)
    if os.path.isdir(file_path):
        raise argparse.ArgumentTypeError(
            "'log_to_file' only supports paths that end with a file."
        )
    dirs = os.path.dirname(file_path)
    try:
        log_lock.acquire()

        if os.path.exists(file_path):
            # Owner and OTH-permission check
            file_utils.check_path_permission(file_path)
        elif os.path.exists(dirs):
            if file_utils.has_owner_write_permission(dirs):
                # Owner and OTH-permission check
                file_utils.check_path_permission(dirs)
            else:
                raise PermissionError("{dirs} should have write permission.")
        else:
            try:
                os.makedirs(dirs, mode=MAX_LOG_DIR_PERM, exist_ok=True)
            except PermissionError as e:
                err_msg = (
                    f"Failed to create the log directory: {dirs}."
                    "Please add write permissions to the parent directory."
                )
                raise PermissionError(err_msg) from e
    finally:
        log_lock.release()

logging.setLoggerClass(CustomLogger)
logger = init_logger(logging.getLogger(__name__), ENV.atb_llm_log_path)



def message_filter(msg: str):
    """
    Truncate message exceeding the limit and filter special characters.
    """
    if len(msg) > MAX_MSG_LEN:
        msg = msg[:MAX_MSG_LEN] + '...'
    for item in SPECIAL_CHARS:
        msg = msg.replace(item, ' ')
    msg = re.sub(r' {5,}', '    ', msg)
    return msg


def print_log(rank_id, logger_fn, msg, need_filter=False):
    if rank_id != 0:
        return
    if need_filter:
        msg = message_filter(str(msg))
    logger_fn(msg, stacklevel=2)
