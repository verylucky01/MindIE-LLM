#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
import logging
import tempfile
import shutil
from unittest.mock import patch
from io import StringIO
import os
import time

from mindie_llm.utils.log.logging import (
    ErrorCodeFormatter,
    SafeRotatingFileHandler,
    message_filter,
    print_log,
    init_logger,
    standard_env,
    makedir_and_change_permissions,
    MAX_CLOSE_LOG_FILE_PERM
)


from mindie_llm.utils.env import ENV

EXTRA = 'extra'


class TestSafeRotatingFileHandlerRollover(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.test_dir, "test.log")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('mindie_llm.utils.file_utils.safe_chmod')
    def test_doRollover_creates_backup_and_sets_permissions(self, mock_safe_chmod):
        # 初始化 handler，设置较小的 maxBytes 便于测试
        handler = SafeRotatingFileHandler(
            filename=self.log_file,
            maxBytes=100,  # 很小，方便触发 rollover
            backupCount=3
        )

        # 先写入一些内容到主日志文件
        with open(self.log_file, 'w') as f:
            f.write("Initial content\n")

        # 模拟已有 .1 和 .2 备份文件
        with open(self.log_file + ".1", 'w') as f:
            f.write("Backup 1\n")
        with open(self.log_file + ".2", 'w') as f:
            f.write("Backup 2\n")

        # 直接调用 doRollover（绕过自动触发）
        handler.doRollover()

        # 验证文件重命名行为
        # doRollover 实现中只处理到 backupCount - 1，且从高往低移
        # 实际行为：.2 → .3（如果 backupCount > 2），代码中 backupCount=3 时：
        #   i in [2, 1] → sfn=log.2 → dfn=log.3; sfn=log.1 → dfn=log.2
        # 然后 base → log.1

        self.assertTrue(os.path.exists(self.log_file + ".3"))  # 原 .2
        self.assertTrue(os.path.exists(self.log_file + ".2"))  # 原 .1
        self.assertTrue(os.path.exists(self.log_file + ".1"))  # 原 base
        self.assertTrue(os.path.exists(self.log_file))         # 新 base（空或带 header）

        # 验证 safe_chmod 被调用：每个备份文件 + 新 .1
        # 应该对 .3, .2, .1 调用 safe_chmod(..., MAX_CLOSE_LOG_FILE_PERM)
        expected_calls = [
            unittest.mock.call(self.log_file + ".3", MAX_CLOSE_LOG_FILE_PERM),
            unittest.mock.call(self.log_file + ".2", MAX_CLOSE_LOG_FILE_PERM),
            unittest.mock.call(self.log_file + ".1", MAX_CLOSE_LOG_FILE_PERM),
        ]
        mock_safe_chmod.assert_has_calls(expected_calls, any_order=True)

        handler.close()


class TestMakeDirAndChangePermissions(unittest.TestCase): 
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_makedir_and_change_permissions(self, mock_exists, mock_makedirs):
        mock_exists.return_value = False
        
        path = 'test/dir/structure'
        mode = 0o750
        
        makedir_and_change_permissions(path, mode)
        
        parts = path.strip(os.sep).split(os.sep)
        current_path = os.sep
        for part in parts:
            current_path = os.path.join(current_path, part)
            mock_exists.assert_any_call(current_path)
            mock_makedirs.assert_any_call(current_path, mode, exist_ok=True)


class TestStandardEnv(unittest.TestCase):
    def setUp(self):
        return super().setUp()
    
    def tearDown(self):
        return super().tearDown()
    
    def test_standard_env_with_1(self):
        env_variable = "1"
        env_variable = standard_env(env_variable)
        self.assertEqual(env_variable, "1")
    
    def test_standard_env_with_llm0(self):
        env_variable = "llm:0"
        env_variable = standard_env(env_variable)
        self.assertEqual(env_variable, "0")

    def test_standard_env_with_server0(self):
        env_variable = "server:0"
        env_variable = standard_env(env_variable)
        self.assertEqual(env_variable, "")

    def test_standard_env_with_servererrorllminfo(self):
        env_variable = "server:error;llm:info"
        env_variable = standard_env(env_variable)
        self.assertEqual(env_variable, "info")

    def test_standard_env_with_servererrorllminfollmwarn(self):
        env_variable = "server:error;llm:info;llm:warn"
        env_variable = standard_env(env_variable)
        self.assertEqual(env_variable, "warn")

    def test_standard_env_with_servererrorllminfodebug(self):
        env_variable = "server:error;llm:info;debug"
        env_variable = standard_env(env_variable)
        self.assertEqual(env_variable, "debug")

    def test_standard_env_with_infollmdebug(self):
        env_variable = "info;llm : debug"
        env_variable = standard_env(env_variable)
        self.assertEqual(env_variable, "debug")


class TestCustomLoggerAndFormatter(unittest.TestCase):
    
    def setUp(self):
        ENV.log_to_file = "1"
        ENV.log_to_stdout = "1"
        ENV.log_file_maxnum = 10
        ENV.log_file_maxsize = 1000
        self.log_stdout = StringIO()
        self.log_file = 'test_log.log'
        self.logger = init_logger(logging.getLogger(__name__), self.log_file, self.log_stdout)
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        # 清理 handlers 和删除日志文件
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def test_logger_error_without_error_code(self):
        # 测试 error 方法不传递错误码的情况
        self.logger.error("Test error message without error code")
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        for handler in self.logger.handlers:
            handler.flush()

        console_output = self.log_stdout.getvalue().strip()

        expected_output_pattern = (
            "Test error message without error code"  
        )

        self.assertIn(expected_output_pattern, console_output)

    def test_logger_critical_without_error_code(self):
        # 测试 critical 方法传递错误码的情况
        self.logger.critical("Test critical message without error code")
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        for handler in self.logger.handlers:
            handler.flush()

        console_output = self.log_stdout.getvalue().strip()

        expected_output_pattern = (
            "Test critical message without error code"
        )

        self.assertIn(expected_output_pattern, console_output)

    def test_logger_info_without_error_code(self):
        self.logger.info("Test info message")
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        for handler in self.logger.handlers:
            handler.flush()

        console_output = self.log_stdout.getvalue().strip()

        expected_output_pattern = (
            "Test info message" 
        )

        self.assertIn(expected_output_pattern, console_output)


class TestErrorCodeFormatter(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('test_logger')
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.formatter = ErrorCodeFormatter('%(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    @patch.dict('os.environ', {'log_verbose': 'llm:1'})  # 模拟环境变量 log_verbose 为 '1'
    def test_error_code_verbose_format(self):
        record = logging.LogRecord('test_logger', logging.ERROR, 'test_file.py', 42, 'Test error message', [], None)
        record.error_code = '12345'
        self.logger.handle(record)

        self.assertIn('Test error message', self.stream.getvalue())

    @patch.dict('os.environ', {'log_verbose': '0'})  # 模拟环境变量 log_verbose 为 '0'
    def test_error_code_non_verbose_format(self):
        record = logging.LogRecord('test_logger', logging.ERROR, 'test_file.py', 42, 'Test error message', [], None)
        record.error_code = '12345'
        self.logger.handle(record)

        self.assertIn('[12345]', self.stream.getvalue())

    @patch.dict('os.environ', {'log_verbose': '1'})  # 模拟环境变量 log_verbose 为 '1'
    def test_info_log_format(self):
        record = logging.LogRecord('test_logger', logging.INFO, 'test_file.py', 42, 'Test info message', [], None)
        self.logger.handle(record)

        self.assertIn('Test info message', self.stream.getvalue())

    @patch.dict('os.environ', {'log_verbose': '0'})  # 模拟环境变量 log_verbose 为 '0'
    def test_default_log_format(self):
        record = logging.LogRecord('test_logger', logging.INFO, 'test_file.py', 42, 'Test default message', [], None)
        self.logger.handle(record)

        self.assertIn('Test default message', self.stream.getvalue())
    
    # 测试超长字符串
    def test_message_length(self):
        long_msg = "llmm" * 1025
        filtered_msg = message_filter(long_msg)
        expect_msg = "llmm" * 1024 + "..."
        self.assertEqual(filtered_msg, expect_msg)

    #测试正常长度字符串
    def test_message_no_truncation(self):
        normal_msg = "Short message"
        filtered_msg = message_filter(normal_msg)
        self.assertEqual(filtered_msg, normal_msg)

    # 测试特殊字符是否被替换为空格
    def test_special_char_removal(self):
        msg_with_special_chars = "This is a message with \n special \t characters \u000D and \r signs."
        filtered_msg = message_filter(msg_with_special_chars)
        self.assertEqual(filtered_msg, "This is a message with   special   characters   and   signs.")

    # 测试多个连续空格是否被替换为 4 个空格
    def test_multiple_spaces_replacement(self):
        msg_with_spaces = "This    is    a    message    with    excessive    spaces."
        filtered_msg = message_filter(msg_with_spaces)
        self.assertEqual(filtered_msg, "This    is    a    message    with    excessive    spaces.")

    # 测试消息中包含特殊字符、空格的情况
    def test_mixed_cases(self):
        mixed_msg = "A long message with special \v characters \u000C and multiple    spaces."
        filtered_msg = message_filter(mixed_msg)
        self.assertEqual(filtered_msg, "A long message with special   characters   and multiple    spaces.")
    
    @patch('builtins.print')
    def test_print_log_rank_id_not_zero(self, mock_print):
        """测试 rank_id 不等于 0 时，log 不被打印"""
        print_log(rank_id=1, logger_fn=mock_print, msg="Test message", need_filter=False)
        mock_print.assert_not_called()

    @patch('builtins.print')  # 模拟 print 函数
    def test_print_log_no_filter(self, mock_print):
        """测试 rank_id 为 0 且 need_filter 为 False 时，logger_fn 被调用"""
        print_log(rank_id=0, logger_fn=mock_print, msg="Test message", need_filter=False)
        mock_print.assert_called_once_with("Test message", stacklevel=2)  # 打印时应该调用 logger_fn
    
    @patch('builtins.print')  # 模拟 print 函数
    def test_print_log_with_filter(self, mock_print):
        """测试 rank_id 为 0 且 need_filter 为 True 时, message_filter 被调用"""
        print_log(rank_id=0, logger_fn=mock_print, msg="Test message", need_filter=True)
        mock_print.assert_called_once_with("Test message", stacklevel=2)  # 确保经过过滤后的消息被传递给 logger_fn
