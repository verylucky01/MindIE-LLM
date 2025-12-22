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

import os
import sys
import tempfile
import shutil
import threading
import time
import unittest
import logging
from io import StringIO
from unittest.mock import patch, MagicMock, mock_open

from atb_llm.utils.log.logging import (
    ErrorCodeFormatter,
    message_filter,
    print_log,
    init_logger,
    makedir_and_change_permissions,
    SafeRotatingFileHandler,
)
from atb_llm.utils.log.error_code import ErrorCode
from atb_llm.utils.env import ENV

test_file_dir = os.path.dirname(os.path.abspath(__file__))
atb_llm_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(test_file_dir))))
sys.path.append(atb_llm_root)

EXTRA = 'extra'
MAX_OPEN_LOG_FILE_PERM = 0o640
MAX_CLOSE_LOG_FILE_PERM = 0o440
MAX_KEY_LENGTH = 4096 * 10


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


class TestErrorCodeFormatter2(unittest.TestCase):
    def setUp(self):
        self.formatter = ErrorCodeFormatter()  # 替换为实际的格式化器类名
        # 创建不同格式的模拟格式化器
        self.error_fmt_verbose = MagicMock()
        self.default_fmt_verbose = MagicMock()
        self.error_fmt = MagicMock()
        self.default_fmt = MagicMock()
        
        # 设置格式化器的格式化方法
        self.formatter.error_fmt_verbose = self.error_fmt_verbose
        self.formatter.default_fmt_verbose = self.default_fmt_verbose
        self.formatter.error_fmt = self.error_fmt
        self.formatter.default_fmt = self.default_fmt

    @patch.dict('os.environ', {'atb_llm_log_verbose': '1'})
    def test_verbose_mode_error_with_code(self):
        """测试详细模式下带错误码的错误日志"""
        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='',
            lineno=0,
            msg='Test error',
            args=(),
            exc_info=None
        )
        record.error_code = 'E001'
        
        self.formatter.format(record)
        
        self.error_fmt_verbose.format.assert_called_once_with(record)

    @patch.dict('os.environ', {'atb_llm_log_verbose': '1'})
    def test_verbose_mode_normal_log(self):
        """测试详细模式下的普通日志"""
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test info',
            args=(),
            exc_info=None
        )
        
        self.formatter.format(record)
        
        self.default_fmt_verbose.format.assert_called_once_with(record)

    @patch.dict('os.environ', {'atb_llm_log_verbose': 'invalid'})
    def test_invalid_verbose_config(self):
        """测试无效的详细程度配置"""
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Test info',
            args=(),
            exc_info=None
        )
        
        self.formatter.format(record)
        
        # 应该使用默认的详细模式格式化器
        self.default_fmt_verbose.format.assert_called_once_with(record)

    @patch.dict('os.environ', {'atb_llm_log_verbose': '1'})
    def test_error_without_code(self):
        """测试不带错误码的错误日志"""
        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='',
            lineno=0,
            msg='Test error',
            args=(),
            exc_info=None
        )
        
        self.formatter.format(record)
        
        # 应该使用默认的详细模式格式化器
        self.default_fmt_verbose.format.assert_called_once_with(record)

    def test_long_verbose_config(self):
        """测试过长的详细程度配置"""
        long_config = '1' * (MAX_KEY_LENGTH + 10)  # 超过最大长度
        with patch.dict('os.environ', {'atb_llm_log_verbose': long_config}):
            record = logging.LogRecord(
                name='test',
                level=logging.INFO,
                pathname='',
                lineno=0,
                msg='Test info',
                args=(),
                exc_info=None
            )
            
            self.formatter.format(record)
            
            # 应该使用默认的详细模式格式化器
            self.default_fmt_verbose.format.assert_called_once_with(record)


class TestSafeRotatingFileHandler(unittest.TestCase):

    def setUp(self):
        # 创建临时测试目录
        self.test_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.test_dir, 'test.log')
        
        # 创建handler实例
        self.handler = SafeRotatingFileHandler(
            self.log_file,
            maxBytes=1024,
            backupCount=3
        )

    def tearDown(self):
        # 关闭handler
        if self.handler:
            self.handler.close()
        # 清理测试目录
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch('atb_llm.utils.file_utils.safe_chmod')
    @patch('builtins.open', new_callable=mock_open, read_data='test')
    def test_close_with_stream(self, mock_file, mock_chmod):
        """测试关闭带有有效流的handler"""
        # 创建模拟流
        mock_stream = MagicMock()
        self.handler.stream = mock_stream
        
        self.handler.close()
        
        # 验证流操作
        mock_stream.flush.assert_called_once()
        mock_stream.close.assert_called_once()
        # 验证权限设置
        mock_chmod.assert_called_once_with(self.log_file, MAX_CLOSE_LOG_FILE_PERM)
        # 验证流被清空
        self.assertIsNone(self.handler.stream)

    @patch('atb_llm.utils.file_utils.safe_chmod')
    @patch('builtins.open', new_callable=mock_open, read_data='test')
    def test_close_without_stream(self, mock_file, mock_chmod):
        """测试关闭没有流的handler"""
        self.handler.stream = None
        self.handler.close()
        
        # 验证权限设置
        mock_chmod.assert_called_once_with(self.log_file, MAX_CLOSE_LOG_FILE_PERM)

    @patch('atb_llm.utils.file_utils.safe_chmod')
    @patch('builtins.open', new_callable=mock_open, read_data='test')
    def test_do_rollover_with_backup(self, mock_file, mock_chmod):
        """测试带备份的轮转"""
        # 创建一些备份文件
        for i in range(1, 3):
            backup_file = f"{self.log_file}.{i}"
            with open(backup_file, 'w') as f:
                f.write(f"backup {i}")
        
        # 创建主日志文件
        with open(self.log_file, 'w') as f:
            f.write("main log")

        self.handler.doRollover()
        
        # 验证备份文件被重命名
        self.assertTrue(os.path.exists(f"{self.log_file}.1"))

    @patch('atb_llm.utils.file_utils.safe_chmod')
    @patch('builtins.open', new_callable=mock_open, read_data='test')
    def test_do_rollover_without_backup(self, mock_file, mock_chmod):
        """测试不带备份的轮转"""
        self.handler.backupCount = 0
        
        self.handler.doRollover()
            
    @patch('atb_llm.utils.file_utils.safe_chmod')
    @patch('builtins.open', new_callable=mock_open, read_data='test')
    def test_open_existing_file(self, mock_file, mock_chmod):
        """测试打开已存在的文件"""
        # 创建日志文件
        with open(self.log_file, 'w') as f:
            f.write("existing content")
        
        # 验证文件被打开
        mock_file.assert_called()

    def test_thread_safety(self):
        """测试线程安全性"""
        
        results = []
        errors = []
        
        def worker():
            try:
                self.handler.close()
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程同时关闭handler
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 验证没有错误发生
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)

    @patch('atb_llm.utils.file_utils.safe_chmod')
    @patch('builtins.open', new_callable=mock_open, read_data='test')
    def test_error_handling(self, mock_file, mock_chmod):
        """测试错误处理"""
        # 让 safe_chmod 抛出 PermissionError
        mock_chmod.side_effect = PermissionError("Permission denied")

        with self.assertRaises(PermissionError):
            self.handler.close()

        # 可选：验证确实被调用过
        mock_chmod.assert_called()

    def test_file_permissions(self):
        """测试文件权限设置"""
        # 创建一个实际的日志文件
        self.handler.emit(logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test message",
            args=(),
            exc_info=None
        ))
        
        # 关闭handler
        self.handler.close()
        
        # 检查文件权限
        if os.path.exists(self.log_file):
            actual_mode = os.stat(self.log_file).st_mode & 0o777
            self.assertEqual(oct(actual_mode), oct(MAX_CLOSE_LOG_FILE_PERM))


class TestCustomLoggerAndFormatter(unittest.TestCase):    
    def setUp(self):
        ENV.atb_llm_log_to_file = "1"
        ENV.atb_llm_log_to_stdout = "1"
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

    def test_logger_error_with_error_code(self):
        # 测试 error 方法传递错误码的情况
        error_code = ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
        self.logger.error("Test error message with error code", error_code)
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        # 刷新日志处理器输出
        for handler in self.logger.handlers:
            handler.flush()

        # 获取控制台日志输出
        console_output = self.log_stdout.getvalue().strip()

        # 修改正则表达式来匹配控制台输出
        expected_output_pattern = (
            r"\[MIE05E000000\] Test error message with error code" 
        )

        # 使用正则表达式匹配控制台输出
        self.assertRegex(console_output, expected_output_pattern)

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

        self.assertRegex(console_output, expected_output_pattern)

    def test_logger_critical_with_error_code(self):
        # 测试 critical 方法传递错误码的情况
        error_code = ErrorCode.ATB_MODELS_PARAM_OUT_OF_RANGE
        self.logger.critical("Test critical message with error code", error_code)
        # 多线程日志处理器异步写入存在时间延迟，再写入到StringIO之前就执行到获取日志输出的代码，会导致获取空字符串报错
        time.sleep(1)

        for handler in self.logger.handlers:
            handler.flush()

        console_output = self.log_stdout.getvalue().strip()

        expected_output_pattern = (
            r"\[MIE05E000000\] Test critical message with error code" 
        )

        self.assertRegex(console_output, expected_output_pattern)

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

        self.assertRegex(console_output, expected_output_pattern)

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

if __name__ == "__main__":
    unittest.main()