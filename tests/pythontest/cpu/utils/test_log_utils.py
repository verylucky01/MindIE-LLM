# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import unittest
from unittest.mock import patch
from mindie_llm.utils.log.utils import update_log_file_param


class TestUpdateLogFileParam(unittest.TestCase):
    def setUp(self):
        """每个测试用例执行前的设置"""
        self.default_size = 20 * 1024 * 1024
        self.default_files = 10

    def test_empty_config(self):
        """测试空配置字符串"""
        result = update_log_file_param("")
        self.assertEqual(result, (self.default_size, self.default_files))

    def test_valid_file_size_config(self):
        """测试有效的文件大小配置"""
        result = update_log_file_param("-fs 100")
        self.assertEqual(result, (100 * 1024 * 1024, self.default_files))

    def test_valid_rotation_config(self):
        """测试有效的轮转次数配置"""
        result = update_log_file_param("-r 5")
        self.assertEqual(result, (self.default_size, 5))

    def test_valid_both_config(self):
        """测试同时配置文件大小和轮转次数"""
        result = update_log_file_param("-fs 50 -r 8")
        self.assertEqual(result, (50 * 1024 * 1024, 8))

    def test_invalid_file_size_zero(self):
        """测试文件大小为0"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-fs 0")
        self.assertIn("should be between 1 and 500 MB", str(context.exception))
        self.assertIn("but got 0 MB", str(context.exception))

    def test_invalid_file_size_negative(self):
        """测试文件大小为负数"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-fs -1")
        self.assertIn("should be between 1 and 500 MB", str(context.exception))
        self.assertIn("but got -1 MB", str(context.exception))

    def test_invalid_file_size_too_large(self):
        """测试文件大小过大"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-fs 501")
        self.assertIn("should be between 1 and 500 MB", str(context.exception))
        self.assertIn("but got 501 MB", str(context.exception))

    def test_invalid_rotation_zero(self):
        """测试轮转次数为0"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-r 0")
        self.assertIn("should be between 1 and 64", str(context.exception))
        self.assertIn("but got 0", str(context.exception))

    def test_invalid_rotation_negative(self):
        """测试轮转次数为负数"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-r -1")
        self.assertIn("should be between 1 and 64", str(context.exception))
        self.assertIn("but got -1", str(context.exception))

    def test_invalid_rotation_too_large(self):
        """测试轮转次数过大"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-r 65")
        self.assertIn("should be between 1 and 64", str(context.exception))
        self.assertIn("but got 65", str(context.exception))

    def test_non_numeric_file_size(self):
        """测试文件大小为非数字"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-fs abc")
        self.assertIn("should be an integer", str(context.exception))
        self.assertIn("but got 'abc'", str(context.exception))

    def test_non_numeric_rotation(self):
        """测试轮转次数为非数字"""
        with self.assertRaises(ValueError) as context:
            update_log_file_param("-r xyz")
        self.assertIn("should be an integer", str(context.exception))
        self.assertIn("but got 'xyz'", str(context.exception))

    def test_incomplete_file_size_config(self):
        """测试不完整的文件大小配置"""
        result = update_log_file_param("-fs")
        self.assertEqual(result, (self.default_size, self.default_files))

    def test_incomplete_rotation_config(self):
        """测试不完整的轮转配置"""
        result = update_log_file_param("-r")
        self.assertEqual(result, (self.default_size, self.default_files))

    def test_invalid_option(self):
        """测试无效的选项"""
        result = update_log_file_param("-x 100 -y 10")
        self.assertEqual(result, (self.default_size, self.default_files))

    def test_boundary_values(self):
        """测试边界值"""
        # 测试最小值
        result = update_log_file_param("-fs 1 -r 1")
        self.assertEqual(result, (1 * 1024 * 1024, 1))
        
        # 测试最大值
        result = update_log_file_param("-fs 500 -r 64")
        self.assertEqual(result, (500 * 1024 * 1024, 64))

    def test_multiple_configs(self):
        """测试多个配置项，只使用最后一个有效值"""
        result = update_log_file_param("-fs 100 -fs 200 -r 5 -r 10")
        self.assertEqual(result, (200 * 1024 * 1024, 10))

    def test_config_with_extra_spaces(self):
        """测试带有多余空格的配置"""
        result = update_log_file_param("  -fs  100  -r  5  ")
        self.assertEqual(result, (100 * 1024 * 1024, 5))

    def test_config_with_newlines(self):
        """测试带有换行符的配置"""
        result = update_log_file_param("-fs 100\n-r 5")
        self.assertEqual(result, (100 * 1024 * 1024, 5))

    def test_config_with_tabs(self):
        """测试带有制表符的配置"""
        result = update_log_file_param("-fs\t100\t-r\t5")
        self.assertEqual(result, (100 * 1024 * 1024, 5))

    @patch('builtins.print')
    def test_error_messages(self, mock_print):
        """测试错误信息的格式"""
        with self.assertRaises(ValueError):
            update_log_file_param("-fs 0")
        
        # 验证错误信息格式
        try:
            update_log_file_param("-fs 0")
        except ValueError as e:
            error_msg = str(e)
            self.assertIn("Log file size (-fs)", error_msg)
            self.assertIn("should be between 1 and 500 MB", error_msg)
            self.assertIn("but got 0 MB", error_msg)

    def test_large_file_size(self):
        """测试大文件大小的处理"""
        result = update_log_file_param("-fs 500")
        self.assertEqual(result, (500 * 1024 * 1024, self.default_files))

    def test_large_rotation_count(self):
        """测试大轮转次数的处理"""
        result = update_log_file_param("-r 64")
        self.assertEqual(result, (self.default_size, 64))

if __name__ == '__main__':
    unittest.main()
