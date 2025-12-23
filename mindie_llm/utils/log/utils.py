# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.

MB = 1024 * 1024
MAX_LOG_FILE_SIZE_MB = 500          # 最大文件大小：500MB
DEFAULT_MAX_FILE_SIZE_MB = 20       # 默认轮转文件大小：20MB
DEFAULT_MAX_FILES = 10           # 默认轮转文件个数：10个


def update_log_file_param(
    rotate_config: str,
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB,
    max_files: int = DEFAULT_MAX_FILES
) -> tuple[int, int]:
    """
    更新日志文件参数
    
    Args:
        rotate_config: 配置字符串，格式如 "-fs 100 -r 5"
        max_file_size_mb: 最大文件大小（MB）
        max_files: 最大文件数量
    
    Returns:
        tuple[int, int]: (max_file_size_bytes, max_files)
    """
    max_file_size_bytes = max_file_size_mb * MB
    
    if not rotate_config:
        return max_file_size_bytes, max_files

    def validate_numeric_value(s: str, param_name: str) -> int:
        """验证并转换数字字符串
        
        Args:
            s: 待验证的字符串
            param_name: 参数名称（用于错误信息）
        
        Returns:
            int: 转换后的数字
        
        Raises:
            ValueError: 当输入不是有效数字时，包含原始错误信息
        """
        try:
            return int(s)
        except ValueError as e:
            raise ValueError(
                f"{param_name} should be an integer, "
                f"but got '{s}'. Original error: {str(e)}"
            ) from e

    # 将配置字符串分割成列表
    config_list = rotate_config.split()
    
    # 遍历配置，每次取两个元素（选项和值）
    for i in range(0, len(config_list), 2):
        if i + 1 >= len(config_list):
            continue
            
        option = config_list[i]
        value = config_list[i + 1]

        if option == "-fs":
            file_size_mb = validate_numeric_value(value, "Log file size (-fs)")
            if not (1 <= file_size_mb <= MAX_LOG_FILE_SIZE_MB):
                raise ValueError(
                    f"Log file size (-fs) should be between 1 and {MAX_LOG_FILE_SIZE_MB} MB, "
                    f"but got {file_size_mb} MB."
                )
            max_file_size_mb = file_size_mb
            max_file_size_bytes = max_file_size_mb * MB
        elif option == "-r":
            files = validate_numeric_value(value, "Log rotation count (-r)")
            if not (1 <= files <= 64):
                raise ValueError(
                    f"Log rotation count (-r) should be between 1 and 64, "
                    f"but got {files}."
                )
            max_files = files
    
    return max_file_size_bytes, max_files