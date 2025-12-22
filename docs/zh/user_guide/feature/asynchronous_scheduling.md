# 异步调度

MindIE推理的过程是同步执行，一次推理的过程按照在CPU/NPU上执行可以分为以下三个阶段：

-  数据准备阶段（CPU上执行）
-  模型推理阶段（NPU上执行）
-  数据返回阶段（CPU上执行）

异步调度的原理是使用模型推理阶段的耗时掩盖数据准备阶段和数据返回阶段的耗时，即使用NPU上执行的时间掩盖CPU上执行的时间，以及Sampling之外的CPU耗时，但是已经EOS（终止推理）的请求会被重复计算一次，造成NPU计算资源和显存资源有部分浪费。该特性适用于maxBatchSize较大，且输入输出长度比较长的场景。

## 限制与约束

-  支持PD混部和PD分离场景。
-  该特性不能和Look Ahead、Memory Decoding同时使用。
-  该特性暂不支持n、best\_of、use\_beam\_search等与多序列推理相关的后处理参数。

## 执行推理

1. 设置环境变量，打开异步调度功能。

    ```
    export MINDIE_ASYNC_SCHEDULING_ENABLE=1
    ```

    > [!NOTE]说明 
    > PD分离部署场景下，请仅在D节点设置环境变量打开异步调度功能。

2. 打开Server的config.json文件。

    ```
    cd {MindIE安装目录}/latest/mindie-service/
    vi conf/config.json
    ```

3. 配置服务化参数。服务化参数说明请参见[配置参数说明（服务化）](../user_manual/service_parameter_configuration.md)章节。
4. 启动服务。

    ```
    ./bin/mindieservice_daemon
    ```

5. 执行如下命令，使用MindIE Benchmark工具开始调优，MindIE Benchmark参数详细说明请参见《MindIE Motor开发指南》中的“MindIE Benchmark \> 输入参数”章节。

    ```
    benchmark \
    --DatasetPath "/{数据集路径}/GSM8K" \
    --DatasetType "gsm8k" \
    --ModelName "llama3-8b" \
    --ModelPath "/{模型权重路径}/Meta-Llama-3-8B" \
    --TestType client \
    --Http https://{ipAddress}:{port} \
    --ManagementHttp https://{managementIpAddress}:{managementPort} \
    --Tokenizer True \
    --MaxOutputLen 512 \
    --TaskKind stream \
    --WarmupSize 1 \
    --DoSampling False  \
    --Concurrency 200 \
    --TestAccuracy True\
    ```

