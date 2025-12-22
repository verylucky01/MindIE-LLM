# Lookahead运行方法

## 1. 当前支持模型
qwen-7B

## 2. 使用run_generator.py运行
### 2.1 执行脚本代码示例
    export PYTHONPATH=/LLM代码路径/examples/atb_models/:$PYTHONPATH
    export PYTHONPATH=/LLM代码路径/:$PYTHONPATH

    # 设置执行时运行在哪些NPU上
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2

    export INF_NAN_MODE_ENABLE=0
    export ATB_OPERATION_EXECUTE_ASYNC=1
    export TASK_QUEUE_ENABLE=1
    export ATB_CONVERT_NCHW_TO_ND=1

    export LCCL_DETERMINISTIC=0
    export HCCL_DETERMINISTIC=false
    export ATB_MATMUL_SHUFFLE_K_ENABLE=1
    export ATB_LLM_LCOC_ENABLE=1

    # 适配长序列场景的环境变量
    export LONG_SEQ_ENABLE=1

    # LOG开关按需设置（推荐使用）
    export MINDIE_LOG_LEVEL=INFO
    export MINDIE_LOG_TO_FILE=1
    export MINDIE_LOG_TO_STDOUT=0

    # LOG开关按需打开, 若需要打印LA优化效果数据，LOG_LEVEL可设置为DEBUG（下方环境变量支持到2025.06）
    export ATB_LOG_LEVEL=0
    export LOG_LEVEL=INFO
    export ATB_LOG_TO_FILE=0
    export ASDOPS_LOG_LEVEL=0
    export ASDOPS_LOG_TO_FILE=0
    export MINDIE_LLM_PYTHON_LOG_TO_STDOUT=1
    export MINDIE_LLM_PYTHON_LOG_LEVEL=INFO

    # 1表示执行的NPU数目，16表示最大batchsize，可按需调整
    torchrun \
    --nproc_per_node 1 \
    --master_port 25632 \
    -m mindie_llm.examples.run_generator \
    --model_path /模型路径/ \
    --max_batch_size 16 \
    --max_output_length 20 \
    --npu_mem 10 \
    --plugin_params "{\"plugin_type\": \"la\", \"level\": 4, \"window\": 5, \"guess_set_size\": 5}" \
    --speculation_gamma 30

### 2.2 参数配置
参数在run_generator.py中修改，主要涉及的参数如下（推荐配置，可按需修改）：

model_config中：

    plugin_type: la
    level: 4
    window: 5
    guess_set_size: 5

    # 若出现OOM可适当调小npu_men
    npu_mem:10

parse_arguments的配置示例

    '--input_texts'：
    default=["登鹳雀楼->王之涣\n夜雨寄北->", "Hamlet->Shakespeare\nOne hundred Years of Solitude->","登鹳雀楼->王之涣\n夜雨寄北->", "Hamlet->Shakespeare\nOne hundred Years of Solitude->"]

    '--max_output_length'：256

### 2.3 运行
1. 编译：代码根目录下执行 bash scripts/build.sh
2. source output/set_env.sh
3. 执行2.1中的脚本。

## 3. 连接服务化执行

### 3.1 连接调度 llm_engin_test
#### 3.1.1 config文件配置
此处只呈现启动LA需要额外配置的字段

"ModelDeployConfig"下添加：
    
    "speculationGamma": 30
"ModelDeployConfig"中的"ModelConfig"下添加：

    "plugin_params": "{\"plugin_type\":\"la\",\"level\": 4,\"window\": 5,\"guess_set_size\": 5}"
    

[说明]<br>
N: level<br>
W: window<br>
G: guess_set_size<br>
speculationGamma: (N-1)*(W+G)

#### 3.1.2 执行
    ./llm_engin_test token_input.csv
其中，.csv文件为输入token文件

### 3.2 对话验证 mindieservice_daemon
#### 3.2.1 config配置
同3.1.1
#### 3.2.2 执行
1. 执行./mindieservice_daemon
2. 执行对话请求<br>
对话请求demo代码如下：

        curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{
            "inputs": "登鹳雀楼->王之涣\n夜雨寄北->"
            "parameters": {
                "best_of": 1,
                "decoder_input_details": false,
                "details": false,
                "do_sample": false,
                "max_new_tokens": 256,
                "repetition_penalty": 1,
                "return_full_text": false,
                "seed": null,
                "stop":[
                    "photographer"
                ],
                "temperature": null,
                "top_k": null,
                "top_n_tokens":null,
                "top_p":null,
                "truncate": null,
                "typical_p":null,
                "watermark": true
            },
            "stream":false}' http://127.0.0.1:1025/generate
