# MemoryDecoding运行方法

## 0. Memory Decoding简介
    本算法为一种并行解码类算法，中文名称为基于动态token知识库检索的并行解码算法。
    其算法原理为，通过trie-tree（前缀树）的方式，缓存模型历史的输入输出，在解码阶段，从trie-tree中获取草稿token，并用作并行解码，以此提升模型的推理速度。  


## 1. 适用场景

    * 用户问答具有一定结构性的场景，如代码生成，代码续写等
    * 可从prompt中获取参考的问答场景，如多轮对话、RAG、长文本等
    * 用户问答间具有较高的相似性的场景


## 2. 使用run_generator.py运行
### 2.1 执行脚本代码示例
    export PYTHONPATH=/LLM代码路径/examples/atb_models/:$PYTHONPATH
    export PYTHONPATH=/LLM代码路径/:$PYTHONPATH

    # 设置执行时运行在哪些NPU上
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

    export INF_NAN_MODE_ENABLE=0
    export ATB_CONVERT_NCHW_TO_ND=1
    export LCCL_DETERMINISTIC=1

    # LOG开关按需设置（可按需调整）
    export MINDIE_LOG_LEVEL=INFO
    export MINDIE_LOG_TO_FILE=1
    export MINDIE_LOG_TO_STDOUT=1

    # 1表示执行的NPU数目，16表示最大batchsize，可按需调整
    torchrun \
    --nproc_per_node 1 \
    --master_port 25632 \
    -m mindie_llm.examples.run_generator \
    --model_path /模型路径/ \
    --max_batch_size 16 \
    --max_output_length 20 \
    --npu_mem 10 \
    --plugin_params "{\"plugin_type\": \"memory_decoding\", \"decoding_length\": 16}" \
    --speculation_gamma 16

### 2.2 参数配置
参数在run_generator.py中修改，主要涉及的参数如下（推荐配置，可按需修改）：

model_config中：

    plugin_type: memory_decoding
    decoding_length: 16

    # 若出现OOM可适当调小npu_men
    npu_mem:10

parse_arguments的配置示例

    '--input_texts'：
    default=["What's deep learning?", "How do you feel?", "What's deep learning?", "How do you feel?"]

    '--max_output_length'：256

### 2.3 运行
在Mindie_llm代码根目录下编译后执行3.1中的脚本即可。

## 3. 使用服务化执行

### 3.1 使用llm_engin_test执行性能测试
#### 3.1.1 config文件配置
"ModelDeployConfig"下添加：
    
    "speculationGamma": 16
"ModelDeployConfig"中的"ModelConfig"下添加：

    "plugin_params": "{\"plugin_type\":\"memory_decoding\",\"decoding_length\": 16,\"dynamic_algo\": true}"
    

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
            "inputs": "What's deep learning?"
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

## 参考
    [1] @misc{zhao2024lookahead,
      title={Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy}, 
      author={Yao Zhao and Zhitian Xie and Chen Liang and Chenyi Zhuang and Jinjie Gu},
      year={2024},
      eprint={2312.12728},
      archivePrefix={arXiv},
      primaryClass={id='cs.IR' full_name='Information Retrieval' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers indexing, dictionaries, retrieval, content and analysis. Roughly includes material in ACM Subject Classes H.3.0, H.3.1, H.3.2, H.3.3, and H.3.4.'}}
    [2] @misc{he2024rest,
      title={REST: Retrieval-Based Speculative Decoding}, 
      author={Zhenyu He and Zexuan Zhong and Tianle Cai and Jason D. Lee and Di He},
      year={2024},
      eprint={2311.08252},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}}
    [3] @misc{yang2023inference,
      title={Inference with Reference: Lossless Acceleration of Large Language Models}, 
      author={Nan Yang and Tao Ge and Liang Wang and Binxing Jiao and Daxin Jiang and Linjun Yang and Rangan Majumder and Furu Wei},
      year={2023},
      eprint={2304.04487},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}}


