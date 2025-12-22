*面向用户的完整使用指南，包括模型加载、推理配置、参数调优、批量处理等功能的详细说明肯代码实例。*

# PD混合部署

## 单机混部

### 前提条件
- 服务器或容器环境上已经安装好NPU驱动和固件、CANN包、PyTorch、ATB Models和MindIE。

- 若开启HTTPS双向认证，需要提前准备好服务证书、服务器私钥和验签证书等。
- 若使用容器化部署启动，要求共享内存设置不小于1GB。
- Server对于Python的环境要求为Python3.10.x或者Python3.11.x。此处以Python3.10.13为例，如果环境中的Python3.10.13不是默认版本，需要参考如下方法添加环境变量（Python路径根据实际路径进行修改）。

```
export LD_LIBRARY_PATH=/usr/local/python3.10.13/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.10.13/bin:$PATH
```

### 操作步骤

1. 以安装用户进入MindIE安装目录。

   ```
   cd {MindIE安装目录}/latest
   ```

2. 确认目录文件权限是否如下所示，若存在不匹配项，则参考以下命令修改权限。

   ```
   chmod 750 mindie-service
   chmod -R 550 mindie-service/bin
   chmod -R 500 mindie-service/bin/mindie_llm_backend_connector
   chmod 550 mindie-service/lib
   chmod 440 mindie-service/lib/*
   chmod 550 mindie-service/lib/grpc
   chmod 440 mindie-service/lib/grpc/*
   chmod -R 550 mindie-service/include
   chmod -R 550 mindie-service/scripts
   chmod 750 mindie-service/logs
   chmod 750 mindie-service/conf
   chmod 640 mindie-service/conf/config.json
   chmod 700 mindie-service/security
   chmod -R 700 mindie-service/security/*
   ```

   <div class="note" style="background:#f0f9ff;border-left:4px solid #2196f3;padding:12px;margin:12px 0;border-radius:4px;">
   <strong>💡 说明：</strong> 若文件权限不符合要求将会导致Server启动失败。
   </div>

3. 根据用户需要设置配置参数。

   配置前注意事项如下所示：

   | 参数名称              | 说明                                             | 注意事项                                                     |
   | --------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
   | httpsEnabled          | 开启HTTPS通信（即“httpsEnabled”=false时）        | 不开启，会存在较高的网络安全风险                             |
   | maxLinkNum            | 默认值为1000，推荐设置为300                      | 1000并发能力受模型性能影响受限支持，一般较小模型、较低序列长度下才可以使用1000并发 |
   | MIES_CONFIG_JSON_PATH | 用户可通过设置该环境变量提供此Server的配置文件   | 需要用户自行保障此配置文件的安全性                           |
   | modelWeightPath       | 模型权重路径，此路径下的所有文件由用户自行提供   | 需要用户自行保障此处所有文件的安全性；且该路径下的config.json文件需保证其用户组和用户名与当前用户一致，并且为非软链接，文件权限不高于750，若不符合要求将会导致Server启动失败 |
   | tlsCaFile             | 业务面RESTful接口使用的服务证书文件              | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | tlsCert               | 业务面RESTful接口使用的服务证书文件              | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | tlsPk                 | 业务面RESTful接口使用的服务证书私钥文件          | 建议用户使用加密后的私钥文件，文件由用户提供，需要用户自行保障此文件的安全性 |
   | tlsCrlFiles           | 业务面RESTful接口使用的吊销列表文件列表          | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | managementTlsCaFile   | 管理面RESTful接口使用的CA证书文件列表            | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | managementTlsCert     | 管理面RESTful接口使用的服务证书文件              | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | managementTlsPk       | 管理面RESTful接口使用的服务证书私钥文件          | 建议用户使用加密后的私钥文件，文件由用户提供，需要用户自行保障此文件的安全性 |
   | managementTlsCrlFiles | 管理面RESTful接口使用的吊销列表文件列表          | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | interCommTlsCaFiles   | PD分离场景下，PD节点间通信使用的CA证书文件列表   | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | interCommTlsCert      | PD分离场景下，PD节点间通信使用的服务证书文件     | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | interCommPk           | PD分离场景下，PD节点间通信使用的服务证书私钥文件 | 建议用户使用加密后的私钥文件，文件由用户提供，需要用户自行保障此文件的安全性 |
   | interCommTlsCrlFiles  | PD分离场景下，PD节点间通信使用的吊销列表文件列表 | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | interNodeTlsCaFiles   | 多机场景下，主从节点间通信使用的CA证书文件列表   | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |
   | interNodeTlsCert      | 多机场景下，主从节点间通信使用的服务证书文件     | 文件由用户提供，需要用户自行保障此文件的安全性               |
   | interNodeTlsPk        | 多机场景下，主从节点间通信使用的服务证书私钥文件 | 建议用户使用加密后的私钥文件，文件由用户提供，需要用户自行保障此文件的安全性 |
   | interNodeTlsCrlFiles  | 多机场景下，主从节点间通信使用的吊销列表文件列表 | 文件由用户提供，需要用户自行保障此部分所有文件的安全性       |

   a. 进入conf目录，打开“config.json”文件。

     ```
     cd mindie-service/conf
     vim config.json
     ```

   b. 按“i”进入编辑模式，根据用户需要修改配置参数，参数详情请参见《MindIE LLM开发指南》中的“核心概念与配置 >[配置参数说明（服务化）](https://gitcode.com/Ascend/MindIE-LLM/blob/br_develop_mindie/docs/zh/user_guide/user_manual/service_parameter_configuration.md)”章节。

     配置文件config.json格式如下：

     ```
     {
         "Version": "2.2.RC1",
         "ServerConfig" :
         {
             "ipAddress" : "127.0.0.1",
             "managementIpAddress": "127.0.0.2",
             "port" : 1025,
             "managementPort" : 1026,
             "metricsPort" : 1027,
             "allowAllZeroIpListening" : false,
             "maxLinkNum" : 1000,
             "httpsEnabled" : true,
             "fullTextEnabled" : false,
             "tlsCaPath" : "security/ca/",
             "tlsCaFile" : ["ca.pem"],
             "tlsCert" : "security/certs/server.pem",
             "tlsPk" : "security/keys/server.key.pem",
             "tlsPkPwd" : "security/pass/key_pwd.txt",
             "tlsCrlPath" : "security/certs/",
             "tlsCrlFiles" : ["server_crl.pem"],
             "managementTlsCaFile" : ["management_ca.pem"],
             "managementTlsCert" : "security/certs/management/server.pem",
             "managementTlsPk" : "security/keys/management/server.key.pem",
             "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
             "managementTlsCrlPath" : "security/management/certs/",
             "managementTlsCrlFiles" : ["server_crl.pem"],
             "kmcKsfMaster" : "tools/pmt/master/ksfa",
             "kmcKsfStandby" : "tools/pmt/standby/ksfb",
             "inferMode" : "standard",
             "interCommTLSEnabled" : true,
             "interCommPort" : 1121,
             "interCommTlsCaPath" : "security/grpc/ca/",
             "interCommTlsCaFiles" : ["ca.pem"],
             "interCommTlsCert" : "security/grpc/certs/server.pem",
             "interCommPk" : "security/grpc/keys/server.key.pem",
             "interCommPkPwd" : "security/grpc/pass/key_pwd.txt",
             "interCommTlsCrlPath" : "security/grpc/certs/",
             "interCommTlsCrlFiles" : ["server_crl.pem"],
             "openAiSupport" : "vllm",
             "tokenTimeout" : 600,
             "e2eTimeout" : 600,
             "distDPServerEnabled": false
         },
     
         "BackendConfig": {
             "backendName" : "mindieservice_llm_engine",
             "modelInstanceNumber" : 1,
             "npuDeviceIds" : [[0,1,2,3]],
             "tokenizerProcessNumber" : 8,
             "multiNodesInferEnabled": false,
             "multiNodesInferPort": 1120,
             "interNodeTLSEnabled": true,
             "interNodeTlsCaPath": "security/grpc/ca/",
             "interNodeTlsCaFiles": ["ca.pem"],
             "interNodeTlsCert": "security/grpc/certs/server.pem",
             "interNodeTlsPk": "security/grpc/keys/server.key.pem",
             "interNodeTlsPkPwd": "security/grpc/pass/mindie_server_key_pwd.txt",
             "interNodeTlsCrlPath" : "security/grpc/certs/",
             "interNodeTlsCrlfiles" : ["server_crl.pem"],
             "interNodeKmcKsfMaster": "tools/pmt/master/ksfa",
             "interNodeKmcKsfStandby": "tools/pmt/standby/ksfb",
             "ModelDeployConfig":
             {
                 "maxSeqLen" : 2560,
                 "maxInputTokenLen" : 2048,
                 "truncation" : false,
                 "ModelConfig" : [
                     {
                         "modelInstanceType": "Standard",
                         "modelName" : "llama_65b",
                         "modelWeightPath" : "/data/atb_testdata/weights/llama1-65b-safetensors",
                         "worldSize" : 4,
                         "cpuMemSize" : 5,
                         "npuMemSize" : -1,
                         "backendType": "atb",
                         "trustRemoteCode": false,
                         "async_scheduler_wait_time": 120,
                         "kv_trans_timeout" : 10,
                         "kv_link_timeout" : 1080
                     }
                 ]
             },
      
             "ScheduleConfig":
             {
                 "templateType": "Standard",
                 "templateName" : "Standard_LLM",
                 "cacheBlockSize" : 128,
                 "maxPrefillBatchSize" : 50,
                 "maxPrefillTokens" : 8192,
                 "prefillTimeMsPerReq" : 150,
                 "prefillPolicyType" : 0,
                 "decodeTimeMsPerReq" : 50,
                 "decodePolicyType" : 0,
                 "maxBatchSize" : 200,
                 "maxIterTimes" : 512,
                 "maxPreemptCount" : 0,
                 "supportSelectBatch" : false,
                 "maxQueueDelayMicroseconds" : 5000,
                 "maxFirstTokenWaitTime": 2500
             }
         },
         "LogConfig": {
             "dynamicLogLevel" : "",
             "dynamicLogLevelValidHours" : 2,
             "dynamicLogLevelValidTime" : ""
         }
     }
     ```

   c. 按“Esc”键，输入**:wq!**，按“Enter”保存并退出编辑。

4. （可选）若开启了HTTPS认证（即“httpsEnabled” : true时，默认开启）。

   a. 则使用Service的证书导入脚本导入证书，各证书信息如表1所示。  

   <div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
   <strong>💡 说明</strong>
   <ul style="margin:8px 0;padding-left:20px;">
   <li>HTTPS使用三面隔离时，HTTPS的业务面和管理面不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。</li>
   <li>HTTPS和GRPC不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。</li>
   <li>导入证书时，对于用户导入CA证书的脚本权限要求为600，服务证书的脚本权限要求为600，私钥证书的脚本权限要求为400，吊销列表证书的脚本权限要求为600。</li>
   <li>Service的证书导入脚本请参见《MindIE Motor开发指南》中的"配套工具 > MindIE Service Tools > <a href="https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindieservice/servicedev/mindie_service0312.html">CertTools</a>"章节。</li>
   <li>如果导入证书超时，请参考<a href="https://www.hiascend.com/document/detail/zh/mindie/22RC1/envdeployment/instg/mindie_instg_0088.html">启动haveged服务</a>处理。</li></ul></div>
   
   

   表1 证书文件清单

   | 证书文件               | 默认目标路径                                             | 说明                                              |
   | ---------------------- | -------------------------------------------------------- | ------------------------------------------------- |
   | 根证书                 | *{MindIE安装目录}*/latest/mindie-service/security/ca/    | 支持多个CA证书。<br/><br/>开启HTTPS后必选。       |
   | 服务证书               | *{MindIE安装目录}*/latest/mindie-service/security/certs/ | 开启HTTPS后必选。                                 |
   | 服务证书私钥           | *{MindIE安装目录}*/latest/mindie-service/security/keys/  | 支持私钥文件加密场景。<br/><br/>开启HTTPS后必选。 |
   | 服务证书吊销列表       | {MindIE安装目录}/latest/mindie-service/security/certs/   | 开启HTTPS后可选。                                 |
   | 服务证书私钥的加密口令 | *{MindIE安装目录}*/latest/mindie-service/security/pass/  | 可选。                                            |

   b. 在*{MindIE安装目录}*/latest下执行以下命令修改证书文件的用户权限。

        ```
        chmod 400 mindie-service/security/ca/*
        chmod 400 mindie-service/security/certs/*
        chmod 400 mindie-service/security/keys/*
        chmod 400 mindie-service/security/pass/*
        ```

5. 使用以下命令配置环境变量。

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh                                 # CANN
   source /usr/local/Ascend/nnal/atb/set_env.sh                                       # ATB
   source /usr/local/Ascend/atb-models/set_env.sh                                # ATB Models
   ```

6. 将模型权重文件（由用户自行准备）拷贝到3.b中模型配置参数“modelWeightPath”指定的目录下。

   ```
   cp -r {模型权重文件所在路径} /data/atb_testdata/weights/llama1-65b-safetensors
   ```

7. 加载环境变量。

   ```
   source mindie-service/set_env.sh
   ```

8. 启动服务。启动命令需在/*{MindIE安装目录}*/latest/mindie-service目录中执行。

   <div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
   <strong>💡 说明</strong>
   <ul style="margin:8px 0;padding-left:20px;">
   <li>拉起服务前，建议用户使用MindStudio的预检工具进行配置文件字段校验，辅助校验配置的合法性，详情请参见[链接](https://gitcode.com/Ascend/msit/tree/master/msprechecker)
   </li>
   </ul></div>

   - （推荐）使用后台进程方式启动服务。

     ```
     nohup ./bin/mindieservice_daemon > output.log 2>&1 &
     ```

     在标准输出流捕获到的文件中，打印如下信息说明启动成功。

     ```
     Daemon start success!
     ```

   - 直接启动服务。

     ```
     ./bin/mindieservice_daemon
     ```

     回显如下则说明启动成功。

     ```
     Daemon start success!
     ```

<div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
<strong>📌 须知</strong>
<ul style="margin:8px 0;padding-left:20px;">
<li>Ascend-cann-toolkit工具会在执行服务启动的目录下生成kernel_meta_temp_xxxx目录，该目录为算子的cce文件保存目录。因此需要在当前用户拥有写权限目录下（例如Ascend-mindie-server_{version}_linux-{arch}_{abi}目录，或者用户在Ascend-mindie-server_{version}_linux-{arch}目录下自行创建临时目录）启动推理服务。</li>
<li>如需切换用户，请在切换用户后执行rm -f /dev/shm/*命令，删除由之前用户运行创建的共享文件。避免切换用户后，该用户没有之前用户创建的共享文件的读写权限，造成推理失败。</li>
<li>bin目录按照安全要求，目录权限为550，没有写权限，不能直接在bin启动mindieservice_daemon。</li>
<li>标准输出流捕获到的文件output.log支持用户自定义文件和路径。</li>
<li>服务启动报缺失lib*.so依赖的错误时，处理方法请参见启动MindIE Motor服务时，出现找不到libboost_thread.so.1.82.0报错章节。</li>
<li>不建议在同一容器中反复拉起服务，重复拉起前请清理容器“/dev/shm/”目录下的*llm_backend_*和llm_tokenizer_shared_memory_*文件，参考命令如下：</li>
<code>find /dev/shm -name '*llm_backend_*' -type f -delete</code>
<br><code>find /dev/shm -name 'llm_tokenizer_shared_memory_*' -type f -delete</code><br>
</ul></div>


## 多机混部

单个模型权重过大，单台推理机显存有限，无法容纳整个模型权重参数时，需要采用多个节点进行多机推理。

### 前提条件

- Server对于Python的环境要求为Python3.10.*x*或者Python3.11*.**x*。此处以Python3.10.13为例，如果环境中的Python3.10.13不是默认版本，需要参考如下方法添加环境变量（Python路径根据实际路径进行修改）。

  ```
  export LD_LIBRARY_PATH=/usr/local/python3.10.13/lib:$LD_LIBRARY_PATH
  export PATH=/usr/local/python3.10.13/bin:$PATH
  ```

- 服务器或容器环境上已经安装好NPU驱动和固件、CANN包、PyTorch、ATB Models和MindIE。

- 若使用容器化部署启动，要求共享内存设置不小于1GB。

- 若开启HTTPS双向认证或多机通信认证，需要提前准备好服务证书、服务器私钥、验签证书等，详情请参见《MindIE Motor开发指南》中的“集群服务部署 > 单机（非分布式）服务部署 > 安装部署 > 使用Deployer部署服务示例 > 部署Deployer服务端 > [准备TLS证书](https://gitcode.com/Ascend/MindIE-Motor/blob/br_develop_mindie/docs/zh/User_Guide/SERVICE_DEPLOYMENT/single_machine_service_deployment.md)”章节。

### 使用限制

- 仅支持Atlas 800I A2 推理服务器环境，最大支持4机32卡的多机推理，多机推理支持的模型请参见[模型列表](https://gitcode.com/Ascend/MindIE-LLM/blob/br_develop_mindie/docs/zh/user_guide/model_support_list.md)；不支持Atlas 300I Duo 推理卡环境。
- “maxLinkNum”默认值为1000，推荐设置为300。1000并发能力受模型性能影响受限支持，一般较小模型、较低序列长度下才可以使用1000并发。
- 不同节点的权重的默认采样参数需要配置一致，否则在没有配置采样参数的情况下，推理服务可能卡死。

### 相关环境变量

| 参数名称              | 参数说明                                                     |
| --------------------- | ------------------------------------------------------------ |
| MIES_CONTAINER_IP     | 容器部署时，请设置成容器的IP地址（如果容器与裸机共用IP地址，应当配置为裸机IP地址），会用于多机间gRPC（Google Remote Procedure Call）通信和EndPoint业务面接收请求。裸机部署时，不配置。 |
| HOST_IP               | 裸机部署时（不建议使用裸机部署），请设置成机器的物理机或虚拟机的IP地址。容器部署时不配置。 |
| RANK_TABLE_FILE       | ranktable.json文件的绝对路径。  多机推理必须配置。 单机推理建议取消该环境变量（取消命令：**unset RANK_TABLE_FILE**）。如果设置该环境变量，文件内容必须正确有效（节点IP地址和device_ip必须正确），否则会导致模型初始化失败。 |
| MIES_CONFIG_JSON_PATH | config.json文件的路径。如果该环境变量存在，则读取该值。如果不存在，则读取$*{MIES_INSTALL_PATH}*/conf/config.json文件。 |
| HCCL_DETERMINISTIC    | HCCL通信的确定性计算。多机推理时，建议配置为true。           |

<div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
<strong>💡 说明</strong>
<ul style="margin:8px 0;padding-left:20px;">
Server启动时，会根据“multiNodesInferEnabled”参数判断是单机推理还是多机推理：
<li>“multiNodesInferEnabled” : false代表单机推理，Server在启动过程中不会读取“RANK_TABLE_FILE”环境变量，但是底层模型加速库初始化时，会尝试读取该环境变量。所以在单机推理场景中，如果设置了该环境变量，请保证文件内容值的正确性（即：server_count=1、节点IP、device_ip和rank_id等必须正确）。</li>
<li>“multiNodesInferEnabled” : true代表多机推理，Server在启动过程中，会读取“RANK_TABLE_FILE”环境变量，并判断ranktable文件内容是否有效。
</li>
<li>rank_id=0的节点为Master节点，其余为Slave节点。</li>
<li>Master服务实例，可以接收用户推理请求；Slave实例无法接收用户推理请求。</li>
</ul></div>

<div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
<strong>📌 须知</strong>
<ul style="margin:8px 0;padding-left:20px;">
当开启多机推理时，config.json中的“npuDeviceIds”和“worldSize”将失效，具体使用卡号及总体Rank数，将根据ranktable文件确定。
</ul></div>

### ranktable文件样例

ranktable.json文件权限需要设置为640，详细内容请根据以下样例进行配置。（该文件需要用户自行编写）

```
{
    "version": "1.0",
    "server_count": "2",
    "server_list": [
        {
            "server_id": "Master节点IP地址",
            "container_ip": "Master节点容器IP地址",
            "device": [
                { "device_id": "0", "device_ip": "10.20.0.2", "rank_id": "0" }, 
                { "device_id": "1", "device_ip": "10.20.0.3", "rank_id": "1" },
                { "device_id": "2", "device_ip": "10.20.0.4", "rank_id": "2" },
                { "device_id": "3", "device_ip": "10.20.0.5", "rank_id": "3" },
                { "device_id": "4", "device_ip": "10.20.0.6", "rank_id": "4" },
                { "device_id": "5", "device_ip": "10.20.0.7", "rank_id": "5" },
                { "device_id": "6", "device_ip": "10.20.0.8", "rank_id": "6" },
                { "device_id": "7", "device_ip": "10.20.0.9", "rank_id": "7" }
            ]
        },
        {
            "server_id": "Slave节点IP地址",
            "container_ip": "Slave节点容器IP地址",
            "device": [
                { "device_id": "0", "device_ip": "10.20.0.10", "rank_id": "8" },
                { "device_id": "1", "device_ip": "10.20.0.11", "rank_id": "9" },
                { "device_id": "2", "device_ip": "10.20.0.12", "rank_id": "10" },
                { "device_id": "3", "device_ip": "10.20.0.13", "rank_id": "11" },
                { "device_id": "4", "device_ip": "10.20.0.14", "rank_id": "12" },
                { "device_id": "5", "device_ip": "10.20.0.15", "rank_id": "13" },
                { "device_id": "6", "device_ip": "10.20.0.16", "rank_id": "14" },
                { "device_id": "7", "device_ip": "10.20.0.17", "rank_id": "15" }
            ]
        }
    ],
    "status": "completed"
}
```

参数说明：

- Master/Slave节点IP地址：请根据实际情况进行修改。
- Master/Slave节点容器IP地址：一般与Master/Slave节点IP地址一致，如果启动容器时使用了**--net=host**，则需要与宿主机IP地址一致，请根据实际情况进行修改。
- device_id：表示在实际节点上的第几个NPU设备。
- device_ip：表示NPU设备的IP地址，可通过hccn_tool进行配置。
- rank_id：表示推理进程Rank编号。

<div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
<strong>💡 说明</strong>
<ul style="margin:8px 0;padding-left:20px;">
ranktable.json文件通过环境变量“RANK_TABLE_FILE”配置，若用户自行提供此文件，需要用户自行保障此配置文件的安全性，且Master节点和Slave节点上都需要创建该文件。
</ul></div>

### 操作步骤

<div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
<strong>💡 说明</strong>
<ul style="margin:8px 0;padding-left:20px;">
Master节点和Slave节点上均需执行以下操作。
</ul></div>

1. 创建并启动Docker容器，此处以8卡昇腾环境为例。

   以下启动命令仅供参考，可根据需求自行修改。

   ```
   docker run -it -d --net=host --shm-size=1g \
       --name container_name \
       --device=/dev/davinci_manager \
       --device=/dev/hisi_hdc \
       --device=/dev/devmm_svm \
       --device=/dev/davinci0 \
       --device=/dev/davinci1 \
       --device=/dev/davinci2 \
       --device=/dev/davinci3 \
       --device=/dev/davinci4 \
       --device=/dev/davinci5 \
       --device=/dev/davinci6 \
       --device=/dev/davinci7 \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
       -v /usr/local/sbin:/usr/local/sbin:ro \
       -v /path-to-weights:/path-to-weights:ro \
       mindie:2.2.RC1-800I-A2-aarch64
   ```

2. 以安装用户进入MindIE安装目录。

   ```
   cd {MindIE安装目录}/latest
   ```

3. 确认目录文件权限是否如下所示，若存在不匹配项，则参考以下命令修改权限。

   ```
   chmod 750 mindie-service
   chmod -R 550 mindie-service/bin
   chmod -R 500 mindie-service/bin/mindie_llm_backend_connector
   chmod 550 mindie-service/lib
   chmod 440 mindie-service/lib/*
   chmod 550 mindie-service/lib/grpc
   chmod 440 mindie-service/lib/grpc/*
   chmod -R 550 mindie-service/include
   chmod -R 550 mindie-service/scripts
   chmod 750 mindie-service/logs
   chmod 750 mindie-service/conf
   chmod 640 mindie-service/conf/config.json
   chmod 700 mindie-service/security
   chmod -R 700 mindie-service/security/*
   ```

   <div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
   <strong>💡 说明</strong>
   <ul style="margin:8px 0;padding-left:20px;">
   若文件权限不符合要求将会导致Server启动失败。
   </ul></div>

4. 在容器中，根据用户需要设置配置参数。

   配置前请参见3中的注意事项。

   a. 进入conf目录，打开“config.json”文件。

      ```
      cd ../conf
      vim config.json
      ```

   b. 按“i”进入编辑模式，设置“multiNodesInferEnabled”=true开启多机推理，并根据用户需要修改表1的参数，参数详情请参见《MindIE LLM开发指南》中的“核心概念与配置 >[配置参数说明（服务化）](https://gitcode.com/Ascend/MindIE-LLM/blob/br_develop_mindie/docs/zh/user_guide/user_manual/service_parameter_configuration.md)”章节。

      表1 多机推理相关配置

   | 配置项                 | 配置说明                                                     |
   | ---------------------- | ------------------------------------------------------------ |
   | multiNodesInferPort    | 跨机通信的端口号。                                           |
   | interNodeTLSEnabled    | 跨机通信是否开启证书安全认证。true：开启证书安全认证。false：关闭证书安全认证。若关闭证书安全认证，可忽略以下参数。 |
   | interNodeTlsCaPath     | 根证书名称路径。“interNodeTLSEnabled”=true生效。             |
   | interNodeTlsCaFiles    | 根证书名称列表。“interNodeTLSEnabled”=true生效。             |
   | interNodeTlsCert       | 服务证书文件路径。“interNodeTLSEnabled” : true生效。         |
   | interNodeTlsPk         | 服务证书私钥文件路径。“interNodeTLSEnabled” : true生效。     |
   | interNodeTlsPkPwd      | 服务证书私钥文件路径。“interNodeTLSEnabled” : true生效。     |
   | interNodeTlsPkPwd      | 服务证书私钥加密密钥文件路径。“interNodeTLSEnabled” : true生效。 |
   | interNodeTlsCrlPath    | 服务证书吊销列表文件夹路径。“interNodeTLSEnabled”=true生效。 |
   | interNodeTlsCrlFiles   | 服务证书吊销列表名称列表。“interNodeTLSEnabled”=true生效。   |
   | interNodeKmcKsfMaster  | KMC密钥库备份文件路径。“interNodeTLSEnabled” : true生效。    |
   | interNodeKmcKsfStandby | KMC密钥库备份文件路径。“interNodeTLSEnabled” : true生效。    |

      <div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
      <strong>📌 须知</strong>
      <ul style="margin:8px 0;padding-left:20px;">
          <li>如果不开启HTTPS通信（即“httpsEnabled” : false时），会存在较高的网络安全风险。</li>
          <li>“modelWeightPath”参数配置路径下的config.json文件，需保证其用户组和用户名与当前用户一致，并且为非软链接，文件权限不高于640，若不符合要求将会导致启动失败。
      </li>
          <li>在数据中心内部，如果不需要开启跨机通信安全认证，请配置“interNodeTLSEnabled” : false，若关闭跨机通信安全认证（即“interNodeTLSEnabled” : false），会存在较高的网络安全风险。
      </li>
      </ul></div>

   c. 按“Esc”键，输入**:wq!**，按“Enter”保存并退出编辑。

5. （可选）若开启了GRPC双向认证（即“interNodeTLSEnabled”=true时）。


  	 a. 使用证书管理工具导入证书，证书管理工具的使用请参见《MindIE Motor开发指南》中的“配套工具 > MindIE Service Tools > [CertTools](https://gitcode.com/Ascend/MindIE-Motor/blob/br_develop_mindie/docs/zh/User_Guide/SERVICE_DEPLOYMENT/single_machine_service_deployment.md)”章节。
  	
  	  <div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
  	  <strong>💡 说明</strong>
  	  <ul style="margin:8px 0;padding-left:20px;">
  	  <li>HTTPS使用三面隔离时，HTTPS的业务面和管理面不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。</li>
  	  <li>HTTPS和GRPC不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。</li>
  	  <li>导入证书时，对于用户导入的CA文件证书工具要求的权限为600，服务证书文件证书工具要求的权限为600，私钥文件证书工具要求的权限要求为400，吊销列表证书工具要求的权限为600。</li>
  	  <li>各证书文件信息如表2所示。</li>
  	      <li>如果导入证书超时，请参考启动haveged服务处理。</li>
  	  </ul></div>
  	
  	  表2 证书文件信息
  	
  	  | 证书文件               | 默认目标路径                        | 说明                                                         |
  	  | ---------------------- | ----------------------------------- | ------------------------------------------------------------ |
  	  | 根证书                 | mindie-service/security/grpc/ca/    | 开启“interNodeTLSEnabled” : true后必选。                     |
  	  | 服务证书               | mindie-service/grpc/certs/          | 开启“interNodeTLSEnabled” : true后必选。                     |
  	  | 服务证书私钥           | mindie-service/security/grpc/keys/  | 支持私钥文件加密场景。开启“interNodeTLSEnabled” : true后必选。 |
  	  | 服务证书吊销列表       | mindie-service/security/grpc/certs/ | 必选。                                                       |
  	  | 服务证书私钥的加密口令 | mindie-service/security/pass/       | 必选。                                                       |
  	
  	   b. 在*{MindIE安装目录}*/latest下执行以下命令修改证书文件的用户权限。
  	
  	  ```
  	  chmod 400 mindie-service/security/grpc/ca/*
  	  chmod 400 mindie-service/security/grpc/certs/*
  	  chmod 400 mindie-service/security/grpc/keys/*
  	  chmod 400 mindie-service/security/grpc/pass/*
  	  ```

6. （可选）若开启了HTTPS认证（即“httpsEnabled” : true时，默认开启）。

   1. 则使用Service的证书导入脚本导入证书，各证书信息如表3所示。

      <div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
      <strong>💡 说明</strong>
      <ul style="margin:8px 0;padding-left:20px;">
          <li>HTTPS使用三面隔离时，HTTPS的业务面和管理面不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。
      </li>
          <li>HTTPS和GRPC不建议使用同一套安全证书，使用同一套安全证书会存在较高的网络安全风险。
      </li>
          <li>导入证书时，对于用户导入CA证书的脚本权限要求为600，服务证书的脚本权限要求为600，私钥证书的脚本权限要求为400，吊销列表证书的脚本权限要求为600。
      </li>
          <li>Service的证书导入脚本请参见《MindIE Motor开发指南》中的“配套工具 > MindIE Service Tools > CertTools”章节。
      </li>
          <li>如果导入证书超时，请参考启动haveged服务处理。
      </li>
      </ul></div>

      表3 证书文件清单

      | 证书文件               | 默认目标路径                                             | 说明                                    |
      | ---------------------- | -------------------------------------------------------- | --------------------------------------- |
      | 根证书                 | *{MindIE安装目录}*/latest/mindie-service/security/ca/    | 支持多个CA证书。开启HTTPS后必选。       |
      | 服务证书               | *{MindIE安装目录}*/latest/mindie-service/security/certs/ | 开启HTTPS后必选。                       |
      | 服务证书私钥           | *{MindIE安装目录}*/latest/mindie-service/security/keys/  | 支持私钥文件加密场景。开启HTTPS后必选。 |
      | 服务证书吊销列表       | *{MindIE安装目录}*/latest/mindie-service/security/certs/ | 开启HTTPS后可选。                       |
      | 服务证书私钥的加密口令 | *{MindIE安装目录}*/latest/mindie-service/security/pass/  | 可选。                                  |

      

   2. 在*{MindIE安装目录}*/latest下执行以下命令修改证书文件的用户权限。

      ```
      chmod 400 mindie-service/security/ca/*
      chmod 400 mindie-service/security/certs/*
      chmod 400 mindie-service/security/keys/*
      chmod 400 mindie-service/security/pass/*
      ```

      

7. 使用以下命令配置环境变量。

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh                                 # CANN
   source /usr/local/Ascend/nnal/atb/set_env.sh                                       # ATB
   source /usr/local/Ascend/atb-models/set_env.sh                                # ATB Models
   ```

8. 将模型权重文件（由用户自行准备）拷贝到config.json中模型配置参数“modelWeightPath”指定的目录下。 

   ```
   cp -r {模型权重文件所在路径} /data/atb_testdata/weights/llama1-65b-safetensors
   ```

9. 加载环境变量。

   ```
   source mindie-service/set_env.sh
   ```

10. 配置环境变量“RANK_TABLE_FILE”和“MIES_CONTAINER_IP”（以[ranktable文件样例](https://gitcode.com/Ascend/MindIE-Motor/blob/br_develop_mindie/docs/zh/User_Guide/SERVICE_DEPLOYMENT/pd_separation_service_deployment.md)中的ranktable为例，具体在链接的表4）。

    - Master节点容器中

      ```
      export MIES_CONTAINER_IP=Master节点IP地址
      export RANK_TABLE_FILE=${path}/ranktable.json
      export HCCL_DETERMINISTIC=true
      ```

    - Slave节点容器中

      ```
      export MIES_CONTAINER_IP=Slave节点IP地址
      export RANK_TABLE_FILE=${path}/ranktable.json
      export HCCL_DETERMINISTIC=true
      ```

11. 启动服务，启动命令需在/*{MindIE安装目录}*/latest/mindie-service目录中执行。此操作在Master节点容器和Slave节点容器中均需执行。

    - （推荐）使用后台进程方式启动。

      ```
      nohup ./bin/mindieservice_daemon > output.log 2>&1 &
      ```

      在标准输出流捕获到的文件中，打印如下信息说明启动成功。

      ```
      Daemon start success!
      ```

    - 直接启动。

      ```
      ./bin/mindieservice_daemon
      ```

      回显如下则说明启动成功。

      ```
      Daemon start success!
      ```

<div style="background:#f0f9ff;border-left:4px solid #2196f3;padding:14px;margin:16px 0;border-radius:6px;">
<strong>📌 须知</strong>
<ul style="margin:8px 0;padding-left:20px;">
<li>Ascend-cann-toolkit工具会在执行服务启动的目录下生成kernel_meta_temp_xxxx目录，该目录为算子的cce文件保存目录。因此需要在当前用户拥有写权限目录下（例如Ascend-mindie-server_{version}_linux-{arch}_{abi}目录，或者用户在Ascend-mindie-server_{version}_linux-{arch}目录下自行创建临时目录）启动推理服务。</li>
<li>如需切换用户，请在切换用户后执行rm -f /dev/shm/*命令，删除由之前用户运行创建的共享文件。避免切换用户后，该用户没有之前用户创建的共享文件的读写权限，造成推理失败。</li>
<li>bin目录按照安全要求，目录权限为550，没有写权限，不能直接在bin启动mindieservice_daemon。</li>
<li>标准输出流捕获到的文件output.log支持用户自定义文件和路径。</li>
<li>服务启动报缺失lib*.so依赖的错误时，处理方法请参见启动MindIE Motor服务时，出现找不到libboost_thread.so.1.82.0报错章节。
<li>不建议在同一容器中反复拉起服务，重复拉起前请清理容器“/dev/shm/”目录下的*llm_backend_*和llm_tokenizer_shared_memory_*文件，参考命令如下：</li>
<code>find /dev/shm -name '*llm_backend_*' -type f -delete</code>
<br><code>find /dev/shm -name 'llm_tokenizer_shared_memory_*' -type f -delete</code><br>
</ul></div>