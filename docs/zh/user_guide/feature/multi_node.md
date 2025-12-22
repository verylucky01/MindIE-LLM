# 多机

针对模型参数特别大的场景（如175B以上参数量模型）以及长序列场景，用户可选择使用多台服务器组合进行推理以达到较优效果，多机通信目前有两种硬件连接方式，PCIE建联以及RDMA建联。使用多机推理时，需要将包含设备IP，进程号信息，服务器IP信息的JSON文件地址传递给底层通信算子。

多机通信有多种并行策略可以结合，目前主流的并行策略包括Tensor Parallel（TP，张量并行），Pipeline Parallel（PP，流水并行），Sequence Parallel（SP，序列并行）以及MoE（Mixture of Experts）类模型特有的Expert Parallel（EP，专家并行）。通过不同的并行策略的结合，可以最大化整体吞吐量，获得最优性能表现。

目前多机通信只支持TP并行方式执行模型推理，部分开源模型已经完成多机多卡推理能力构建，其中包括LLaMA系列模型（包括基于LLaMA实现的模型），DeepSeek-MoE以及Mixtral-MoE。

> [!NOTE]说明 
> 单机通信只支持TP并行策略。

## 限制与约束

- 仅支持模型同时部署在多台Atlas 800I A2 推理服务器上，服务器台数为![](../../figures/formulate_multi_node.png)。
- Atlas 300I Duo 推理卡不支持多机模型部署。
- LLaMA3系列、DeepSeek-MoE系列、Qwen-MoE系列、Mixtral-MoE系列、Kim K2、GLM4.5和Ernie 4.5支持该特性。
- 仅支持和W8A8、W8A16、KV Cache int8量化、Function Call和MoE特性配合使用。

## 模型适配

模型需预先基于ATB加速库组图方式完成流水图构建，即模型可以通过MindIE LLM调用ATB加速库组图的方式实现单机推理，在此基础上，需要将一个json文件地址传递给通信算子用于设备的ip，进程关系解析，本文件需借助hccn\_tool手动生成，生成方式如下：

> [!NOTE]说明 
> hccn\_tool工具的安装及使用详见**HCCN Tool接口参考**。

- **获取卡号对应ip信息：**

    在裸机上执行以下命令：

    ```
    for i in {0..7}
    do
        hccn_tool -i $i -ip -g
    done
    ```

    获得输出如图所示：

    ```
    root@huawei89:~# 
    for i in {0..7}
    do
    hccn_tool -i $i -ip -g
    done
    ipaddr:10.10.100.10
    netmask:255.255.255.0
    ipaddr:10.10.100.11
    netmask:255.255.255.0
    ipaddr:10.10.100.12
    netmask:255.255.255.0
    ipaddr:10.10.100.13
    netmask:255.255.255.0
    ipaddr:10.10.100.14
    netmask:255.255.255.0
    ipaddr:10.10.100.15
    netmask:255.255.255.0
    ipaddr:10.10.100.16
    netmask:255.255.255.0
    ipaddr:10.10.100.17
    netmask:255.255.255.0
    root@huawei89:~#
    ```

- **检查两台机器连通性：**

    使用如下命令检查两台机器双卡之间连通性：

    ```
    hccn_tool -i 0 -ping -g address 1.1.1.1
    ```

    其中1.1.1.1为对端机器每一张npu卡的ip地址，如果连接成功应当有输出如下：

    > [!NOTE]说明 
    > 多机参数面互联：即服务器NPU卡对应的端口处在同一个VLAN，可以通过ROCE互通。

    ```
    root@huawei:~# hccn_tool -i 0 -ping -g address 10.10.100.10
    device 0 PING 10.10.100.10
    recv seq=0,time=0.067000ms
    recv seq=1,time-0.049000ms
    recv seq=2,time=0.047000ms
    3 packets transmitted, 3 received, 0.00% packet loss
    root@huawei: #
    ```

    如果存在丢包，则会输出如下内容，如存在此种情况，请优先对硬件连通性进行检查校验。

    ```
    recv time out seq=O
    recv seq=1,time=0.097000ms
    recv seq=2,time=0.066000ms
    3 packets transmitted, 2 received, 33.33% packet loss
    ```

- **生成RankTableFile**

    确认两机连通性成功后，可通过查询到的ip信息生成RankTableFile，步骤如下：

    新建一个ranktablefile.json文件，将样例的内容复制粘贴到json文件里，得到RankTableFile接下来需手动将样例中的加粗内容替换为查询到的机器实际ip等配置信息。

    ```
    {
        "server_count":"2", 
        "server_list":
        [
           {
                "device":[  
                        {
                        "device_id":"0",   
                        "device_ip":"10.10.100.10",    
                        "rank_id":"0"     
                        },
                        {
                        "device_id":"1",
                        "device_ip":"10.10.100.11", 
                        "rank_id":"1"
                        },
                        {
                        "device_id":"2",
                        "device_ip":"10.10.100.12", 
                        "rank_id":"2"
                        },
                        {
                        "device_id":"3",
                        "device_ip":"10.10.100.13", 
                        "rank_id":"3"
                        }],
                 "server_id":"90.90.152.89"   
            },
            {
              "device":[  
    		  {
                      "device_id":"0",   
                      "device_ip":"10.10.100.22",    
                      "rank_id":"4"     
                      },
                      {
                      "device_id":"1",
                      "device_ip":"10.10.100.23", 
                      "rank_id":"5"
                      },
                      {
                      "device_id":"2",
                      "device_ip":"10.10.100.24", 
                      "rank_id":"6"
                      },
                      {
                      "device_id":"3",
                      "device_ip":"10.10.100.25", 
                      "rank_id":"7"
                      }
                        ],
               "server_id":"90.90.152.91"   
          }
        ],
        "status":"completed",  
        "version":"1.0"
        }
    ```

## 执行推理

若需要实现多机推理，在基础单机推理的启动脚本基础上，需要做出以下适配修改，首先应声明以下额外的环境变量，以双机16卡为例：

```
export ATB_LLM_HCCL_ENABLE=1
export WORLD_SIZE=16
export RANK_TABLE_FILE="RankTableFile的绝对路径"
```

> [!NOTE]说明
> 多机推理时，每一台机器上的推理脚本都需要进行单独的适配修改，在执行推理时，每一台机器上的推理脚本都需要被同步执行。

将torchrun进行进程拉起的部分替换为如下使用python的方式进行拉起：

```
cd ${ATB_SPEED_HOME_PATH}
RANK_ID_START=0  #主节点该值为0，副节点此值应为（WORLD_SIZE / 2）
WORLD_SIZE_LOCAL=8
for((RANK_ID=$RANK_ID_START;RANK_ID<$((WORLD_SIZE_LOCAL+RANK_ID_START));RANK_ID++));
do
export RANK=$RANK_ID
export LOCAL_RANK=$RANK_ID  #主节点LOCAL_RANK与rank相同，副节点local_rank = rank - WORLD_SIZE_LOCAL
python3 ./examples/run_pa.py  --model_path /data/atb_testdata/weights/llama3-70b/ \
--max_output_length 35  --max_batch_size 1 --input_text "What's Deep Learning?" &
done
wait
```

