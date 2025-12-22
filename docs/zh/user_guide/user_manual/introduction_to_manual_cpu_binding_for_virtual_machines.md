# 虚拟机手动绑核配置介绍

在虚拟机场景下使用ATB Models进行推理时，为提高性能，需要用户进行手动绑核操作：

1. 在虚拟机中，使用**npu-smi info**命令查询npu对应的pci id，如图1所示。

    **图 1**  npu 0对应的pci id为0000:08:00.0<a name="fig1703125515381"></a>  
    
    ![](../../figures/virtual_machines_figure1.png)

2.  查询物理机和虚拟机的pci id对应关系。在物理机上使用virsh edit \{虚拟机名称\}查询，虚拟机名称可通过**virsh list --all**查询到。

    查询页面如图2所示，其中可以找到物理机和虚拟机的pci之间的关系。

    **图 2**  虚拟机上的pci id 0000:08:00.0对应了物理机的0000:C1:00.0<a name="fig8270192611200"></a>  
    ![](../../figures/virtual_pciic_physical_c1.png "虚拟机上的pci-id-0000-08-00-0对应了物理机的0000-C1-00-0")

3.  在物理机上使用命令**cat /sys/bus/pci/devices/\{pci id\}/numa\_node**查询NUMA（Non Uniform Memory Access Architecture） node，如图3所示。

    **图 3**  0000:C1:00.0对应的NUMA node为6<a name="fig19697612142117"></a>  
    ![](../../figures/numa_node6.png "0000-C1-00-0对应的NUMA-node为6")

4.  在物理机上使用命令**lscpu**查询NUMA node对应的cpu，如图4所示。

    **图 4**  NUMA node 6对应的cpu为192-223<a name="fig361218288212"></a>  
    ![](../../figures/numa_node6_cpu_192223.png "NUMA-node-6对应的cpu为192-223")

5.  <a id="step5"></a>查询物理机和虚拟机的NUMA node对应关系，在物理机上使用命令**virsh edit \{_虚拟机名称_\}**查询，虚拟机名称可通过命令**virsh list --all**查询。查询页面如图5所示：

    红框1表示了虚拟机cpu和物理机cpu的对应关系，如：虚拟机cpu 191对应了物理机247。

    红框2的<numa\>中cell id='0' cpus='0-23'表示numa\_node = 0时，虚拟机cpu号为0-23。

    物理机NUMA node 6的cpu为192-223，经查询，其对应虚拟机cpu为144-167，即物理机NUMA node 6对应虚拟机NUMA node 6，因此npu 0对应的虚拟机NUMA node为6。

    **图 5**  虚拟机配置界面<a name="fig1849375910214"></a>  
    ![](../../figures/virtual_interface.png "虚拟机配置界面")

6.  在虚拟机中执行**echo  _x_  \> /sys/bus/pci/devices/\{_pci id_\}/numa\_node**，其中x为[5](#step5)中查询到的npu对应的虚拟机NUMA node，_pci id_为虚拟机上的pci id。

    以npu 0为例，命令为**echo 6 \> /sys/bus/pci/devices/0000\\:08\\:00.0/numa\_node**。

    至此虚拟机npu 0完成绑核操作。

7.  通过查询MindIE LLM日志，验证绑核成功的npu。查询日志的方式请参见《MindIE日志参考》的“查看日志”章节。

    当日志如图6所示，即可显示绑定成功的npu。

    **图 6**  绑定成功的npu<a name="fig152351139145918"></a>  
    ![](../../figures/successful_npu.png "绑定成功的npu")
