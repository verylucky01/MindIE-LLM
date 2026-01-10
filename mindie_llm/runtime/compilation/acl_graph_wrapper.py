# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import torch
from mindie_llm.runtime.model_runner.forward_context import get_forward_context

# Status indicating whether aclgraph can be captured.
_aclgraph_capturing_enable = False
# Reuse this memory pool across all acl graph runners.
_global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return _global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global _global_graph_memory_pool
    _global_graph_memory_pool = val


def validate_aclgraph_capturing_enabled():
    global _aclgraph_capturing_enable
    if not _aclgraph_capturing_enable:
        raise RuntimeError(
            "ACL graph capturing detected at an inappropriate "
            "time. This operation is currently disabled."
        )


def set_aclgraph_capturing_enabled(enabled: bool):
    global _aclgraph_capturing_enable
    _aclgraph_capturing_enable = enabled


class AclGraphWrapper:
    def __init__(self, model, capture_sizes):
        self.capture_sizes = capture_sizes
        self.model = model
        self.graphs = {}
        self.output_buffer = {}
        self.input_addresses = {}

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        # case prefill or num tokens lager than max capture size: eager mode
        if forward_context.is_prefill or forward_context.num_tokens > self.capture_sizes[-1]:
            return self.model(*args, **kwargs)

        num_tokens = forward_context.num_tokens
        num_actual_tokens = forward_context.num_actual_tokens
        if num_tokens not in self.graphs:
            self.graphs[num_tokens] = None
        
        graph = self.graphs[num_tokens]
        
        attn_metadata = forward_context.attn_metadata
        if graph is None:
            # store input addresses for debugging
            input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            attn_metadata_addresses = []
            for attr in dir(attn_metadata):
                if not attr.startswith('_') and torch.is_tensor(getattr(attn_metadata, attr)):
                    attn_metadata_addresses.append(getattr(attn_metadata, attr).data_ptr())
            input_addresses += attn_metadata_addresses
            self.input_addresses[num_tokens] = input_addresses

            # start capturing
            validate_aclgraph_capturing_enabled()
            aclgraph = torch.npu.NPUGraph()

            if get_global_graph_memory_pool() is None:
                set_global_graph_memory_pool(torch.npu.graph_pool_handle())

            forward_context.capturing = True
            with torch.npu.graph(
                npu_graph=aclgraph,
                pool=get_global_graph_memory_pool(),
                capture_error_mode="thread_local",
                auto_dispatch_capture=True
            ):
                output = self.model.forward(*args, **kwargs)

            self.output_buffer[num_tokens] = output
            self.graphs[num_tokens] = aclgraph
            graph = aclgraph
        
        # sync, update host input and replay
        torch.npu.synchronize()
        graph.replay()
        graph.update(
            cpu_update_input=[{"actual_seq_lengths_kv": attn_metadata[next(iter(attn_metadata))].seq_lens_list}]
        ) # NOTE: hardcode first attn_metadata.

        return self.output_buffer[num_tokens][:num_actual_tokens]

    def find_padding_bs(self, x):
        return next((bs for bs in self.capture_sizes if bs >= x), -1)
    
    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)
