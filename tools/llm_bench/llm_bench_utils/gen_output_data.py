# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import datetime


def gen_token_timestamps(generate_begin, first_token_latency_ms, generate_end=None):
    """Mark where the first token ends inside a generate() call, as wall-clock stamps.

    Three points, two spans: [generate_begin, first_token_end] is the first token and
    [first_token_end, generate_end] is the decode that follows. The two bounds are measured;
    only the split is derived, from the reported first token latency, because the C++ token
    timestamps use a steady clock whose epoch cannot be mapped onto wall time.

    The split is dropped unless it lands inside the measured window. A latency reported in
    the wrong unit, or a mean over a batch that does not describe a single token event, would
    otherwise name a span that never ran — and a consumer cutting monitoring samples by it
    has no way to notice. Losing the split costs the phase breakdown; keeping a bad one
    would silently attribute the wrong machine state to the first token.

    A single token is much shorter than a monitoring sampling interval, so the decode span
    deliberately covers every token after the first rather than just the second.

    Both bounds must be timezone-aware so the result can be compared with other UTC data.
    """
    if generate_begin is None or generate_end is None or generate_end <= generate_begin:
        return {}
    token_timestamps = {
        "generate_begin": generate_begin.isoformat(),
        "generate_end": generate_end.isoformat(),
    }
    if first_token_latency_ms is None or first_token_latency_ms <= 0:
        return token_timestamps
    first_token_end = generate_begin + datetime.timedelta(milliseconds=float(first_token_latency_ms))
    if first_token_end < generate_end:
        token_timestamps["first_token_end"] = first_token_end.isoformat()
    return token_timestamps


def gen_iterate_data(
    iter_idx="",
    in_size="",
    infer_count="",
    out_size="",
    gen_time="",
    latency="",
    res_md5="",
    max_rss_mem="",
    max_rss_mem_increase="",
    max_rss_mem_share="",
    max_sys_mem="",
    max_sys_mem_increase="",
    max_sys_mem_share="",
    prompt_idx="",
    tokenization_time=[],
    token_timestamps=None,
    mm_embeddings_preparation_time="",
    chat_idx="",
):
    iter_data = {}
    iter_data["iteration"] = iter_idx
    iter_data["input_size"] = in_size
    iter_data["infer_count"] = infer_count
    iter_data["output_size"] = out_size
    iter_data["generation_time"] = gen_time
    iter_data["latency"] = latency
    iter_data["result_md5"] = res_md5
    iter_data["first_token_latency"] = -1
    iter_data["other_tokens_avg_latency"] = -1
    iter_data["first_token_infer_latency"] = -1
    iter_data["other_tokens_infer_avg_latency"] = -1
    iter_data["max_rss_mem_consumption"] = max_rss_mem
    iter_data["max_rss_mem_increase"] = max_rss_mem_increase
    iter_data["max_rss_mem_share"] = max_rss_mem_share
    iter_data["max_sys_mem_consumption"] = max_sys_mem
    iter_data["max_sys_mem_increase"] = max_sys_mem_increase
    iter_data["max_sys_mem_share"] = max_sys_mem_share
    iter_data["prompt_idx"] = prompt_idx
    iter_data["tokenization_time"] = tokenization_time[0] if len(tokenization_time) > 0 else ""
    iter_data["detokenization_time"] = tokenization_time[1] if len(tokenization_time) > 1 else ""
    iter_data["mm_embeddings_preparation_time"] = mm_embeddings_preparation_time
    iter_data["chat_idx"] = chat_idx
    iter_data["token_timestamps"] = token_timestamps or {}
    return iter_data


def embed_iterate_data(
    iter_idx="",
    in_size="",
    infer_count="",
    total_time="",
    latency="",
    available_mem="",
    max_rss_mem="",
    max_rss_mem_increase="",
    max_rss_mem_share="",
    max_sys_mem="",
    max_sys_mem_increase="",
    max_sys_mem_share="",
    prompt_idx="",
    tokenization_time=[],
):
    iter_data = {}
    iter_data["iteration"] = iter_idx
    iter_data["input_size"] = in_size
    iter_data["infer_count"] = infer_count
    iter_data["generation_time"] = total_time
    iter_data["latency"] = latency
    iter_data["first_token_latency"] = -1
    iter_data["other_tokens_avg_latency"] = -1
    iter_data["first_token_infer_latency"] = -1
    iter_data["other_tokens_infer_avg_latency"] = -1
    iter_data["available_mem"] = available_mem
    iter_data["max_rss_mem_consumption"] = max_rss_mem
    iter_data["max_rss_mem_increase"] = max_rss_mem_increase
    iter_data["max_rss_mem_share"] = max_rss_mem_share
    iter_data["max_sys_mem_consumption"] = max_sys_mem
    iter_data["max_sys_mem_increase"] = max_sys_mem_increase
    iter_data["max_sys_mem_share"] = max_sys_mem_share
    iter_data["prompt_idx"] = prompt_idx
    iter_data["tokenization_time"] = tokenization_time[0] if len(tokenization_time) > 0 else ""
    iter_data["detokenization_time"] = ""
    iter_data["result_md5"] = ""
    iter_data["output_size"] = ""
    return iter_data
