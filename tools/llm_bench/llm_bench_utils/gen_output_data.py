# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import datetime


def gen_token_timestamps(generation_start, first_token_latency_ms, second_token_latency_ms=None):
    """Reconstruct first/second token boundaries from the generation start time and token latencies.

    The boundaries are derived from averaged latency metrics rather than captured at the token
    events themselves, so they are only precise enough to align a run against wall-clock data
    such as machine monitoring samples. With batching or speculative decoding the latencies are
    per-token averages, which widens that gap further.

    `generation_start` must be timezone-aware so the result can be compared with other UTC data.
    """
    if generation_start is None or first_token_latency_ms is None or first_token_latency_ms <= 0:
        return {}
    first_token_end = generation_start + datetime.timedelta(milliseconds=float(first_token_latency_ms))
    token_timestamps = {
        "first_token_begin": generation_start.isoformat(),
        "first_token_end": first_token_end.isoformat(),
    }
    if second_token_latency_ms is not None and second_token_latency_ms > 0:
        second_token_end = first_token_end + datetime.timedelta(milliseconds=float(second_token_latency_ms))
        token_timestamps["second_token_begin"] = first_token_end.isoformat()
        token_timestamps["second_token_end"] = second_token_end.isoformat()
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
