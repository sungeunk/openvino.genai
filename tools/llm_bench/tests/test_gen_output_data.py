# -*- coding: utf-8 -*-
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_bench_utils.gen_output_data import gen_token_timestamps  # noqa: E402

START = datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)


def elapsed_ms(begin, end):
    a = datetime.datetime.fromisoformat(begin)
    b = datetime.datetime.fromisoformat(end)
    return (b - a).total_seconds() * 1000


def test_first_token_span_matches_latency_in_milliseconds():
    ts = gen_token_timestamps(START, 1234.5)
    assert ts["first_token_begin"] == START.isoformat()
    assert elapsed_ms(ts["first_token_begin"], ts["first_token_end"]) == pytest.approx(1234.5)


def test_second_token_starts_at_first_token_end_and_spans_its_latency():
    ts = gen_token_timestamps(START, 1000.0, 25.0)
    assert ts["second_token_begin"] == ts["first_token_end"]
    assert elapsed_ms(ts["second_token_begin"], ts["second_token_end"]) == pytest.approx(25.0)


def test_second_token_omitted_when_latency_missing():
    ts = gen_token_timestamps(START, 1000.0)
    assert "second_token_begin" not in ts
    assert "second_token_end" not in ts


@pytest.mark.parametrize("first_token_latency_ms", [None, -1.0, 0.0])
def test_no_timestamps_for_missing_or_sentinel_latency(first_token_latency_ms):
    assert gen_token_timestamps(START, first_token_latency_ms) == {}


def test_no_timestamps_without_generation_start():
    assert gen_token_timestamps(None, 1000.0) == {}


def test_result_is_timezone_aware():
    ts = gen_token_timestamps(START, 1000.0, 25.0)
    for value in ts.values():
        assert datetime.datetime.fromisoformat(value).tzinfo is not None
