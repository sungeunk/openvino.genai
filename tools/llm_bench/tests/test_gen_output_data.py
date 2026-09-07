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
    ts = gen_token_timestamps(START, 1234.5, START + datetime.timedelta(seconds=9))
    assert ts["generate_begin"] == START.isoformat()
    assert elapsed_ms(ts["generate_begin"], ts["first_token_end"]) == pytest.approx(1234.5)


def test_decode_span_runs_from_first_token_end_to_generate_end():
    generate_end = START + datetime.timedelta(seconds=9)
    ts = gen_token_timestamps(START, 1000.0, generate_end)
    assert ts["generate_end"] == generate_end.isoformat()
    assert elapsed_ms(ts["first_token_end"], ts["generate_end"]) == pytest.approx(8000.0)


@pytest.mark.parametrize("first_token_latency_ms", [None, -1.0, 0.0, 9001.0])
def test_measured_bounds_survive_an_unusable_split(first_token_latency_ms):
    generate_end = START + datetime.timedelta(seconds=9)
    ts = gen_token_timestamps(START, first_token_latency_ms, generate_end)
    assert ts == {"generate_begin": START.isoformat(),
                  "generate_end": generate_end.isoformat()}


def test_nothing_without_a_measured_window():
    assert gen_token_timestamps(None, 1000.0, START) == {}
    assert gen_token_timestamps(START, 1000.0) == {}
    assert gen_token_timestamps(START, 1000.0, START) == {}
    assert gen_token_timestamps(START, 1000.0, START - datetime.timedelta(seconds=1)) == {}


def test_result_is_timezone_aware():
    ts = gen_token_timestamps(START, 1000.0, START + datetime.timedelta(seconds=5))
    for value in ts.values():
        assert datetime.datetime.fromisoformat(value).tzinfo is not None
