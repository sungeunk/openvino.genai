#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to model")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Device")

    args = parser.parse_args()

    models_path = args.model
    device = args.device

    pipe = openvino_genai.LLMPipeline(models_path, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    # Predefined list of prompts
    prompts = [
        "Hello there! How are you doing?",
        "What is OpenVINO?",
        "Who are you?",
        "Can you explain to me briefly what is Python programming language?",
        "Explain the plot of Cinderella in a sentence.",
        "What are some common mistakes to avoid when writing code?",
        "Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“",
    ]

    pipe.start_chat()

    for prompt in prompts:
        print(f"question:\n{prompt}")
        pipe.generate(prompt, config, streamer)
        print('\n----------')
    pipe.finish_chat()


if '__main__' == __name__:
    main()
