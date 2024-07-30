// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

const std::string prompts[] =
{
    "Hello there! How are you doing?",
    "What is OpenVINO?",
    "Who are you?",
    "Can you explain to me briefly what is Python programming language?",
    "Explain the plot of Cinderella in a sentence.",
    "What are some common mistakes to avoid when writing code?",
    "Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“",
};

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DEVICE>");
    }
    std::string prompt;
    std::string model_path = argv[1];
    std::string device = argv[2];  // GPU, NPU can be used as well

    ov::genai::LLMPipeline pipe(model_path, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    std::function<bool(std::string)> streamer = [](std::string word) { 
        std::cout << word << std::flush;
        // Return flag corresponds whether generation should be stopped.
        // false means continue generation.
        return false; 
    };

    pipe.start_chat();
    for (std::string prompt : prompts) {
        std::cout << "question:\n";
        std::cout << prompt << std::endl;

        pipe.generate(prompt, config, streamer);

        std::cout << "\n----------\n";
    }
    pipe.finish_chat();
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
