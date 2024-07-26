// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <openvino/openvino.hpp>

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

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


struct Args {
    std::string model_path = "";
    std::string device = "GPU";
    int max_new_tokens = 100;
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
        << "\n"
        << "options:\n"
        << "  -h, --help              show this help message and exit\n"
        << "  -m, --model PATH        Chatglm OpenVINO model path (default: openvino_model.xml)\n"
        << "  -d, --device            Device (default: GPU)\n"
        << "  --max_new_tokens        max_new_tokens (default: 100)\n";
}

static Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if (arg == "-m" || arg == "--model") {
            args.model_path = argv[++i];
        }
        else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        }
        else if (arg == "--max_new_tokens") {
            args.max_new_tokens = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

static Args parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

int main(int argc, char* argv[]) try {
    Args args = parse_args(argc, argv);
    ov::genai::LLMPipeline pipe(args.model_path, args.device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = args.max_new_tokens;
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
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
