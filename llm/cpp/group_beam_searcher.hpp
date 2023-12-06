// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/runtime/tensor.hpp>

// Modifyed Knuth–Morris–Pratt algorithm which returns tokens following after every needle occurance in haystack
std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, const std::vector<int64_t>& needle) {
    if (needle.empty()) {  // NO_REPEAT_NGRAM_SIZE == 1, ban every token
        return {haystack.begin(), haystack.end()};
    }
    std::vector<int> partial_match_table(needle.size() + 1, -1);
    int cnd = 0;
    for (size_t pos = 1; pos < needle.size(); ++pos) {
        if (needle.at(pos) == needle.at(size_t(cnd))) {
            partial_match_table.at(pos) = partial_match_table.at(size_t(cnd));
        } else {
            partial_match_table.at(pos) = cnd;
            while (cnd >= 0 && needle.at(pos) != needle.at(size_t(cnd))) {
                cnd = partial_match_table.at(size_t(cnd));
            }
        }
        ++cnd;
    }
    partial_match_table.back() = cnd;
    std::vector<int64_t> res;
    size_t j = 0;  // The position of the current character in haystack
    int k = 0;  // The position of the current character in needle
    while (j < haystack.size() - 1) {
        if (needle.at(size_t(k)) == haystack.at(j)) {
            ++j;
            ++k;
            if (k == int(needle.size())) {
                res.push_back(haystack.at(j));
                k = partial_match_table.at(size_t(k));
            }
        } else {
            k = partial_match_table.at(size_t(k));
            if (k < 0) {
                ++j;
                ++k;
            }
        }
    }
    return res;
}

struct Beam {
    float score = -std::numeric_limits<float>::infinity();  // The bigger, the better
    std::vector<int64_t> tokens;
    size_t global_beam_idx = 0;
};

bool greater(const Beam& left, const Beam& right) {
    return left.score > right.score;
}

enum class StopCriteria {early, heuristic, never};

struct Parameters {
    std::vector<int64_t> prompt;
    size_t max_new_tokens;
    size_t n_groups;
    size_t group_size;
    StopCriteria stop_criteria;
    size_t no_repeat_ngram_size;
    float diversity_penalty;
    float length_penalty;
    // There's no way to extract special token values from the tokenizer for now
    int64_t eos_token;
    int64_t pad_token;
    std::function<bool(const Beam&)> early_finish = [](const Beam&){return false;};
};

struct Group {
    std::vector<Beam> ongoing;  // best beams in front
    std::vector<Beam> min_heap;  // The worst beam is the first
    bool done = false;
    void finish(Beam&& beam, const Parameters& parameters) {
        beam.score /= std::pow(float(parameters.prompt.size() + beam.tokens.size()), parameters.length_penalty);
        min_heap.push_back(std::move(beam));
        std::push_heap(min_heap.begin(), min_heap.end(), greater);
        if (min_heap.size() > parameters.group_size) {
            std::pop_heap(min_heap.begin(), min_heap.end(), greater);
            min_heap.pop_back();
        }
    }
    void is_done(const Parameters& parameters) {
        if (min_heap.size() < parameters.group_size) {
            return;
        }
        size_t cur_len = parameters.prompt.size() + ongoing.front().tokens.size();
        float best_sum_logprobs = ongoing.front().score;
        float worst_score = min_heap.front().score;
        switch (parameters.stop_criteria) {
            case StopCriteria::early:
                done = true;
                return;
            case StopCriteria::heuristic: {
                float highest_attainable_score = best_sum_logprobs / std::pow(float(cur_len), parameters.length_penalty);
                done = worst_score >= highest_attainable_score;
                return;
            }
            case StopCriteria::never: {
                size_t length = parameters.length_penalty > 0.0f ? parameters.max_new_tokens : cur_len;
                float highest_attainable_score = best_sum_logprobs / std::pow(float(length), parameters.length_penalty);
                done = worst_score >= highest_attainable_score;
                return;
            }
            default: throw std::runtime_error("Never reached");
        }
    }
};

struct TokenToBeam {int64_t token_idx; size_t beam_idx;};

struct GroupBeamSearcher {
    Parameters parameters;
    std::vector<Group> groups;
    GroupBeamSearcher(Parameters parameters) : parameters{std::move(parameters)}, groups{parameters.n_groups} {
        for (Group& group : groups) {
            group.ongoing.resize(parameters.group_size);
            group.ongoing.front().score = 0.0;
        }
    }
    std::vector<TokenToBeam> process(const ov::Tensor& logits) {
        std::vector<TokenToBeam> next_tokens;
        next_tokens.reserve(parameters.n_groups * parameters.group_size);
        size_t beam_count = 0;
        for (Group& group : groups) {
            if (!group.done) {
                for (Beam& beam : group.ongoing) {
                    beam.global_beam_idx = beam_count;
                    if (!beam.tokens.empty() && logits.get_shape().at(0) != 1) {
                        ++beam_count;
                    }
                }
            }
        }
        for (auto group = groups.begin(); group != groups.end(); ++group) {
            if (group->done) {
                for (Beam& beam : group->ongoing) {
                    beam.tokens.push_back(parameters.pad_token);
                }
                continue;
            }
            std::vector<Beam> candidates;
            candidates.reserve(2 * parameters.group_size);
            for (const Beam& beam : group->ongoing) {
                if (logits.get_shape().at(0) <= beam.global_beam_idx) {
                    throw std::runtime_error("logits batch size doesn't match the number of beams");
                }
                size_t vocab_size = logits.get_shape().back();
                size_t batch_offset = beam.global_beam_idx * logits.get_shape().at(1) * vocab_size;
                size_t sequence_offset = (logits.get_shape().at(1) - 1) * vocab_size;
                const float* beam_logits = logits.data<const float>() + batch_offset + sequence_offset;
                float max_logit = *std::max_element(beam_logits, beam_logits + vocab_size);
                float log_sum = std::log(std::accumulate(
                    beam_logits, beam_logits + vocab_size, 0.0f, [max_logit](float accumulated, float to_add) {
                        return accumulated + std::exp(to_add - max_logit);
                }));
                struct Token {float log_prob; int64_t idx;};
                std::vector<Token> tokens;
                tokens.reserve(vocab_size);
                for (size_t idx = 0; idx < vocab_size; ++idx) {
                    tokens.push_back({beam_logits[idx] - max_logit - log_sum, int64_t(idx)});
                }
                for (auto prev_group = groups.begin(); prev_group != group; ++prev_group) {
                    for (const Beam& prev_beam : prev_group->ongoing) {
                        tokens.at(size_t(prev_beam.tokens.back())).log_prob -= parameters.diversity_penalty;
                    }
                }
                std::vector<int64_t> full_text{parameters.prompt};
                full_text.insert(full_text.end(), beam.tokens.begin(), beam.tokens.end());
                if (full_text.size() > 1 && full_text.size() >= parameters.no_repeat_ngram_size) {
                    auto tail_start = full_text.end() - ptrdiff_t(parameters.no_repeat_ngram_size) + 1;
                    for (int64_t banned_token : kmp_search(full_text, {tail_start, full_text.end()})) {
                        tokens.at(size_t(banned_token)).log_prob = -std::numeric_limits<float>::infinity();
                    }
                }
                std::sort(tokens.begin(), tokens.end(), [](Token left, Token right) {
                    return left.log_prob > right.log_prob;  // Most probable tokens in front
                });
                size_t add_count = 0;
                for (Token token : tokens) {
                    Beam new_candidate = beam;
                    new_candidate.score += token.log_prob;
                    new_candidate.tokens.push_back(token.idx);
                    if (parameters.early_finish(new_candidate)) {
                        group->finish(std::move(new_candidate), parameters);
                    } else {
                        candidates.push_back(std::move(new_candidate));
                        ++add_count;
                        if (add_count == 2 * parameters.group_size) {
                            break;
                        }
                    }
                }
            }
            // Sample 2 * group_size highest score tokens to get at least 1 non EOS token per beam
            if (candidates.size() < 2 * parameters.group_size) {
                throw std::runtime_error("No beams left to search");
            }
            auto to_sort = candidates.begin() + ptrdiff_t(2 * parameters.group_size);
            std::partial_sort(candidates.begin(), to_sort, candidates.end(), greater);
            group->ongoing.clear();
            for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
                if (parameters.eos_token == candidates.at(cand_idx).tokens.back()) {
                    // if beam_token does not belong to top num_beams tokens, it should not be added
                    if (cand_idx >= parameters.group_size) {
                        continue;
                    }
                    candidates.at(cand_idx).tokens.resize(candidates.at(cand_idx).tokens.size() - 1);
                    group->finish(std::move(candidates.at(cand_idx)), parameters);
                } else {
                    group->ongoing.push_back(std::move(candidates.at(cand_idx)));
                    if (group->ongoing.size() == parameters.group_size) {
                        break;
                    }
                }
            }
            group->is_done(parameters);
            if (!group->done) {
                for (const Beam& beam : group->ongoing) {
                    next_tokens.push_back({beam.tokens.back(), beam.global_beam_idx});
                }
            }
        }
        return next_tokens;
    }
};
