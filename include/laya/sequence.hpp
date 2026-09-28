#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "laya/pyjson.hpp"
#include "laya/tokenizer.hpp"

namespace laya {

enum class QType : int32_t { Choice = 0, Score = 1, Noul = 2 };

struct Question {
    QType type = QType::Choice;
    std::string type_name;
    std::string instructions;
    std::vector<std::pair<py::Json, py::Json>> options;
    std::optional<py::Json> noul_criteria;
    std::optional<py::Json> labels;
};

struct OptionStats {
    int32_t options = 0;
    int32_t options_distinct = 0;
    std::optional<int32_t> tokens_per_option;
};

struct SequenceItem {
    std::string qid;
    std::vector<int32_t> ids;
    std::vector<int32_t> markers;
    int32_t qtype = 0;
    OptionStats stats;
};

struct AgentConfig {
    int32_t max_len = 512;
    int32_t head_max_len = 192;

    static AgentConfig from_file(const std::string& path);
};

class QuestionError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

void check_question(const std::string& qid, const py::Json& definition);
Question to_internal(const py::Json& definition);
std::vector<std::string> render_options(const Question& q);
std::string render_criterion(const py::Json& value);
std::string serialize_state(const py::Json& state);

SequenceItem build_sequence(const Tokenizer& tok, const Question& q, const std::vector<int32_t>& state_ids,
                            int32_t max_len, int32_t head_max_len, bool truncate_left);

std::vector<SequenceItem> encode_state(const Tokenizer& tok, const py::Json& state, const py::Json& questions,
                                       int32_t max_len, int32_t head_max_len);

}
