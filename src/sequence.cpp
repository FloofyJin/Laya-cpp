#include "laya/sequence.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>

#include "laya/unicode.hpp"

namespace laya {

using py::Json;

namespace {

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return s;
    }
    std::string out;
    size_t pos = 0;
    for (;;) {
        size_t hit = s.find(from, pos);
        if (hit == std::string::npos) {
            out.append(s, pos, std::string::npos);
            return out;
        }
        out.append(s, pos, hit - pos);
        out += to;
        pos = hit + from.size();
    }
}

std::string ascii_lower(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return out;
}

int32_t floor_div(int32_t a, int32_t b) {
    int32_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) {
        --q;
    }
    return q;
}

std::string quoted(const std::string& qid) {
    return py::dumps(Json(qid));
}

[[noreturn]] void fail(const std::string& qid, const std::string& message) {
    throw QuestionError("question " + quoted(qid) + ": " + message);
}

bool is_nested(const Json& v) {
    return v.is_array() || v.is_object();
}

struct LabelKey {
    bool is_string;
    std::string s;
    double d;
    bool operator<(const LabelKey& o) const {
        if (is_string != o.is_string) {
            return is_string < o.is_string;
        }
        return is_string ? s < o.s : d < o.d;
    }
};

LabelKey label_key(const Json& v) {
    if (v.is_string()) {
        return {true, v.get<std::string>(), 0.0};
    }
    if (v.is_boolean()) {
        return {false, {}, v.get<bool>() ? 1.0 : 0.0};
    }
    return {false, {}, v.get<double>()};
}

std::pair<std::string, std::string> resolve_noul_labels(const std::optional<Json>& labels) {
    if (!labels) {
        return {"false", "true"};
    }
    const std::string msg = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";
    const Json& l = *labels;
    if (!l.is_object() || l.size() != 2 || !l.contains("false") || !l.contains("true")) {
        throw QuestionError(msg);
    }
    if (!l["false"].is_string() || !l["true"].is_string()) {
        throw QuestionError(msg);
    }
    std::string f = unicode::python_strip(l["false"].get<std::string>());
    std::string t = unicode::python_strip(l["true"].get<std::string>());
    if (f.empty() || t.empty() || f == t) {
        throw QuestionError(msg);
    }
    return {f, t};
}

bool is_blank(const Json& v) {
    return v.is_null() || (v.is_string() && v.get_ref<const std::string&>().empty());
}

}

AgentConfig AgentConfig::from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const Json cfg = Json::parse(ss.str());
    AgentConfig c;
    if (cfg.contains("max_len")) {
        c.max_len = cfg["max_len"].get<int32_t>();
    }
    if (cfg.contains("head_max_len")) {
        c.head_max_len = cfg["head_max_len"].get<int32_t>();
    }
    return c;
}

void check_question(const std::string& qid, const Json& q) {
    if (!q.is_object()) {
        fail(qid, "definition must be a dict");
    }
    const Json type = q.contains("type") ? q["type"] : Json();
    if (!type.is_string() || (type != "choice" && type != "score" && type != "noul")) {
        fail(qid, "unknown type " + py::dumps(type) + "; use one of ['choice', 'noul', 'score']");
    }
    if (!q.contains("instructions")) {
        fail(qid, "no 'instructions'; add the text the model should answer");
    }
    const std::string t = type.get<std::string>();
    const Json crit = q.contains("criteria") ? q["criteria"] : Json();
    if (t == "choice") {
        if (!crit.is_object() && !crit.is_array()) {
            fail(qid, "a choice question takes 'criteria' as a dict of label -> description, or a list of labels");
        }
        if (crit.empty()) {
            fail(qid, "a choice question needs at least one criterion");
        }
        if (crit.is_array()) {
            std::set<LabelKey> seen;
            for (size_t i = 0; i < crit.size(); ++i) {
                const Json& label = crit[i];
                if (is_nested(label)) {
                    fail(qid, "choice label " + std::to_string(i) + " must be a scalar");
                }
                if (label.is_null()) {
                    fail(qid, "choice label " + std::to_string(i) + " is null");
                }
                if (!seen.insert(label_key(label)).second) {
                    fail(qid, "choice label " + std::to_string(i) + " repeats an earlier label");
                }
            }
        }
    } else if (t == "score") {
        if (!crit.is_array()) {
            fail(qid, "a score question takes 'criteria' as a list of level descriptions, index 0 first");
        }
        if (crit.empty()) {
            fail(qid, "a score question needs at least one level");
        }
        for (size_t i = 0; i < crit.size(); ++i) {
            if (crit[i].is_null()) {
                fail(qid, "score level " + std::to_string(i) + " is null");
            }
        }
    } else if (!crit.is_null() && !crit.is_object()) {
        fail(qid, "a noul question takes 'criteria' as a dict with optional 'true'/'false' descriptions");
    } else if (crit.is_object()) {
        for (auto it = crit.begin(); it != crit.end(); ++it) {
            const std::string k = ascii_lower(it.key());
            if (k != "true" && k != "false") {
                fail(qid, "a noul question takes 'criteria' keyed only 'true'/'false'");
            }
        }
    }
    if (q.contains("labels")) {
        if (t != "noul") {
            fail(qid, "'labels' is only supported for noul questions");
        }
        try {
            resolve_noul_labels(q["labels"]);
        } catch (const QuestionError& e) {
            fail(qid, e.what());
        }
    }
}

Question to_internal(const Json& def) {
    Question q;
    q.type_name = def["type"].get<std::string>();
    q.type = q.type_name == "choice" ? QType::Choice : q.type_name == "score" ? QType::Score : QType::Noul;
    const Json crit = def.contains("criteria") ? def["criteria"] : Json();
    if (q.type == QType::Choice) {
        if (crit.is_array()) {
            for (const Json& label : crit) {
                q.options.emplace_back(label, Json());
            }
        } else {
            for (auto it = crit.begin(); it != crit.end(); ++it) {
                q.options.emplace_back(Json(it.key()), it.value());
            }
        }
    } else if (q.type == QType::Score) {
        for (const Json& level : crit) {
            q.options.emplace_back(Json(), level);
        }
    } else if (crit.is_object()) {
        Json lowered = Json::object();
        for (auto it = crit.begin(); it != crit.end(); ++it) {
            lowered[ascii_lower(it.key())] = it.value();
        }
        q.noul_criteria = lowered;
    }
    const Json& ins = def["instructions"];
    q.instructions = ins.is_string() ? ins.get<std::string>() : py::dumps(ins);
    if (def.contains("labels")) {
        q.labels = def["labels"];
    }
    return q;
}

std::string render_criterion(const Json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    return py::dumps(value);
}

std::vector<std::string> render_options(const Question& q) {
    std::vector<std::string> out;
    if (q.type != QType::Noul && q.labels) {
        throw QuestionError("labels is only supported for noul questions");
    }
    if (q.type == QType::Choice) {
        for (const auto& [label, desc] : q.options) {
            if (is_blank(desc)) {
                out.push_back(py::str(label));
            } else {
                out.push_back(py::str(label) + ": " + render_criterion(desc));
            }
        }
        return out;
    }
    if (q.type == QType::Score) {
        int i = 0;
        for (const auto& opt : q.options) {
            out.push_back("level " + std::to_string(i++) + ": " + render_criterion(opt.second));
        }
        return out;
    }
    const auto [false_label, true_label] = resolve_noul_labels(q.labels);
    Json fc;
    Json tc;
    if (q.noul_criteria) {
        if (q.noul_criteria->contains("false")) {
            fc = (*q.noul_criteria)["false"];
        }
        if (q.noul_criteria->contains("true")) {
            tc = (*q.noul_criteria)["true"];
        }
    }
    out.push_back(false_label + ": " + (is_blank(fc) ? std::string("no, the statement does not hold") : render_criterion(fc)));
    out.push_back(true_label + ": " + (is_blank(tc) ? std::string("yes, the statement holds") : render_criterion(tc)));
    return out;
}

std::string serialize_state(const Json& state) {
    if (state.is_string()) {
        return state.get<std::string>();
    }
    return py::dumps(state);
}

SequenceItem build_sequence(const Tokenizer& tok, const Question& q, const std::vector<int32_t>& state_ids,
                            int32_t max_len, int32_t head_max_len, bool truncate_left) {
    const std::string& mask = tok.mask_token();
    const std::vector<std::string> opts = render_options(q);
    const std::string ins = replace_all(q.instructions, mask, " ");
    std::vector<int32_t> head_ids = tok.encode(q.type_name + " question: " + ins);

    std::vector<std::vector<int32_t>> opt_ids;
    for (const std::string& opt : opts) {
        std::vector<int32_t> o{tok.mask_id()};
        const std::vector<int32_t> body = tok.encode(" " + replace_all(opt, mask, " "), 48);
        o.insert(o.end(), body.begin(), body.end());
        opt_ids.push_back(std::move(o));
    }

    auto total = [&]() {
        int32_t n = 0;
        for (const auto& o : opt_ids) {
            n += static_cast<int32_t>(o.size());
        }
        return n;
    };
    int32_t opt_budget = head_max_len - total();
    std::optional<int32_t> per_option;
    if (opt_budget < 16) {
        const int32_t count = std::max<int32_t>(1, static_cast<int32_t>(opt_ids.size()));
        const int32_t per = std::max<int32_t>(4, floor_div(head_max_len - 16, count));
        per_option = per;
        for (auto& o : opt_ids) {
            if (static_cast<int32_t>(o.size()) > per) {
                o.resize(static_cast<size_t>(per));
            }
        }
        opt_budget = head_max_len - total();
    }
    const size_t head_keep = static_cast<size_t>(std::max<int32_t>(8, opt_budget));
    if (head_ids.size() > head_keep) {
        head_ids.resize(head_keep);
    }

    std::vector<int32_t> ids{tok.cls_id()};
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(tok.sep_id());
    std::vector<int32_t> markers;
    for (const auto& o : opt_ids) {
        markers.push_back(static_cast<int32_t>(ids.size()));
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(tok.sep_id());

    const int32_t room = std::max<int32_t>(0, max_len - static_cast<int32_t>(ids.size()) - 1);
    const size_t n_state = state_ids.size();
    const size_t take = std::min(n_state, static_cast<size_t>(room));
    if (truncate_left) {
        ids.insert(ids.end(), state_ids.end() - static_cast<std::ptrdiff_t>(take), state_ids.end());
    } else {
        ids.insert(ids.end(), state_ids.begin(), state_ids.begin() + static_cast<std::ptrdiff_t>(take));
    }
    ids.push_back(tok.sep_id());

    SequenceItem item;
    const size_t limit = max_len < 0 ? 0 : static_cast<size_t>(max_len);
    if (ids.size() > limit) {
        ids.resize(limit);
    }
    item.ids = std::move(ids);
    for (int32_t m : markers) {
        if (m < max_len) {
            item.markers.push_back(m);
        }
    }
    item.qtype = static_cast<int32_t>(q.type);
    item.stats.options = static_cast<int32_t>(opt_ids.size());
    std::set<std::vector<int32_t>> distinct(opt_ids.begin(), opt_ids.end());
    item.stats.options_distinct = static_cast<int32_t>(distinct.size());
    item.stats.tokens_per_option = per_option;
    return item;
}

std::vector<SequenceItem> encode_state(const Tokenizer& tok, const Json& state, const Json& questions,
                                       int32_t max_len, int32_t head_max_len) {
    if (!questions.is_object()) {
        throw QuestionError("questions must be a dict of question id -> definition");
    }
    if (state.is_null()) {
        throw QuestionError("state must not be None; pass a string, dict, or list");
    }
    std::vector<std::pair<std::string, Question>> internal;
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        check_question(it.key(), it.value());
    }
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        internal.emplace_back(it.key(), to_internal(it.value()));
    }
    std::vector<SequenceItem> items;
    if (internal.empty()) {
        return items;
    }
    const bool truncate_left = state.is_array();
    const std::vector<int32_t> state_ids = tok.encode(replace_all(serialize_state(state), tok.mask_token(), " "));
    for (const auto& [qid, q] : internal) {
        SequenceItem item = build_sequence(tok, q, state_ids, max_len, head_max_len, truncate_left);
        if (item.markers.size() != render_options(q).size()) {
            throw QuestionError("question " + quoted(qid) + " options exceed head_max_len=" + std::to_string(head_max_len));
        }
        item.qid = qid;
        items.push_back(std::move(item));
    }
    return items;
}

}
