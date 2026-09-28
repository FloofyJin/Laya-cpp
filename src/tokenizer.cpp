#include "laya/tokenizer.hpp"

#include <algorithm>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "laya/unicode.hpp"

namespace laya {

using json = nlohmann::json;

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

uint64_t pair_key(int32_t a, int32_t b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

bool is_other(char32_t c) {
    return !unicode::is_white_space(c) && !unicode::is_letter(c) && !unicode::is_number(c);
}

std::vector<std::pair<size_t, size_t>> byte_level_split(const std::u32string& s) {
    std::vector<std::pair<size_t, size_t>> out;
    const size_t n = s.size();
    size_t i = 0;
    auto run = [&](size_t j, bool (*pred)(char32_t)) {
        while (j < n && pred(s[j])) {
            ++j;
        }
        return j;
    };
    while (i < n) {
        const char32_t c = s[i];
        if (c == U'\'' && i + 1 < n) {
            const char32_t a = s[i + 1];
            if (a == U's' || a == U't' || a == U'm' || a == U'd') {
                out.emplace_back(i, i + 2);
                i += 2;
                continue;
            }
            if (i + 2 < n) {
                const char32_t b = s[i + 2];
                if ((a == U'r' && b == U'e') || (a == U'v' && b == U'e') || (a == U'l' && b == U'l')) {
                    out.emplace_back(i, i + 3);
                    i += 3;
                    continue;
                }
            }
        }
        const bool spaced = c == U' ' && i + 1 < n;
        if (spaced && unicode::is_letter(s[i + 1])) {
            size_t j = run(i + 1, unicode::is_letter);
            out.emplace_back(i, j);
            i = j;
            continue;
        }
        if (unicode::is_letter(c)) {
            size_t j = run(i, unicode::is_letter);
            out.emplace_back(i, j);
            i = j;
            continue;
        }
        if (spaced && unicode::is_number(s[i + 1])) {
            size_t j = run(i + 1, unicode::is_number);
            out.emplace_back(i, j);
            i = j;
            continue;
        }
        if (unicode::is_number(c)) {
            size_t j = run(i, unicode::is_number);
            out.emplace_back(i, j);
            i = j;
            continue;
        }
        if (spaced && is_other(s[i + 1])) {
            size_t j = run(i + 1, is_other);
            out.emplace_back(i, j);
            i = j;
            continue;
        }
        if (is_other(c)) {
            size_t j = run(i, is_other);
            out.emplace_back(i, j);
            i = j;
            continue;
        }
        size_t k = run(i, unicode::is_white_space);
        size_t end = (k == n || k - i < 2) ? k : k - 1;
        out.emplace_back(i, end);
        i = end;
    }
    return out;
}

bool ends_with_word(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    std::u32string cps = unicode::decode(s);
    return unicode::is_word(cps.back());
}

bool starts_with_word(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    size_t len = 1;
    const unsigned char lead = static_cast<unsigned char>(s[0]);
    if (lead >= 0xF0) {
        len = 4;
    } else if (lead >= 0xE0) {
        len = 3;
    } else if (lead >= 0xC0) {
        len = 2;
    }
    std::u32string cps = unicode::decode(s.substr(0, std::min(len, s.size())));
    return !cps.empty() && unicode::is_word(cps.front());
}

size_t space_leftmost_at_end(const std::string& s) {
    std::u32string cps = unicode::decode(s);
    size_t bytes = s.size();
    size_t k = cps.size();
    while (k > 0 && unicode::is_white_space(cps[k - 1])) {
        std::string tmp;
        unicode::append(tmp, cps[k - 1]);
        bytes -= tmp.size();
        --k;
    }
    return bytes;
}

size_t space_rightmost_at_start(const std::string& s) {
    std::u32string cps = unicode::decode(s);
    size_t bytes = 0;
    for (char32_t c : cps) {
        if (!unicode::is_white_space(c)) {
            break;
        }
        std::string tmp;
        unicode::append(tmp, c);
        bytes += tmp.size();
    }
    return bytes;
}

std::string special_content(const json& cfg, const char* name) {
    if (!cfg.contains(name)) {
        return {};
    }
    const json& v = cfg[name];
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_object() && v.contains("content") && v["content"].is_string()) {
        return v["content"].get<std::string>();
    }
    return {};
}

}

void Tokenizer::Trie::insert(const std::string& key, int32_t token) {
    int32_t node = 0;
    for (unsigned char ch : key) {
        auto& next = nodes[static_cast<size_t>(node)].next;
        auto it = std::find_if(next.begin(), next.end(), [&](const auto& p) { return p.first == ch; });
        if (it == next.end()) {
            nodes.push_back(TrieNode{});
            int32_t child = static_cast<int32_t>(nodes.size() - 1);
            nodes[static_cast<size_t>(node)].next.emplace_back(ch, child);
            node = child;
        } else {
            node = it->second;
        }
    }
    if (nodes[static_cast<size_t>(node)].token < 0) {
        nodes[static_cast<size_t>(node)].token = token;
    }
}

std::pair<size_t, int32_t> Tokenizer::Trie::longest_at(const std::string& text, size_t pos) const {
    size_t best_len = 0;
    int32_t best = -1;
    int32_t node = 0;
    for (size_t i = pos; i < text.size(); ++i) {
        const auto& next = nodes[static_cast<size_t>(node)].next;
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        auto it = std::find_if(next.begin(), next.end(), [&](const auto& p) { return p.first == ch; });
        if (it == next.end()) {
            break;
        }
        node = it->second;
        if (nodes[static_cast<size_t>(node)].token >= 0) {
            best_len = i + 1 - pos;
            best = nodes[static_cast<size_t>(node)].token;
        }
    }
    return {best_len, best};
}

Tokenizer Tokenizer::from_directory(const std::string& dir) {
    return from_files(dir + "/tokenizer.json", dir + "/tokenizer_config.json");
}

Tokenizer Tokenizer::from_files(const std::string& tokenizer_json, const std::string& tokenizer_config_json) {
    Tokenizer t;
    t.load_tokenizer_json(tokenizer_json);
    t.load_special_tokens(tokenizer_config_json);
    return t;
}

void Tokenizer::load_tokenizer_json(const std::string& path) {
    const json root = json::parse(read_file(path));

    const json& model = root.at("model");
    if (model.value("type", std::string("BPE")) != "BPE") {
        throw std::runtime_error("only BPE tokenizer models are supported");
    }
    if (model.contains("dropout") && !model["dropout"].is_null()) {
        throw std::runtime_error("BPE dropout is not supported");
    }
    for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
        if (model.contains(key) && model[key].is_string() && !model[key].get<std::string>().empty()) {
            throw std::runtime_error(std::string("BPE ") + key + " is not supported");
        }
    }
    ignore_merges_ = model.value("ignore_merges", false);

    for (const auto& [token, id] : model.at("vocab").items()) {
        const int32_t v = id.get<int32_t>();
        vocab_[token] = v;
        vocab_r_[v] = token;
    }
    if (model.contains("unk_token") && model["unk_token"].is_string()) {
        auto it = vocab_.find(model["unk_token"].get<std::string>());
        if (it != vocab_.end()) {
            unk_id_ = it->second;
        }
    }

    const json& merges = model.at("merges");
    merges_.reserve(merges.size() * 2);
    uint32_t rank = 0;
    for (const json& m : merges) {
        std::string a;
        std::string b;
        if (m.is_string()) {
            const std::string s = m.get<std::string>();
            const size_t sp = s.find(' ');
            if (sp == std::string::npos || s.find(' ', sp + 1) != std::string::npos) {
                throw std::runtime_error("malformed merge: " + s);
            }
            a = s.substr(0, sp);
            b = s.substr(sp + 1);
        } else {
            a = m.at(0).get<std::string>();
            b = m.at(1).get<std::string>();
        }
        auto ia = vocab_.find(a);
        auto ib = vocab_.find(b);
        auto in = vocab_.find(a + b);
        if (ia == vocab_.end() || ib == vocab_.end() || in == vocab_.end()) {
            throw std::runtime_error("merge token out of vocabulary: " + a + " " + b);
        }
        merges_[pair_key(ia->second, ib->second)] = {rank, in->second};
        ++rank;
    }

    const json& norm = root.contains("normalizer") ? root["normalizer"] : json();
    auto check_norm = [&](const json& n) {
        const std::string type = n.at("type").get<std::string>();
        if (type == "NFC") {
            nfc_ = true;
        } else {
            throw std::runtime_error("unsupported normalizer: " + type);
        }
    };
    if (!norm.is_null()) {
        if (norm.at("type") == "Sequence") {
            for (const json& n : norm.at("normalizers")) {
                check_norm(n);
            }
        } else {
            check_norm(norm);
        }
    }

    const json& pre = root.contains("pre_tokenizer") ? root["pre_tokenizer"] : json();
    bool byte_level = false;
    auto check_pre = [&](const json& p) {
        const std::string type = p.at("type").get<std::string>();
        if (type != "ByteLevel" || byte_level) {
            throw std::runtime_error("unsupported pre_tokenizer: " + type);
        }
        byte_level = true;
        add_prefix_space_ = p.value("add_prefix_space", true);
        use_regex_ = p.value("use_regex", true);
    };
    if (!pre.is_null()) {
        if (pre.at("type") == "Sequence") {
            for (const json& p : pre.at("pretokenizers")) {
                check_pre(p);
            }
        } else {
            check_pre(pre);
        }
    }
    if (!byte_level) {
        throw std::runtime_error("tokenizer.json must use the ByteLevel pre_tokenizer");
    }

    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::array<char32_t, 256> map{};
    std::array<bool, 256> seen{};
    for (int b : bs) {
        map[static_cast<size_t>(b)] = static_cast<char32_t>(b);
        seen[static_cast<size_t>(b)] = true;
    }
    char32_t extra = 0;
    for (int b = 0; b < 256; ++b) {
        if (!seen[static_cast<size_t>(b)]) {
            map[static_cast<size_t>(b)] = 256 + extra++;
        }
    }
    for (size_t b = 0; b < 256; ++b) {
        byte_chars_[b].clear();
        unicode::append(byte_chars_[b], map[b]);
        auto it = vocab_.find(byte_chars_[b]);
        byte_ids_[b] = it == vocab_.end() ? -1 : it->second;
    }

    if (root.contains("added_tokens")) {
        for (const json& t : root["added_tokens"]) {
            AddedToken a;
            a.content = t.at("content").get<std::string>();
            a.id = t.at("id").get<int32_t>();
            a.single_word = t.value("single_word", false);
            a.lstrip = t.value("lstrip", false);
            a.rstrip = t.value("rstrip", false);
            a.normalized = t.value("normalized", false);
            a.special = t.value("special", false);
            added_ids_[a.content] = a.id;
            vocab_r_.try_emplace(a.id, a.content);
            added_.push_back(a);
        }
    }
    for (size_t i = 0; i < added_.size(); ++i) {
        const AddedToken& a = added_[i];
        if (a.content.empty()) {
            continue;
        }
        if (a.normalized) {
            normalized_trie_.insert(normalize(a.content), static_cast<int32_t>(i));
        } else {
            raw_trie_.insert(a.content, static_cast<int32_t>(i));
        }
    }
}

void Tokenizer::load_special_tokens(const std::string& path) {
    json cfg = json::object();
    {
        std::ifstream probe(path);
        if (probe) {
            cfg = json::parse(read_file(path));
        }
    }
    auto resolve = [&](const char* name, std::initializer_list<const char*> aliases, int32_t& id,
                       std::string* token_out) {
        std::string token = special_content(cfg, name);
        if (token.empty()) {
            for (const char* a : aliases) {
                if (token_to_id(a)) {
                    token = a;
                    break;
                }
            }
        }
        auto v = token.empty() ? std::nullopt : token_to_id(token);
        if (!v) {
            throw std::runtime_error(std::string("tokenizer is missing a valid ") + name);
        }
        id = *v;
        if (token_out) {
            *token_out = token;
        }
    };
    resolve("cls_token", {"[CLS]", "<bos>", "<s>"}, cls_id_, nullptr);
    resolve("sep_token", {"[SEP]", "<eos>", "</s>"}, sep_id_, nullptr);
    resolve("pad_token", {"[PAD]", "<pad>"}, pad_id_, nullptr);
    resolve("mask_token", {"[MASK]", "<mask>"}, mask_id_, &mask_token_);
}

std::optional<int32_t> Tokenizer::token_to_id(const std::string& token) const {
    auto a = added_ids_.find(token);
    if (a != added_ids_.end()) {
        return a->second;
    }
    auto v = vocab_.find(token);
    if (v != vocab_.end()) {
        return v->second;
    }
    return std::nullopt;
}

std::string Tokenizer::id_to_token(int32_t id) const {
    auto it = vocab_r_.find(id);
    return it == vocab_r_.end() ? std::string() : it->second;
}

std::string Tokenizer::normalize(const std::string& text) const {
    return nfc_ ? unicode::nfc(text) : text;
}

std::vector<Tokenizer::Piece> Tokenizer::split_added(const std::string& sentence, const Trie& trie) const {
    std::vector<Piece> out;
    if (sentence.empty()) {
        return out;
    }
    std::vector<std::pair<size_t, size_t>> spans;
    std::vector<int32_t> which;
    for (size_t i = 0; i < sentence.size();) {
        auto [len, tok] = trie.longest_at(sentence, i);
        if (tok >= 0) {
            spans.emplace_back(i, i + len);
            which.push_back(tok);
            i += len;
        } else {
            ++i;
        }
    }
    size_t start_offset = 0;
    for (size_t m = 0; m < spans.size(); ++m) {
        size_t start = spans[m].first;
        size_t stop = spans[m].second;
        const AddedToken& tok = added_[static_cast<size_t>(which[m])];
        if (tok.single_word) {
            const bool start_space = start == 0 || !ends_with_word(sentence.substr(0, start));
            const bool stop_space = stop == sentence.size() || !starts_with_word(sentence.substr(stop));
            if (!start_space || !stop_space) {
                continue;
            }
        }
        if (tok.lstrip) {
            start = std::max(space_leftmost_at_end(sentence.substr(0, start)), start_offset);
        }
        if (tok.rstrip) {
            stop += space_rightmost_at_start(sentence.substr(stop));
        }
        if (start_offset < start) {
            out.push_back({sentence.substr(start_offset, start - start_offset), -1});
        }
        out.push_back({sentence.substr(start, stop - start), tok.id});
        start_offset = stop;
    }
    if (start_offset != sentence.size()) {
        out.push_back({sentence.substr(start_offset), -1});
    }
    return out;
}

void Tokenizer::encode_word(const std::string& word, std::vector<int32_t>& out) const {
    if (word.empty()) {
        return;
    }
    if (ignore_merges_) {
        std::string mapped;
        for (unsigned char b : word) {
            mapped += byte_chars_[b];
        }
        auto it = vocab_.find(mapped);
        if (it != vocab_.end()) {
            out.push_back(it->second);
            return;
        }
    }

    struct Symbol {
        int32_t c;
        int32_t prev;
        int32_t next;
        bool alive;
    };
    std::vector<Symbol> sym;
    sym.reserve(word.size());
    for (unsigned char b : word) {
        int32_t id = byte_ids_[b];
        if (id < 0) {
            if (!unk_id_) {
                continue;
            }
            id = *unk_id_;
        }
        const int32_t idx = static_cast<int32_t>(sym.size());
        sym.push_back({id, idx - 1, idx + 1, true});
    }
    if (sym.empty()) {
        return;
    }
    sym.back().next = -1;

    struct Merge {
        uint32_t rank;
        int32_t pos;
        int32_t new_id;
    };
    auto cmp = [](const Merge& x, const Merge& y) {
        if (x.rank != y.rank) {
            return x.rank > y.rank;
        }
        return x.pos > y.pos;
    };
    std::priority_queue<Merge, std::vector<Merge>, decltype(cmp)> queue(cmp);
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
        auto it = merges_.find(pair_key(sym[i].c, sym[i + 1].c));
        if (it != merges_.end()) {
            queue.push({it->second.first, static_cast<int32_t>(i), it->second.second});
        }
    }
    while (!queue.empty()) {
        const Merge top = queue.top();
        queue.pop();
        Symbol& cur = sym[static_cast<size_t>(top.pos)];
        if (!cur.alive || cur.next == -1) {
            continue;
        }
        const int32_t next_pos = cur.next;
        const Symbol right = sym[static_cast<size_t>(next_pos)];
        auto it = merges_.find(pair_key(cur.c, right.c));
        if (it == merges_.end() || it->second.second != top.new_id) {
            continue;
        }
        cur.c = top.new_id;
        cur.next = right.next;
        sym[static_cast<size_t>(next_pos)].alive = false;
        if (right.next >= 0) {
            sym[static_cast<size_t>(right.next)].prev = top.pos;
        }
        if (cur.prev >= 0) {
            auto p = merges_.find(pair_key(sym[static_cast<size_t>(cur.prev)].c, cur.c));
            if (p != merges_.end()) {
                queue.push({p->second.first, cur.prev, p->second.second});
            }
        }
        if (cur.next >= 0) {
            auto n = merges_.find(pair_key(cur.c, sym[static_cast<size_t>(cur.next)].c));
            if (n != merges_.end()) {
                queue.push({n->second.first, top.pos, n->second.second});
            }
        }
    }
    for (int32_t i = 0; i >= 0; i = sym[static_cast<size_t>(i)].next) {
        out.push_back(sym[static_cast<size_t>(i)].c);
    }
}

void Tokenizer::encode_text(const std::string& text, std::vector<int32_t>& out) const {
    std::string piece = text;
    if (piece.empty()) {
        return;
    }
    if (add_prefix_space_ && piece.front() != ' ') {
        piece.insert(piece.begin(), ' ');
    }
    if (!use_regex_) {
        encode_word(piece, out);
        return;
    }
    const std::u32string cps = unicode::decode(piece);
    for (const auto& [b, e] : byte_level_split(cps)) {
        encode_word(unicode::encode(std::u32string_view(cps).substr(b, e - b)), out);
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<int32_t> out;
    for (const Piece& raw : split_added(std::string(text), raw_trie_)) {
        if (raw.id >= 0) {
            out.push_back(raw.id);
            continue;
        }
        for (const Piece& p : split_added(normalize(raw.text), normalized_trie_)) {
            if (p.id >= 0) {
                out.push_back(p.id);
            } else {
                encode_text(p.text, out);
            }
        }
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, size_t max_length) const {
    std::vector<int32_t> ids = encode(text);
    if (ids.size() > max_length) {
        ids.resize(max_length);
    }
    return ids;
}

}
