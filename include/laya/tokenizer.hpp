#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace laya {

struct AddedToken {
    std::string content;
    int32_t id = -1;
    bool single_word = false;
    bool lstrip = false;
    bool rstrip = false;
    bool normalized = false;
    bool special = false;
};

class Tokenizer {
public:
    static Tokenizer from_directory(const std::string& dir);
    static Tokenizer from_files(const std::string& tokenizer_json, const std::string& tokenizer_config_json);

    std::vector<int32_t> encode(std::string_view text) const;
    std::vector<int32_t> encode(std::string_view text, size_t max_length) const;

    std::optional<int32_t> token_to_id(const std::string& token) const;
    std::string id_to_token(int32_t id) const;

    int32_t cls_id() const { return cls_id_; }
    int32_t sep_id() const { return sep_id_; }
    int32_t pad_id() const { return pad_id_; }
    int32_t mask_id() const { return mask_id_; }
    const std::string& mask_token() const { return mask_token_; }

private:
    struct TrieNode {
        std::vector<std::pair<unsigned char, int32_t>> next;
        int32_t token = -1;
    };
    struct Trie {
        std::vector<TrieNode> nodes{TrieNode{}};
        void insert(const std::string& key, int32_t token);
        std::pair<size_t, int32_t> longest_at(const std::string& text, size_t pos) const;
    };
    struct Piece {
        std::string text;
        int32_t id;
    };

    Tokenizer() = default;
    void load_tokenizer_json(const std::string& path);
    void load_special_tokens(const std::string& path);

    std::vector<Piece> split_added(const std::string& sentence, const Trie& trie) const;
    void encode_text(const std::string& text, std::vector<int32_t>& out) const;
    void encode_word(const std::string& word, std::vector<int32_t>& out) const;
    std::string normalize(const std::string& text) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> vocab_r_;
    std::unordered_map<uint64_t, std::pair<uint32_t, int32_t>> merges_;
    std::array<std::string, 256> byte_chars_{};
    std::array<int32_t, 256> byte_ids_{};

    std::vector<AddedToken> added_;
    std::unordered_map<std::string, int32_t> added_ids_;
    Trie raw_trie_;
    Trie normalized_trie_;

    bool nfc_ = false;
    bool add_prefix_space_ = false;
    bool use_regex_ = true;
    bool ignore_merges_ = false;
    std::optional<int32_t> unk_id_;

    int32_t cls_id_ = -1;
    int32_t sep_id_ = -1;
    int32_t pad_id_ = -1;
    int32_t mask_id_ = -1;
    std::string mask_token_;
};

}
