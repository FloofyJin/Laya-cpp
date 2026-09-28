#pragma once

#include <string>
#include <string_view>

namespace laya::unicode {

std::u32string decode(std::string_view utf8);
std::string encode(std::u32string_view text);
void append(std::string& out, char32_t cp);

std::string nfc(std::string_view utf8);

bool is_letter(char32_t cp);
bool is_number(char32_t cp);
bool is_white_space(char32_t cp);
bool is_python_space(char32_t cp);
bool is_word(char32_t cp);

std::string python_strip(std::string_view utf8);

}
