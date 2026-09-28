#include "laya/unicode.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>

#include <utf8proc.h>

namespace laya::unicode {

std::u32string decode(std::string_view utf8) {
    std::u32string out;
    out.reserve(utf8.size());
    const auto* p = reinterpret_cast<const utf8proc_uint8_t*>(utf8.data());
    utf8proc_ssize_t left = static_cast<utf8proc_ssize_t>(utf8.size());
    while (left > 0) {
        utf8proc_int32_t cp = 0;
        utf8proc_ssize_t n = utf8proc_iterate(p, left, &cp);
        if (n <= 0) {
            throw std::runtime_error("invalid UTF-8 input");
        }
        out.push_back(static_cast<char32_t>(cp));
        p += n;
        left -= n;
    }
    return out;
}

void append(std::string& out, char32_t cp) {
    utf8proc_uint8_t buf[4];
    utf8proc_ssize_t n = utf8proc_encode_char(static_cast<utf8proc_int32_t>(cp), buf);
    out.append(reinterpret_cast<const char*>(buf), static_cast<size_t>(n));
}

std::string encode(std::u32string_view text) {
    std::string out;
    out.reserve(text.size());
    for (char32_t cp : text) {
        append(out, cp);
    }
    return out;
}

std::string nfc(std::string_view utf8) {
    if (utf8.empty()) {
        return {};
    }
    utf8proc_uint8_t* result = nullptr;
    utf8proc_ssize_t n = utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(utf8.data()),
                                      static_cast<utf8proc_ssize_t>(utf8.size()), &result,
                                      static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    if (n < 0) {
        throw std::runtime_error(std::string("NFC normalization failed: ") + utf8proc_errmsg(n));
    }
    std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> guard(result, &std::free);
    return std::string(reinterpret_cast<const char*>(result), static_cast<size_t>(n));
}

static utf8proc_category_t category(char32_t cp) {
    return utf8proc_category(static_cast<utf8proc_int32_t>(cp));
}

bool is_letter(char32_t cp) {
    switch (category(cp)) {
        case UTF8PROC_CATEGORY_LU:
        case UTF8PROC_CATEGORY_LL:
        case UTF8PROC_CATEGORY_LT:
        case UTF8PROC_CATEGORY_LM:
        case UTF8PROC_CATEGORY_LO:
            return true;
        default:
            return false;
    }
}

bool is_number(char32_t cp) {
    switch (category(cp)) {
        case UTF8PROC_CATEGORY_ND:
        case UTF8PROC_CATEGORY_NL:
        case UTF8PROC_CATEGORY_NO:
            return true;
        default:
            return false;
    }
}

bool is_white_space(char32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

bool is_python_space(char32_t cp) {
    return is_white_space(cp) || (cp >= 0x1C && cp <= 0x1F);
}

bool is_word(char32_t cp) {
    if (is_letter(cp)) {
        return true;
    }
    switch (category(cp)) {
        case UTF8PROC_CATEGORY_MN:
        case UTF8PROC_CATEGORY_MC:
        case UTF8PROC_CATEGORY_ME:
        case UTF8PROC_CATEGORY_ND:
        case UTF8PROC_CATEGORY_NL:
        case UTF8PROC_CATEGORY_PC:
            return true;
        default:
            return cp == 0x200C || cp == 0x200D;
    }
}

std::string python_strip(std::string_view utf8) {
    std::u32string cps = decode(utf8);
    size_t b = 0;
    size_t e = cps.size();
    while (b < e && is_python_space(cps[b])) {
        ++b;
    }
    while (e > b && is_python_space(cps[e - 1])) {
        --e;
    }
    return encode(std::u32string_view(cps).substr(b, e - b));
}

}
