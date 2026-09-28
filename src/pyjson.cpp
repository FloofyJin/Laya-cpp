#include "laya/pyjson.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace laya::py {

std::string float_repr(double value) {
    if (std::isnan(value)) {
        return "nan";
    }
    if (std::isinf(value)) {
        return value > 0 ? "inf" : "-inf";
    }
    if (value == 0.0) {
        return std::signbit(value) ? "-0.0" : "0.0";
    }
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
    std::string sci(buf, res.ptr);
    std::string sign;
    if (sci[0] == '-') {
        sign = "-";
        sci.erase(0, 1);
    }
    const size_t e = sci.find('e');
    std::string mantissa = sci.substr(0, e);
    const int exp10 = std::stoi(sci.substr(e + 1));
    std::string digits;
    for (char c : mantissa) {
        if (c != '.') {
            digits.push_back(c);
        }
    }
    const int decpt = exp10 + 1;
    const int ndigits = static_cast<int>(digits.size());
    std::string out;
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) {
            out = "0." + std::string(static_cast<size_t>(-decpt), '0') + digits;
        } else if (decpt >= ndigits) {
            out = digits + std::string(static_cast<size_t>(decpt - ndigits), '0') + ".0";
        } else {
            out = digits.substr(0, static_cast<size_t>(decpt)) + "." + digits.substr(static_cast<size_t>(decpt));
        }
    } else {
        out = digits.substr(0, 1);
        if (ndigits > 1) {
            out += "." + digits.substr(1);
        }
        const int x = decpt - 1;
        char eb[16];
        std::snprintf(eb, sizeof(eb), "e%c%02d", x < 0 ? '-' : '+', x < 0 ? -x : x);
        out += eb;
    }
    return sign + out;
}

static void dump_string(const std::string& s, std::string& out) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
}

static void dump(const Json& v, std::string& out) {
    switch (v.type()) {
        case Json::value_t::null:
            out += "null";
            break;
        case Json::value_t::boolean:
            out += v.get<bool>() ? "true" : "false";
            break;
        case Json::value_t::number_integer:
            out += std::to_string(v.get<int64_t>());
            break;
        case Json::value_t::number_unsigned:
            out += std::to_string(v.get<uint64_t>());
            break;
        case Json::value_t::number_float: {
            const double d = v.get<double>();
            if (std::isnan(d)) {
                out += "NaN";
            } else if (std::isinf(d)) {
                out += d > 0 ? "Infinity" : "-Infinity";
            } else {
                out += float_repr(d);
            }
            break;
        }
        case Json::value_t::string:
            dump_string(v.get_ref<const std::string&>(), out);
            break;
        case Json::value_t::array: {
            out.push_back('[');
            bool first = true;
            for (const Json& e : v) {
                if (!first) {
                    out += ", ";
                }
                first = false;
                dump(e, out);
            }
            out.push_back(']');
            break;
        }
        case Json::value_t::object: {
            out.push_back('{');
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) {
                    out += ", ";
                }
                first = false;
                dump_string(it.key(), out);
                out += ": ";
                dump(it.value(), out);
            }
            out.push_back('}');
            break;
        }
        default:
            throw std::runtime_error("unsupported JSON value");
    }
}

std::string dumps(const Json& value) {
    std::string out;
    dump(value, out);
    return out;
}

std::string str(const Json& value) {
    switch (value.type()) {
        case Json::value_t::null:
            return "None";
        case Json::value_t::boolean:
            return value.get<bool>() ? "True" : "False";
        case Json::value_t::number_float:
            return float_repr(value.get<double>());
        case Json::value_t::string:
            return value.get<std::string>();
        default:
            return dumps(value);
    }
}

}
