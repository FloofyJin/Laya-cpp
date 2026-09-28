#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "laya.h"

namespace {

std::string slurp(const std::string& path) {
    if (path == "-") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        return ss.str();
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

int usage_msg() {
    std::cerr << "usage: laya-bit --model DIR request FILE.json|-\n"
                 "       laya-bit --model DIR requests FILE.jsonl\n";
    return 2;
}

}

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--model") {
        return usage_msg();
    }
    const std::string model_dir = argv[2];
    const std::string mode = argv[3];
    const std::string arg = argv[4];

    char err_buf[512];
    laya_agent* agent = laya_agent_load(model_dir.c_str(), err_buf, sizeof(err_buf));
    if (agent == nullptr) {
        std::cerr << "laya-bit: " << err_buf << "\n";
        return 1;
    }

    int status = 0;
    try {
        if (mode == "request") {
            const std::string request = slurp(arg);
            char* response = laya_predict(agent, request.c_str(), err_buf, sizeof(err_buf));
            if (response == nullptr) {
                std::cerr << "laya-bit: " << err_buf << "\n";
                status = 1;
            } else {
                std::cout << response << "\n";
                laya_free_string(response);
            }
        } else if (mode == "requests") {
            std::istringstream lines(slurp(arg));
            std::string line;
            while (std::getline(lines, line)) {
                if (line.empty()) {
                    continue;
                }
                char* response = laya_predict(agent, line.c_str(), err_buf, sizeof(err_buf));
                if (response == nullptr) {
                    std::cout << "{\"error\":\"" << json_escape(err_buf) << "\"}\n";
                } else {
                    std::cout << response << "\n";
                    laya_free_string(response);
                }
            }
        } else {
            status = usage_msg();
        }
    } catch (const std::exception& e) {
        std::cerr << "laya-bit: " << e.what() << "\n";
        status = 1;
    }

    laya_agent_free(agent);
    return status;
}
