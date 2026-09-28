#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "laya/agent.hpp"
#include "laya/pyjson.hpp"

namespace {

using laya::py::Json;

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

int usage_msg() {
    std::cerr << "usage: laya-predict --model DIR request FILE.json|-\n"
                 "       laya-predict --model DIR requests FILE.jsonl\n";
    return 2;
}

}

int main(int argc, char** argv) {
    if (argc != 5 || std::string(argv[1]) != "--model") {
        return usage_msg();
    }
    const std::string mode = argv[3];
    const std::string arg = argv[4];
    try {
        const laya::Agent agent(argv[2]);
        if (mode == "request") {
            std::cout << agent.predict(Json::parse(slurp(arg))).dump() << "\n";
        } else if (mode == "requests") {
            std::istringstream lines(slurp(arg));
            std::string line;
            while (std::getline(lines, line)) {
                if (line.empty()) {
                    continue;
                }
                Json out;
                try {
                    out = agent.predict(Json::parse(line));
                } catch (const std::exception& e) {
                    out = Json{{"error", e.what()}};
                }
                std::cout << out.dump() << "\n";
            }
        } else {
            return usage_msg();
        }
    } catch (const std::exception& e) {
        std::cerr << "laya-predict: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
