#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    std::cerr << "usage: laya-predict --model DIR [--chip cpu|hip|cuda] request FILE.json|-\n"
                 "       laya-predict --model DIR [--chip cpu|hip|cuda] requests FILE.jsonl\n";
    return 2;
}

}

int main(int argc, char** argv) {
    std::string model_dir;
    std::string chip = "cpu";
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (a == "--chip" && i + 1 < argc) {
            chip = argv[++i];
        } else {
            positional.push_back(a);
        }
    }
    if (model_dir.empty() || positional.size() != 2) {
        return usage_msg();
    }
    const std::string mode = positional[0];
    const std::string arg = positional[1];

    try {
        const laya::Agent agent(model_dir, laya::parse_chip(chip));
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
