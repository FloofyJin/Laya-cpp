#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace laya::py {

using Json = nlohmann::ordered_json;

std::string float_repr(double value);
std::string dumps(const Json& value);
std::string str(const Json& value);

}
