#pragma once

#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace laya::py {

using Json = nlohmann::ordered_json;

std::string float_repr(double value);
std::string dumps(const Json& value);
std::string str(const Json& value);
std::string nonfinite_to_null(std::string_view text);

}
