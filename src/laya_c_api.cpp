#include "laya.h"

#include <cstdlib>
#include <cstring>
#include <exception>

#include "laya/agent.hpp"
#include "laya/pyjson.hpp"

namespace {

void set_error(char* err_buf, size_t err_buf_len, const char* message) {
    if (err_buf == nullptr || err_buf_len == 0) {
        return;
    }
    std::strncpy(err_buf, message, err_buf_len - 1);
    err_buf[err_buf_len - 1] = '\0';
}

char* dup_string(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

}

struct laya_agent {
    laya::Agent impl;
};

extern "C" {

laya_agent* laya_agent_load(const char* model_dir, char* err_buf, size_t err_buf_len) {
    if (model_dir == nullptr) {
        set_error(err_buf, err_buf_len, "model_dir is null");
        return nullptr;
    }
    try {
        return new laya_agent{laya::Agent(model_dir)};
    } catch (const std::exception& e) {
        set_error(err_buf, err_buf_len, e.what());
        return nullptr;
    } catch (...) {
        set_error(err_buf, err_buf_len, "unknown error loading model");
        return nullptr;
    }
}

void laya_agent_free(laya_agent* agent) {
    delete agent;
}

char* laya_predict(laya_agent* agent, const char* request_json, char* err_buf, size_t err_buf_len) {
    if (agent == nullptr) {
        set_error(err_buf, err_buf_len, "agent is null");
        return nullptr;
    }
    if (request_json == nullptr) {
        set_error(err_buf, err_buf_len, "request_json is null");
        return nullptr;
    }
    try {
        const laya::py::Json request = laya::py::Json::parse(request_json);
        const laya::py::Json response = agent->impl.predict(request);
        return dup_string(response.dump());
    } catch (const std::exception& e) {
        set_error(err_buf, err_buf_len, e.what());
        return nullptr;
    } catch (...) {
        set_error(err_buf, err_buf_len, "unknown error running prediction");
        return nullptr;
    }
}

void laya_free_string(char* s) {
    std::free(s);
}

int laya_api_version(void) {
    return LAYA_API_VERSION;
}

}
