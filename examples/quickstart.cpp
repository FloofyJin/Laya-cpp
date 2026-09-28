#include <cstdio>

#include "laya.h"

int main() {
    char err[256];
    laya_agent* agent = laya_agent_load("models/laya", err, sizeof(err));
    if (agent == nullptr) {
        std::fprintf(stderr, "load failed: %s\n", err);
        return 1;
    }

    const char* request = R"({
        "state": "This product broke after one day, I want my money back.",
        "questions": {
            "wants_refund": {"type": "noul", "instructions": "Does the customer want a refund?"}
        }
    })";

    char* response = laya_predict(agent, request, err, sizeof(err));
    if (response == nullptr) {
        std::fprintf(stderr, "predict failed: %s\n", err);
        laya_agent_free(agent);
        return 1;
    }

    std::printf("%s\n", response);

    laya_free_string(response);
    laya_agent_free(agent);
    return 0;
}
