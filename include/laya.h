#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LAYA_API_VERSION 1

typedef struct laya_agent laya_agent;

laya_agent* laya_agent_load(const char* model_dir, const char* chip, char* err_buf, size_t err_buf_len);
void laya_agent_free(laya_agent* agent);

char* laya_predict(laya_agent* agent, const char* request_json, char* err_buf, size_t err_buf_len);
void laya_free_string(char* s);

int laya_api_version(void);

#ifdef __cplusplus
}
#endif
