cmake_minimum_required(VERSION 3.20)
project(laya_core_linux LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LAYA_FETCH_DEPS "Download utf8proc and nlohmann_json when not installed" ON)

include(FetchContent)

find_package(nlohmann_json 3.10 QUIET)
if(NOT nlohmann_json_FOUND)
  if(NOT LAYA_FETCH_DEPS)
    message(FATAL_ERROR "nlohmann_json not found (apt install nlohmann-json3-dev)")
  endif()
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(UTF8PROC QUIET IMPORTED_TARGET libutf8proc)
endif()
add_library(laya_utf8proc INTERFACE)
if(UTF8PROC_FOUND)
  target_link_libraries(laya_utf8proc INTERFACE PkgConfig::UTF8PROC)
else()
  if(NOT LAYA_FETCH_DEPS)
    message(FATAL_ERROR "utf8proc not found (apt install libutf8proc-dev)")
  endif()
  set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
  set(UTF8PROC_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  FetchContent_Declare(utf8proc
    URL https://github.com/JuliaStrings/utf8proc/archive/refs/tags/v2.9.0.tar.gz)
  FetchContent_MakeAvailable(utf8proc)
  target_link_libraries(laya_utf8proc INTERFACE utf8proc)
endif()

add_library(laya_core
  src/unicode.cpp
  src/tokenizer.cpp
  src/pyjson.cpp
  src/sequence.cpp
  src/safetensors.cpp
  src/nn.cpp
  src/modernbert.cpp
  src/decision.cpp
)
target_include_directories(laya_core PUBLIC include)
target_link_libraries(laya_core PUBLIC nlohmann_json::nlohmann_json PRIVATE laya_utf8proc)
target_compile_options(laya_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(laya-tokenize tools/laya_tokenize.cpp)
target_link_libraries(laya-tokenize PRIVATE laya_core)
target_compile_options(laya-tokenize PRIVATE -Wall -Wextra -Wpedantic)

add_executable(laya-dump-tensor tools/laya_dump_tensor.cpp)
target_link_libraries(laya-dump-tensor PRIVATE laya_core)
target_compile_options(laya-dump-tensor PRIVATE -Wall -Wextra -Wpedantic)

add_executable(laya-encoder-probe tools/laya_encoder_probe.cpp)
target_link_libraries(laya-encoder-probe PRIVATE laya_core)
target_compile_options(laya-encoder-probe PRIVATE -Wall -Wextra -Wpedantic)

add_executable(laya-predict tools/laya_predict.cpp)
target_link_libraries(laya-predict PRIVATE laya_core)
target_compile_options(laya-predict PRIVATE -Wall -Wextra -Wpedantic)
