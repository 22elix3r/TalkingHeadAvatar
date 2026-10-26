cmake_minimum_required(VERSION 3.20)
project(tha_qwen_bridge LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT QWEN_SOURCE OR NOT QWEN_BUILD)
  message(FATAL_ERROR "Explicit pinned QWEN_SOURCE and locally built QWEN_BUILD are required")
endif()
find_package(Threads REQUIRED)
add_executable(tha_qwen_worker worker.cpp)
target_include_directories(tha_qwen_worker PRIVATE "${QWEN_SOURCE}/engine/src" "${QWEN_SOURCE}/engine/ggml/include")
target_link_libraries(tha_qwen_worker PRIVATE
  -Wl,--start-group
  "${QWEN_BUILD}/engine/libtts_engine.a"
  "${QWEN_BUILD}/engine/libtts_transformer.a"
  "${QWEN_BUILD}/engine/libaudio_tokenizer_encoder.a"
  "${QWEN_BUILD}/engine/libaudio_tokenizer_decoder.a"
  "${QWEN_BUILD}/engine/libtext_tokenizer.a"
  -Wl,--end-group
  "${QWEN_BUILD}/engine/ggml/src/libggml.so"
  "${QWEN_BUILD}/engine/ggml/src/libggml-base.so"
  "${QWEN_BUILD}/engine/ggml/src/libggml-cpu.so"
  "${QWEN_BUILD}/engine/ggml/src/ggml-cuda/libggml-cuda.so"
  Threads::Threads)
