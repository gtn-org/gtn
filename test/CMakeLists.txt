cmake_minimum_required(VERSION 3.5.1)

function(find_python_exec DST)
  if (WIN32)
    set(CMD where.exe python3)
  elseif (UNIX OR APPLE)
    set(CMD which python3)
  endif()
  execute_process(
    COMMAND ${CMD}
    OUTPUT_VARIABLE OUT
    RESULT_VARIABLE RES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  set(${DST} ${OUT} PARENT_SCOPE)
endfunction()

if (GTN_BUILD_CUDA)
  set(
    GTN_CUDA_TEST_SOURCES
    ${PROJECT_SOURCE_DIR}/test/cuda_test.cpp
    ${PROJECT_SOURCE_DIR}/test/cuda_creations_test.cpp
    ${PROJECT_SOURCE_DIR}/test/cuda_functions_test.cpp
    ${PROJECT_SOURCE_DIR}/test/cuda_utils_test.cpp
  )
else()
  set(
    GTN_CUDA_TEST_SOURCES
    ${PROJECT_SOURCE_DIR}/test/no_cuda_test.cpp
  )
endif()

set(
  GTN_TEST_SOURCES
  ${PROJECT_SOURCE_DIR}/test/common.h
  ${PROJECT_SOURCE_DIR}/test/autograd_test.cpp
  ${PROJECT_SOURCE_DIR}/test/creations_test.cpp
  ${PROJECT_SOURCE_DIR}/test/criterion_test.cpp
  ${PROJECT_SOURCE_DIR}/test/device_test.cpp
  ${PROJECT_SOURCE_DIR}/test/functions_test.cpp
  ${PROJECT_SOURCE_DIR}/test/graph_test.cpp
  ${PROJECT_SOURCE_DIR}/test/hd_span_test.cpp
  ${PROJECT_SOURCE_DIR}/test/parallel_test.cpp
  ${PROJECT_SOURCE_DIR}/test/rand_test.cpp
  ${PROJECT_SOURCE_DIR}/test/utils_test.cpp
  ${GTN_CUDA_TEST_SOURCES}
)

add_executable(tests ${PROJECT_SOURCE_DIR}/test/tests.cpp)
target_sources(tests PRIVATE ${GTN_TEST_SOURCES})
target_link_libraries(tests PRIVATE gtn)
target_include_directories(tests PRIVATE ${PROJECT_SOURCE_DIR}/..)
add_test(tests tests)

if (GTN_BUILD_PYTHON_BINDINGS)
  find_python_exec(PYTHON_EXEC_PATH)
  add_test(
    NAME PythonBindingsTest
    COMMAND ${PYTHON_EXEC_PATH} -m unittest discover -v ${PROJECT_SOURCE_DIR}/bindings/python/test
  )
endif()
