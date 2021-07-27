cmake_minimum_required(VERSION 3.18)
project(gtn-experimental LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(build_experimental SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  target_sources(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/prefix_scan.cpp
  )
  target_sources(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/converters.cpp
  )
  target_sources(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/parallel_compose.cpp
  )
  target_link_libraries(
    ${target}
    PRIVATE
    gtn
  )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/..
  )
endfunction(build_experimental)

function(build_experimental_gpu SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}_gpu")
  add_executable(${target} ${SRCFILE})
  target_sources(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/prefix_scan.cpp
  )
  target_sources(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/converters.cpp
  )
  target_sources(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/parallel_compose.cu
  )
  target_link_libraries(
    ${target}
    PRIVATE
    gtn
  )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/..
  )
endfunction(build_experimental_gpu)

build_experimental(test_parallel_compose.cpp)
build_experimental_gpu(test_parallel_compose.cpp)
