cmake_minimum_required(VERSION 3.17)

set(
  GTN_EXPERIMENTAL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/experimental/prefix_scan.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/experimental/prefix_scan.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/experimental/converters.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/experimental/parallel_compose.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/experimental/parallel_compose.cu
  # ${CMAKE_CURRENT_SOURCE_DIR}/gtn/experimental/parallel_compose.cpp -- CPU
)


