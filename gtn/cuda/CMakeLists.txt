cmake_minimum_required(VERSION 3.17)

set(
  GTN_NO_CUDA_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/cuda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/nocuda.cpp
)

set(
  GTN_CUDA_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/creations.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/creations.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/compose.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/cuda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/cuda.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/functions.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/functions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gtn/cuda/shortest.cu
)
