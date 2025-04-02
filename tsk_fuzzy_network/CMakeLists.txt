cmake_minimum_required(VERSION 3.28)

project(tsk-fuzzy-network)

set(CMAKE_CXX_STANDARD 23)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(LEARNING_SOURCE_FILES
    learning/hybrid_algorithm.cpp)

set(LAYERS_SOURCE_FILES
    layers/layer.cpp
    layers/multiple_layer.cpp
    layers/fuzzy_layer.cpp
    layers/role_multiple_layer.cpp
    layers/sum_layer.cpp
    tsk.cpp)
set(INCLUDE_DIRECTORY ../include)

add_library(TSK SHARED ${LAYERS_SOURCE_FILES} ${LEARNING_SOURCE_FILES})

target_include_directories(TSK
    PUBLIC
        ${INCLUDE_DIRECTORY}
)


