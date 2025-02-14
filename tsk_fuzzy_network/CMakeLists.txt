cmake_minimum_required(VERSION 3.28)

project(tsk-fuzzy-network)

set(CMAKE_CXX_STANDARD 23)

set(LAYERS_SOURCE_FILES 
        layers/layer.cpp
        layers/multiple_layer.cpp
        layers/fuzzy_layer.cpp
        layers/role_multiple_layer.cpp
        layers/sum_layer.cpp
        tsk.cpp)
set(INCLUDE_DIRECTORY ../include)

add_library(TSK SHARED ${LAYERS_SOURCE_FILES})

target_include_directories(TSK
        PUBLIC
                ${INCLUDE_DIRECTORY}
)


