cmake_minimum_required(VERSION 3.28)

project(tsk-fuzzy-network LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_ARCHITECTURES "89")

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(LEARNING_SOURCE_FILES
    learning/hybrid_algorithm.cpp
    learning/hybrid_algorithm.cu
    learning/c_means.cpp)

set(LAYERS_SOURCE_FILES
    layers/layer.cpp
    layers/multiple_layer.cpp
    layers/fuzzy_layer.cpp
    layers/role_multiple_layer.cpp
    layers/sum_layer.cpp
    tsk.cpp)

set(INCLUDE_DIRECTORY ../include)

enable_language(CUDA)

add_library(TSK SHARED ${LAYERS_SOURCE_FILES} ${LEARNING_SOURCE_FILES} dataset.cpp logger.cpp fuzzy_api.cpp)

target_include_directories(TSK
    PUBLIC
    ${INCLUDE_DIRECTORY}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(TSK 
    ${CUDA_LIBRARIES} 
    ${CUDA_cudart_static_LIBRARY}
)

# Set CUDA architecture (adjust for your GPU)
set(CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

find_package(Boost REQUIRED COMPONENTS log log_setup system thread serialization)
target_link_libraries(TSK PUBLIC 
    Boost::log 
    Boost::log_setup 
    Boost::system 
    Boost::thread 
    Boost::serialization
)

find_package(OpenMP REQUIRED)
target_link_libraries(TSK PRIVATE OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")