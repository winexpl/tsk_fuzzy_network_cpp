cmake_minimum_required(VERSION 3.28)

project(tsk-fuzzy-network LANGUAGES CXX CUDA)

# Общие настройки
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Настройки CUDA
set(CMAKE_CUDA_ARCHITECTURES "89")  # Ada Lovelace для RTX 4050
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Настройка CUDA runtime
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static" CACHE STRING "CUDA runtime library to use")

# Включение раздельной компиляции CUDA
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_LINK_LIBRARIES_PREFER_SHARED TRUE)
# Поиск необходимых пакетов
find_package(CUDAToolkit REQUIRED)

# Исполняемый файл
add_executable(test_cuda ./test_cuda.cpp ./test_cuda.cu)

# Настройки для целевого объекта
target_compile_features(test_cuda PRIVATE cuda_std_17)
target_include_directories(test_cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Альтернативный вариант линковки (если предыдущий не работает)
target_link_libraries(test_cuda PRIVATE 
    ${CUDAToolkit_LIBRARIES}
)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)  # Для правильной работы с CMAKE_CUDA_RUNTIME_LIBRARY
endif()