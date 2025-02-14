cmake_minimum_required(VERSION 3.28)

project(final-projects)

set(CMAKE_CXX_STANDARD 23)
set(INCLUDE_DIRECTORY ${CMAKE_HOME_DIRECTORY}/../include)

set(TSK_LIB_PATH ${CMAKE_HOME_DIRECTORY}/../lib/libTSK.so)

add_executable(main main.cpp)

target_include_directories(main
        PUBLIC
                ${INCLUDE_DIRECTORY}
)

target_link_libraries(main ${TSK_LIB_PATH})