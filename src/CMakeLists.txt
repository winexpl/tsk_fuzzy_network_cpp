cmake_minimum_required(VERSION 3.28)

project(final-projects)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INCLUDE_DIRECTORY ${CMAKE_HOME_DIRECTORY}/../include)

set(TSK_LIB_PATH ${CMAKE_HOME_DIRECTORY}/../lib/libTSK.so)

find_package(Boost REQUIRED COMPONENTS log log_setup system thread serialization)
find_package(nlohmann_json REQUIRED)
find_package(TBB REQUIRED)

add_executable(tic_tac_toc test_tic-tac-toc.cpp)

add_executable(main main.cpp)

target_include_directories(main
        PUBLIC
                ${INCLUDE_DIRECTORY}
)

target_link_libraries(main PUBLIC ${TSK_LIB_PATH} )
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(main PRIVATE TBB::tbb)
target_link_libraries(main PUBLIC Boost::log Boost::log_setup Boost::system Boost::thread Boost::serialization gtest_main gtest)

target_include_directories(tic_tac_toc
        PUBLIC
                ${INCLUDE_DIRECTORY}
)

target_link_libraries(tic_tac_toc PUBLIC ${TSK_LIB_PATH} )
target_link_libraries(tic_tac_toc PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(tic_tac_toc PRIVATE TBB::tbb)
target_link_libraries(tic_tac_toc PUBLIC Boost::log Boost::log_setup Boost::system Boost::thread Boost::serialization gtest_main gtest)
