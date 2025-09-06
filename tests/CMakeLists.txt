cmake_minimum_required(VERSION 3.28)

project(final-projects)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INCLUDE_DIRECTORY ${CMAKE_HOME_DIRECTORY}/../include)

set(TSK_LIB_PATH ${CMAKE_HOME_DIRECTORY}/../lib/libTSK.so)

find_package(Boost REQUIRED COMPONENTS log log_setup system thread serialization)
find_package(nlohmann_json REQUIRED)
find_package(TBB REQUIRED)

add_executable(test_role test_role.cpp)
add_executable(test_noisy test_noisy.cpp)

target_include_directories(test_role
        PUBLIC
                ${INCLUDE_DIRECTORY}
)

target_link_libraries(test_role PUBLIC ${TSK_LIB_PATH} )
target_link_libraries(test_role PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_role PRIVATE TBB::tbb)
target_link_libraries(test_role PUBLIC Boost::log Boost::log_setup Boost::system Boost::thread Boost::serialization gtest_main gtest)

target_include_directories(test_noisy
        PUBLIC
                ${INCLUDE_DIRECTORY}
)

target_link_libraries(test_noisy PUBLIC ${TSK_LIB_PATH} )
target_link_libraries(test_noisy PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(test_noisy PRIVATE TBB::tbb)
target_link_libraries(test_noisy PUBLIC Boost::log Boost::log_setup Boost::system Boost::thread Boost::serialization gtest_main gtest)
