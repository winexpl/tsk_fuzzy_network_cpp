cmake_minimum_required(VERSION 3.28)

project(final-projects)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INCLUDE_DIRECTORY ${CMAKE_HOME_DIRECTORY}/../include)

set(TSK_LIB_PATH ${CMAKE_HOME_DIRECTORY}/../lib/libTSK.so)

add_executable(customer test_customer.cpp)

add_executable(irises test_irises.cpp)

add_executable(diabet test_diabet.cpp)


target_include_directories(diabet
        PUBLIC
                ${INCLUDE_DIRECTORY}
)

target_link_libraries(diabet PUBLIC ${TSK_LIB_PATH} )

find_package(nlohmann_json REQUIRED)
target_link_libraries(diabet PRIVATE nlohmann_json::nlohmann_json)

find_package(TBB REQUIRED)
target_link_libraries(diabet PRIVATE TBB::tbb)

find_package(Boost REQUIRED COMPONENTS log log_setup system thread serialization)
target_link_libraries(diabet PUBLIC Boost::log Boost::log_setup Boost::system Boost::thread Boost::serialization gtest_main gtest)



target_include_directories(customer
        PUBLIC
                ${INCLUDE_DIRECTORY}
)

target_link_libraries(customer PUBLIC ${TSK_LIB_PATH} )

find_package(nlohmann_json REQUIRED)
target_link_libraries(customer PRIVATE nlohmann_json::nlohmann_json)

find_package(TBB REQUIRED)
target_link_libraries(customer PRIVATE TBB::tbb)

find_package(Boost REQUIRED COMPONENTS log log_setup system thread serialization)
target_link_libraries(customer PUBLIC Boost::log Boost::log_setup Boost::system Boost::thread Boost::serialization gtest_main gtest)


target_include_directories(irises
        PUBLIC
                ${INCLUDE_DIRECTORY}
)

target_link_libraries(irises PUBLIC ${TSK_LIB_PATH} )

find_package(nlohmann_json REQUIRED)
target_link_libraries(irises PRIVATE nlohmann_json::nlohmann_json)

find_package(TBB REQUIRED)
target_link_libraries(irises PRIVATE TBB::tbb)

find_package(Boost REQUIRED COMPONENTS log log_setup system thread serialization)
target_link_libraries(irises PUBLIC Boost::log Boost::log_setup Boost::system Boost::thread Boost::serialization gtest_main gtest)