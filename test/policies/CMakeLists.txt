cmake_minimum_required(VERSION 3.29)

add_executable(test_policy_selector test_policy_selector.cpp)
target_link_libraries(test_policy_selector PRIVATE Catch2::Catch2WithMain)
