cmake_minimum_required(VERSION 3.29)

add_executable(test_policy policies/test_policy.cpp)
target_link_libraries(test_policy PRIVATE Catch2::Catch2WithMain)
add_executable(test_param_initializer model_rel/test_param_initializer.cpp)
target_link_libraries(test_param_initializer PRIVATE Catch2::Catch2WithMain)


add_subdirectory(facilities)
add_subdirectory(data)
