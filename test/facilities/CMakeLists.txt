cmake_minimum_required(VERSION 3.29)

add_executable(test_var_type_dict test_var_type_dict.cpp)
target_link_libraries(test_var_type_dict PRIVATE Catch2::Catch2WithMain)