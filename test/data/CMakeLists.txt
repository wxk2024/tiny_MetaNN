cmake_minimum_required(VERSION 3.29)

add_executable(test_scalar test_scalar.cpp)
target_link_libraries(test_scalar PRIVATE Catch2::Catch2WithMain)
add_executable(test_matrix test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE Catch2::Catch2WithMain)
#add_executable(test_batch test_batch.cpp)
#target_link_libraries(test_batch PRIVATE Catch2::Catch2WithMain)
