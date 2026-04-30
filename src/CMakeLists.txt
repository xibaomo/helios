add_subdirectory(utest)
add_executable(helios main.cpp)
target_link_libraries(helios PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)