
add_subdirectory(basics)
add_subdirectory(rcwa)
add_subdirectory(utest)
add_executable(helios main.cpp)
target_link_libraries(helios PRIVATE
    basics
    rcwa
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::cufft
)
