add_executable(run_utest main.cpp
    utest.cpp
    test_arr.cpp
    test_culinear_ops.cpp
    test_rcwa.cpp
)

target_link_libraries(run_utest PRIVATE
    rcwa
    basics
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
    CUDA::cufft
)