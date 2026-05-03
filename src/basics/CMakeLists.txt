add_library(basics STATIC 
        culinear.cpp
        xmux.cu
        )
target_include_directories(basics PUBLIC ./)
