add_library(basics STATIC 
        culinear.cpp
        xmux.cu
        utils.cu
        )
target_include_directories(basics PUBLIC ./)
