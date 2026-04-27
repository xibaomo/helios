cmake_minimum_required(VERSION 3.10.0)
project(helios VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

add_subdirectory(src)

install(TARGETS helios
        DESTINATION bin)