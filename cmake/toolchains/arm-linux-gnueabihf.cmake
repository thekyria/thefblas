# ARM 32-bit (hard-float EABI) cross-compile toolchain for embedded Linux.
# Usage: cmake -S . -B build-arm
# -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-linux-gnueabihf.cmake
#
# Prerequisites: - ARM cross-compiler toolchain (e.g., arm-linux-gnueabihf-g++,
# arm-linux-gnueabihf-gcc) - ARM sysroot (optional, set THEFBLAS_ARM_SYSROOT if
# needed)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER
    arm-linux-gnueabihf-gcc
    CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER
    arm-linux-gnueabihf-g++
    CACHE STRING "C++ compiler" FORCE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Optional: set sysroot if cross-compiling with a non-standard sysroot
# set(THEFBLAS_ARM_SYSROOT "/path/to/arm/sysroot") if(THEFBLAS_ARM_SYSROOT)
# set(CMAKE_SYSROOT "${THEFBLAS_ARM_SYSROOT}" CACHE PATH "ARM sysroot" FORCE)
# add_compile_options(-isysroot "${CMAKE_SYSROOT}") endif()

# ARM-specific flags for hard-float EABI
set(CMAKE_C_FLAGS_INIT
    "-march=armv7-a -mfpu=neon -mfloat-abi=hard"
    CACHE STRING "C flags" FORCE)
set(CMAKE_CXX_FLAGS_INIT
    "-march=armv7-a -mfpu=neon -mfloat-abi=hard"
    CACHE STRING "CXX flags" FORCE)
