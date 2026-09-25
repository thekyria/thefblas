# ARM Cortex-M (bare-metal) cross-compile toolchain for embedded systems. Usage:
# cmake -S . -B build-arm-bare
# -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/arm-none-eabi.cmake
#
# Prerequisites: - ARM bare-metal toolchain (e.g., arm-none-eabi-gcc,
# arm-none-eabi-g++) - Download from:
# https://developer.arm.com/downloads/-/gnu-rm or via package manager
#
# Suitable for: - ARM Cortex-M0, M0+, M3, M4, M7, M33, etc. - Bare-metal
# firmware development - RTOS and embedded systems without Linux

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER
    arm-none-eabi-gcc
    CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER
    arm-none-eabi-g++
    CACHE STRING "C++ compiler" FORCE)
set(CMAKE_ASM_COMPILER
    arm-none-eabi-gcc
    CACHE STRING "ASM compiler" FORCE)

set(CMAKE_OBJCOPY
    arm-none-eabi-objcopy
    CACHE FILEPATH "ObjCopy tool" FORCE)
set(CMAKE_OBJDUMP
    arm-none-eabi-objdump
    CACHE FILEPATH "ObjDump tool" FORCE)
set(CMAKE_SIZE
    arm-none-eabi-size
    CACHE FILEPATH "Size tool" FORCE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Bare-metal: disable shared libraries and set executable suffix
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# ARM Cortex-M base flags (adjust march, mthumb, mcpu for your target) Common
# options: -mcpu=cortex-m0       for Cortex-M0 -mcpu=cortex-m3       for
# Cortex-M3 -mcpu=cortex-m4       for Cortex-M4 -mcpu=cortex-m7       for
# Cortex-M7 -mcpu=cortex-m33      for Cortex-M33 Default is set to cortex-m4;
# override with -DTHEFBLAS_ARM_MCU if needed

set(THEFBLAS_ARM_MCU
    "cortex-m4"
    CACHE STRING "ARM Cortex-M variant (e.g. cortex-m4, cortex-m7)" FORCE)

set(CMAKE_C_FLAGS_INIT
    "-mcpu=${THEFBLAS_ARM_MCU} -mthumb -mfloat-abi=soft -Wall"
    CACHE STRING "C flags" FORCE)
set(CMAKE_CXX_FLAGS_INIT
    "-mcpu=${THEFBLAS_ARM_MCU} -mthumb -mfloat-abi=soft -Wall"
    CACHE STRING "CXX flags" FORCE)
