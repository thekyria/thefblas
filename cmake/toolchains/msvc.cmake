# MSVC toolchain file for Windows builds. Usage: cmake -S . -B build-msvc
# -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/msvc.cmake

if(NOT WIN32)
  message(
    FATAL_ERROR
      "MSVC toolchain requires a Windows host with Visual Studio Build Tools.")
endif()

set(CMAKE_C_COMPILER
    cl
    CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER
    cl
    CACHE STRING "C++ compiler" FORCE)
