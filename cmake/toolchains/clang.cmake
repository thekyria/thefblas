# Clang toolchain file for local/native builds. Usage: cmake -S . -B build-clang
# -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/clang.cmake

set(CMAKE_C_COMPILER
    clang
    CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER
    clang++
    CACHE STRING "C++ compiler" FORCE)

find_program(THEFBLAS_LLVM_AR NAMES llvm-ar)
if(THEFBLAS_LLVM_AR)
  set(CMAKE_AR
      "${THEFBLAS_LLVM_AR}"
      CACHE FILEPATH "Archiver" FORCE)
  set(CMAKE_C_COMPILER_AR
      "${THEFBLAS_LLVM_AR}"
      CACHE FILEPATH "C compiler archiver" FORCE)
  set(CMAKE_CXX_COMPILER_AR
      "${THEFBLAS_LLVM_AR}"
      CACHE FILEPATH "CXX compiler archiver" FORCE)
endif()

find_program(THEFBLAS_LLVM_RANLIB NAMES llvm-ranlib)
if(THEFBLAS_LLVM_RANLIB)
  set(CMAKE_RANLIB
      "${THEFBLAS_LLVM_RANLIB}"
      CACHE FILEPATH "Ranlib" FORCE)
  set(CMAKE_C_COMPILER_RANLIB
      "${THEFBLAS_LLVM_RANLIB}"
      CACHE FILEPATH "C compiler ranlib" FORCE)
  set(CMAKE_CXX_COMPILER_RANLIB
      "${THEFBLAS_LLVM_RANLIB}"
      CACHE FILEPATH "CXX compiler ranlib" FORCE)
endif()

find_program(THEFBLAS_LLVM_NM NAMES llvm-nm)
if(THEFBLAS_LLVM_NM)
  set(CMAKE_NM
      "${THEFBLAS_LLVM_NM}"
      CACHE FILEPATH "Symbol dumper" FORCE)
endif()
