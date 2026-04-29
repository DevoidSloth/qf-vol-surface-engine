cmake_minimum_required(VERSION 3.20)
project(vse VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VSE_BUILD_TESTS  "Build the C++ property/unit test binary" ON)
option(VSE_BUILD_BENCH  "Build the benchmark binary"              ON)
option(VSE_BUILD_PYTHON "Build the pybind11 extension module"     ON)

# ---------------------------------------------------------------------------
# Core: header-only.
#
# -O2 rather than -O3 because every benchmark in benchmarks/ is quoted at -O2
# and a headline number you cannot reproduce from the documented flags is not a
# number. Fast-math is explicitly off: several routines here depend on IEEE
# semantics (signed zero in the PSOR projection, NaN propagation in the
# calibration guards, and the exactness of the Mills-ratio cancellation).
# ---------------------------------------------------------------------------
add_library(vse INTERFACE)
target_include_directories(vse INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>)

if(MSVC)
  target_compile_options(vse INTERFACE /O2 /fp:precise /W4 /permissive-)
else()
  target_compile_options(vse INTERFACE -O2 -fno-fast-math -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(VSE_BUILD_TESTS)
  file(GLOB VSE_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/*.cpp)
  add_executable(vse_tests ${VSE_TEST_SOURCES})
  target_link_libraries(vse_tests PRIVATE vse)
  target_include_directories(vse_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests)
  enable_testing()
  add_test(NAME vse_tests COMMAND vse_tests)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(VSE_BUILD_BENCH)
  file(GLOB VSE_BENCH_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bench/*.cpp)
  if(VSE_BENCH_SOURCES)
    add_executable(vse_bench ${VSE_BENCH_SOURCES})
    target_link_libraries(vse_bench PRIVATE vse)
    target_include_directories(vse_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bench)
  endif()
endif()

# ---------------------------------------------------------------------------
# Python bindings
#
# The MinGW branch exists because the reference dev box builds with the msys2
# ucrt64 gcc against a python.org (MSVC-built) CPython. Two things break there
# and both are silent until import time:
#   * gcc links libstdc++/libwinpthread dynamically, and Windows does not search
#     PATH for an extension module's dependencies, so the .pyd fails to load
#     with a bare "DLL load failed". -static resolves it.
#   * MinGW ld will not read the MSVC import library python3xx.lib, so we link
#     the DLL directly, which it does handle.
# ---------------------------------------------------------------------------
if(VSE_BUILD_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module QUIET)
  if(NOT Python_FOUND)
    find_package(Python 3.9 COMPONENTS Interpreter Development QUIET)
  endif()

  if(Python_FOUND)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE VSE_PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE VSE_PYBIND11_RC ERROR_QUIET)

    if(VSE_PYBIND11_RC EQUAL 0)
      list(APPEND CMAKE_PREFIX_PATH "${VSE_PYBIND11_DIR}")
      find_package(pybind11 CONFIG REQUIRED)

      pybind11_add_module(_vse MODULE ${CMAKE_CURRENT_SOURCE_DIR}/bindings/module.cpp)
      target_link_libraries(_vse PRIVATE vse)
      set_target_properties(_vse PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/vsepy")

      if(MINGW)
        execute_process(
          COMMAND "${Python_EXECUTABLE}" -c
                  "import sys,os;print(os.path.join(sys.prefix,'python%d%d.dll'%sys.version_info[:2]))"
          OUTPUT_VARIABLE VSE_PYTHON_DLL OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(EXISTS "${VSE_PYTHON_DLL}")
          target_link_options(_vse PRIVATE -static)
          target_link_libraries(_vse PRIVATE "${VSE_PYTHON_DLL}")
        else()
          message(WARNING "MinGW build: could not locate python DLL at ${VSE_PYTHON_DLL}; "
                          "the module will probably fail to import.")
        endif()
      endif()
    else()
      message(STATUS "pybind11 not importable from ${Python_EXECUTABLE}; skipping _vse. "
                     "Install it with: ${Python_EXECUTABLE} -m pip install pybind11")
    endif()
  else()
    message(STATUS "Python development headers not found; skipping _vse.")
  endif()
endif()
