cmake_minimum_required(VERSION 3.18...4.0)
project(cellsight LANGUAGES CXX)

include (FetchContent)

#Eigen lib fetch 
FetchContent_Declare(
Eigen 
GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)

# Stops Eigen from building its own tests, docs, and extra config files.
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
# downloading eigen and making it available for file
FetchContent_MakeAvailable (Eigen)

#now time fr catch2 ... creating a switch so for normal build from pip we can skip - defaaultOFF
option(CELLSIGHT_BUILD_TESTS "Build tests" OFF)
# will turn ON then only we use catch2 for testing otherwise no need to install for user -_-
if(CELLSIGHT_BUILD_TESTS)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
    GIT_SHALLOW TRUE
)
#downlaoding catch2
FetchContent_MakeAvailable(Catch2)
enable_testing() #starting cmake tester
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# core library with Lut2D
add_library(cellsight_core STATIC cpp/src/lut2d.cpp cpp/src/ecm.cpp cpp/src/simulate.cpp)
target_include_directories(cellsight_core PUBLIC cpp/include)
target_link_libraries(cellsight_core PUBLIC Eigen3::Eigen)
target_compile_features(cellsight_core PUBLIC cxx_std_17)
set_target_properties(cellsight_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(_core bindings/module.cpp)
target_link_libraries(_core PRIVATE cellsight_core)

install(TARGETS _core LIBRARY DESTINATION cellsight COMPONENT python)







if(CELLSIGHT_BUILD_TESTS)
    add_executable(test_cellsight cpp/tests/test_lut2d.cpp cpp/tests/test_ecm.cpp cpp/tests/test_simulate.cpp)
    target_link_libraries(test_cellsight PRIVATE cellsight_core Catch2::Catch2WithMain)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
    catch_discover_tests(test_cellsight)
endif()
