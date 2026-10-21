cmake_minimum_required(VERSION 3.15...4.0)
project(cellsight LANGUAGES CXX)



include (FetchContent)

#Eigen lib fetch 
FetchContent_Declare(
Eigen 
GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
# downloading eigen and making it available for file
FetchContent_MakeAvailable (Eigen)


#now time fr catch2 ... creating a switch so for normal build from pip we can skip - defaaultOFF
option(CELLSIGHT_BUILD_TESTS "Build tests" OFF)


# will turn ON then only we use catch2 for testing otherwise no need to install for user -_-
if(CELLSIGHT_BUILD_TESTS)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/ca
    tchorg/Catch2.git
    GIT_TAG v3.5.2
)
#downlaoding catch2
FetchContent_MakeAvailable(Catch2)
enable_testing() #starting cmake tester
endif()



set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core bindings/module.cpp)

#allowing cpp to use eigen
target_include_directories(_core PRIVATE ${eigen_SOURCE_DIR})

install(TARGETS _core LIBRARY DESTINATION cellsight)