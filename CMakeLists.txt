cmake_minimum_required(VERSION 3.15...4.0)
project(cellsight LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core bindings/module.cpp)

install(TARGETS _core LIBRARY DESTINATION cellsight)