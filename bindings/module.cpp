#include  <pybind11/pybind11.h>

namespace py = pybind11;
 

double add(double a, double b){
    return (a+b);
}


PYBIND11_MODULE(_core, m){
    m.def("add",&add," function that add two num");
}
