#include <catch2/catch_test_macros.hpp>
#include "cellsight/lut2d.hpp"
#include <Eigen/Dense>
#include <vector>

using namespace std;

TEST_CASE("grid points return exact values") {
    vector<double> x = {0.0, 0.5, 1.0};
    vector<double> y = {0.0, 25.0};
    Eigen::MatrixXd table(3, 2);
    table << 1.0, 2.0,
             3.0, 4.0,
             5.0, 6.0;

    Lut2D lut(x, y, table);

    REQUIRE(lut(0.0, 0.0) == 1.0);
    REQUIRE(lut(0.5, 0.0) == 3.0);
    REQUIRE(lut(1.0, 25.0) == 6.0);
}

TEST_CASE("interpolation works") {
    vector<double> x = {0.0, 1.0};
    vector<double> y = {0.0, 1.0};
    Eigen::MatrixXd table(2, 2);
    table << 0.0, 1.0,
             1.0, 2.0;

    Lut2D lut(x, y, table);

    // midpoint should be average
    REQUIRE(lut(0.5, 0.5) == Approx(1.0));
}

TEST_CASE("clamps outside the grid") {
    vector<double> x = {0.0, 1.0};
    vector<double> y = {0.0, 1.0};
    Eigen::MatrixXd table(2, 2);
    table << 10.0, 20.0,
             30.0, 40.0;

    Lut2D lut(x, y, table);

    REQUIRE(lut(-1.0, 0.5) == lut(0.0, 0.5));
    REQUIRE(lut(2.0, 2.0) == lut(1.0, 1.0));
}