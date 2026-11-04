#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "cellsight/lut2d.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace std;
using cellsight::Lut2D; 

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
    REQUIRE(lut(0.5, 0.5) == Catch::Approx(1.0));
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

TEST_CASE("uneven grid nodes are exact") {
    vector<double> x = {0.0, 0.1, 0.5, 1.0};
    vector<double> y = {-10.0, 0.0, 25.0, 45.0};
    Eigen::MatrixXd table(4, 4);
    table << 1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0,  10.0, 11.0, 12.0,
             13.0, 14.0, 15.0, 16.0;

    Lut2D lut(x, y, table);

    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < y.size(); ++j) {
            REQUIRE(lut(x[i], y[j]) == Catch::Approx(table(i, j)).margin(1e-12));
        }
    }
}

TEST_CASE("bilinear polynomial is exact inside the grid") {
    auto f = [](double x, double y) {
        return 1.0 + 2.0 * x + 3.0 * y + 0.5 * x * y;
    };
    vector<double> x = {0.0, 1.0, 2.0};
    vector<double> y = {-1.0, 0.5, 3.0};
    Eigen::MatrixXd table(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            table(i, j) = f(x[i], y[j]);
        }
    }

    Lut2D lut(x, y, table);

    REQUIRE(lut(0.3, -0.2) == Catch::Approx(f(0.3, -0.2)).margin(1e-12));
    REQUIRE(lut(1.7, 1.5) == Catch::Approx(f(1.7, 1.5)).margin(1e-12));
    REQUIRE(lut(0.5, 2.0) == Catch::Approx(f(0.5, 2.0)).margin(1e-12));
}

TEST_CASE("rejects bad grids and NaN") {
    Eigen::MatrixXd one_point(1, 2);
    one_point << 0.0, 1.0;
    REQUIRE_THROWS_AS(Lut2D({0.0}, {0.0, 1.0}, one_point), std::invalid_argument);

    Eigen::MatrixXd unsorted(3, 2);
    unsorted.setZero();
    REQUIRE_THROWS_AS(Lut2D({0.0, 1.0, 0.5}, {0.0, 1.0}, unsorted), std::invalid_argument);

    Eigen::MatrixXd duplicate(2, 2);
    duplicate.setZero();
    REQUIRE_THROWS_AS(Lut2D({0.0, 0.0}, {0.0, 1.0}, duplicate), std::invalid_argument);

    Eigen::MatrixXd wrong_shape(2, 3);
    wrong_shape.setZero();
    REQUIRE_THROWS_AS(Lut2D({0.0, 1.0}, {0.0, 1.0}, wrong_shape), std::invalid_argument);

    Eigen::MatrixXd ok(2, 2);
    ok << 1.0, 2.0,
          3.0, 4.0;
    Lut2D lut({0.0, 1.0}, {0.0, 1.0}, ok);
    REQUIRE_THROWS_AS(lut(std::numeric_limits<double>::quiet_NaN(), 0.0), std::invalid_argument);
}

TEST_CASE("d_dx of a bilinear polynomial is exact") {
    auto clamp_y = [](double y) {
        return std::clamp(y, -1.0, 3.0);
    };
    auto f = [](double x, double y) {
        return 1.0 + 2.0 * x + 3.0 * y + 0.5 * x * y;
    };
    vector<double> x = {0.0, 1.0, 2.0};
    vector<double> y = {-1.0, 0.5, 3.0};
    Eigen::MatrixXd table(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            table(i, j) = f(x[i], y[j]);
        }
    }

    Lut2D lut(x, y, table);

    auto expect = [&](double xq, double yq) {
        const double slope = 2.0 + 0.5 * clamp_y(yq);
        REQUIRE(lut.d_dx(xq, yq) == Catch::Approx(slope).margin(1e-12));
    };

    expect(0.3, -0.2);   // interior
    expect(1.2, 1.0);    // interior, other cell
    expect(0.3, -5.0);   // y below the grid
    expect(1.2, 10.0);   // y above the grid
    expect(-2.0, 0.0);   // x left of the grid, edge cell
    expect(5.0, 1.0);    // x right of the grid, edge cell
}

TEST_CASE("d_dx matches a central difference inside a cell") {
    vector<double> x = {0.0, 0.1, 0.5, 1.0};
    vector<double> y = {-10.0, 0.0, 25.0, 45.0};
    Eigen::MatrixXd table(4, 4);
    table << 1.0,  2.0,  3.0,  4.0,
             5.0,  8.0,  6.0,  9.0,
             4.0,  1.0,  7.0,  2.0,
             3.0,  5.0,  9.0,  1.0;

    Lut2D lut(x, y, table);
    const double h = 1e-6;

    auto check = [&](double xq, double yq) {
        const double finite = (lut(xq + h, yq) - lut(xq - h, yq)) / (2.0 * h);
        REQUIRE(lut.d_dx(xq, yq) == Catch::Approx(finite).margin(1e-6));
    };

    check(0.3, 10.0);   // cell x (0.1, 0.5), y (0, 25)
    check(0.75, -4.0);  // cell x (0.5, 1), y (-10, 0)
}

TEST_CASE("d_dx rejects non-finite axes and NaN queries") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd table(2, 2);
    table.setZero();

    REQUIRE_THROWS_AS(Lut2D({0.0, nan}, {0.0, 1.0}, table), std::invalid_argument);
    REQUIRE_THROWS_AS(Lut2D({0.0, 1.0}, {nan, 2.0}, table), std::invalid_argument);
    REQUIRE_THROWS_AS(Lut2D({0.0, inf}, {0.0, 1.0}, table), std::invalid_argument);
    REQUIRE_THROWS_AS(Lut2D({0.0, 1.0}, {-inf, 1.0}, table), std::invalid_argument);

    Lut2D lut({0.0, 1.0}, {0.0, 1.0}, table);
    REQUIRE_THROWS_AS(lut.d_dx(nan, 0.5), std::invalid_argument);
    REQUIRE_THROWS_AS(lut.d_dx(0.5, nan), std::invalid_argument);
}