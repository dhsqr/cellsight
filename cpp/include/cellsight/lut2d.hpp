#pragma once
#include <Eigen/Dense>
#include <vector>

namespace cellsight {

class Lut2D {
public:
    Lut2D(std::vector<double> x, std::vector<double> y, Eigen::MatrixXd table);
    double operator()(double x, double y) const;
    double d_dx(double x, double y) const;
    const Eigen::MatrixXd& table() const { return table_; }

private:
    std::vector<double> x_;
    std::vector<double> y_;
    Eigen::MatrixXd table_;
};

}  // namespace cellsight
