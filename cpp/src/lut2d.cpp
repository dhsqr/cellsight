#include "cellsight/lut2d.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace cellsight {
namespace {

struct Cell {
    int i;
    int j;
    double x;
    double y;
};

// Same cell as operator(): clamp into the grid, then upper_bound.
Cell locate(const std::vector<double>& xs,
            const std::vector<double>& ys,
            double x,
            double y)
{
    if (std::isnan(x) || std::isnan(y)) {
        throw std::invalid_argument("NaN input");
    }

    x = std::clamp(x, xs.front(), xs.back());
    y = std::clamp(y, ys.front(), ys.back());

    auto x_it = std::upper_bound(xs.begin(), xs.end(), x);
    auto y_it = std::upper_bound(ys.begin(), ys.end(), y);

    if (x_it == xs.begin()) ++x_it;
    if (y_it == ys.begin()) ++y_it;
    if (x_it == xs.end()) --x_it;
    if (y_it == ys.end()) --y_it;

    const int i = static_cast<int>(std::distance(xs.begin(), x_it)) - 1;
    const int j = static_cast<int>(std::distance(ys.begin(), y_it)) - 1;
    return {i, j, x, y};
}

}  // namespace

// Save the data and make sure it is valid
Lut2D::Lut2D(std::vector<double> soc_points,
             std::vector<double> temp_points,
             Eigen::MatrixXd values)
    : x_(std::move(soc_points)),
      y_(std::move(temp_points)),
      table_(std::move(values))
{
    if (x_.size() < 2 || y_.size() < 2) {
        throw std::invalid_argument("Need at least 2 points on each axis");
    }
    if (table_.rows() != static_cast<Eigen::Index>(x_.size()) ||
        table_.cols() != static_cast<Eigen::Index>(y_.size())) {
        throw std::invalid_argument("Grid size does not match the points");
    }
    for (const std::vector<double>* axis : {&x_, &y_}) {
        const std::vector<double>& a = *axis;
        for (double value : a) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("axis values must be finite");
            }
        }
        for (std::size_t k = 0; k + 1 < a.size(); ++k) {
            if (a[k + 1] <= a[k]) {
                throw std::invalid_argument("axis must be strictly increasing");
            }
        }
    }
}

// Look up a value at any SOC and temperature
double Lut2D::operator()(double soc, double temp) const
{
    const Cell cell = locate(x_, y_, soc, temp);
    const int i = cell.i;
    const int j = cell.j;
    soc = cell.x;
    temp = cell.y;

    // The four corners of the cell
    double soc1  = x_[i],     soc2  = x_[i + 1];
    double temp1 = y_[j],     temp2 = y_[j + 1];

    double v11 = table_(i,     j);       // bottom-left
    double v21 = table_(i + 1, j);       // bottom-right
    double v12 = table_(i,     j + 1);   // top-left
    double v22 = table_(i + 1, j + 1);   // top-right

    // How far across the cell (0.0 to 1.0)
    double frac_soc  = (soc  - soc1)  / (soc2  - soc1);
    double frac_temp = (temp - temp1) / (temp2 - temp1);

    // Blend the four values
    double bottom = v11 * (1 - frac_soc) + v21 * frac_soc;
    double top    = v12 * (1 - frac_soc) + v22 * frac_soc;

    return bottom * (1 - frac_temp) + top * frac_temp;
}

double Lut2D::d_dx(double soc, double temp) const
{
    const Cell cell = locate(x_, y_, soc, temp);
    const int i = cell.i;
    const int j = cell.j;

    const double f00 = table_(i,     j);
    const double f10 = table_(i + 1, j);
    const double f01 = table_(i,     j + 1);
    const double f11 = table_(i + 1, j + 1);

    const double ty = (cell.y - y_[j]) / (y_[j + 1] - y_[j]);
    return ((1.0 - ty) * (f10 - f00) + ty * (f11 - f01)) / (x_[i + 1] - x_[i]);
}

}  // namespace cellsight
