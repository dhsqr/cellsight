  #include "cellsight/lut2d.hpp"
#include <algorithm>

using namespace std;

// Save the data and make sure it is valid
Lut2D::Lut2D(vector<double> soc_points,
             vector<double> temp_points,
             Eigen::MatrixXd values)
    : x_(move(soc_points)),
      y_(move(temp_points)),
      table_(move(values))
{
    if (x_.size() < 2 || y_.size() < 2) {
        throw invalid_argument("Need at least 2 points on each axis");
    }
    if (table_.rows() != (int)x_.size() || table_.cols() != (int)y_.size()) {
        throw invalid_argument("Grid size does not match the points");
    }
}

// Look up a value at any SOC and temperature
double Lut2D::operator()(double soc, double temp) const
{
    // Stay inside the table
    soc  = clamp(soc,  x_.front(), x_.back());
    temp = clamp(temp, y_.front(), y_.back());

    // Find the cell that contains this point
    auto soc_it  = upper_bound(x_.begin(), x_.end(), soc);
    auto temp_it = upper_bound(y_.begin(), y_.end(), temp);

    if (soc_it  == x_.begin()) ++soc_it;
    if (temp_it == y_.begin()) ++temp_it;
    if (soc_it  == x_.end())   --soc_it;
    if (temp_it == y_.end())   --temp_it;

    int i = distance(x_.begin(), soc_it)  - 1;   // SOC index
    int j = distance(y_.begin(), temp_it) - 1;   // temp index

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