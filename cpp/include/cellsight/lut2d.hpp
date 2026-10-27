#pragma once 
#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace std;

class  Lut2D{

public :
            Lut2D(vector<double> x ,vector <double> y,Eigen::MatrixXd table);
            double operator() ( double x , double y) const;

private : 
     vector<double> x_;
     vector<double> y_;
     Eigen::MatrixXd table_;
};