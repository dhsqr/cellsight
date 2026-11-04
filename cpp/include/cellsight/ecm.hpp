#pragma once

#include<cellsight/lut2d.hpp>


namespace cellsight{
struct EcmParams{
          double ocv , r0,r1,c1,r2,c2;
          double q_ah;
          double eta_charge = 1.0;
};

struct EcmState{
    double soc,v1,v2;
       double h = 0.0; 
       double s = 0.0;
};
}