#pragma once

#include <cellsight/lut2d.hpp>

namespace cellsight {

// 2RC Thevenin cell. Tables are f(soc, temperature).
// Discharge current is positive. soc is in [0, 1], q_ah is amp-hours.
struct EcmParams {
    Lut2D ocv,r0,r1,c1,r2,c2;
    double q_ah;
    double eta_charge = 1.0;
};

// The three dynamic states: SOC and the two RC voltages.
struct EcmState {
    double soc = 0.0;
    double v1 = 0.0;
    double v2 = 0.0;
};

void validate(const EcmParams& p);

EcmState ecm_step(
    const EcmParams& p,
    const EcmState& x,
    double i_a,
    double temp_c,
    double dt_s,
    double cap_scale = 1.0,
    double r0_scale = 1.0);

double terminal_voltage(
    const EcmParams& p,
    const EcmState& x,
    double i_a,
    double temp_c,
    double cap_scale = 1.0,
    double r0_scale = 1.0);

}  // namespace cellsight
