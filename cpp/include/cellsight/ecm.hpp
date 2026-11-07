#pragma once

#include <cellsight/lut2d.hpp>

namespace cellsight {

// 2RC Thevenin cell. Tables are f(soc, temp_c).
//
// Units: soc is a fraction. v1 and v2 are volts. ocv is volts.
// r0, r1, and r2 are ohms. c1 and c2 are farads. q_ah is amp-hours.
// i_a is amperes, temp_c is degrees Celsius, and dt_s is seconds.
// cap_scale and r0_scale are positive dimensionless multipliers.
//
// Discharge current is positive. eta_charge applies only while charging
// (i_a < 0). Discharge and zero current use eta = 1.
//
// Stored soc is not clamped. Lookups clamp soc and temperature to the
// table edges.
//
// ecm_step looks up r1, c1, r2, and c2 at the soc and temperature from
// the start of the step, then
//   tau = R * C
//   a = exp(-dt / tau)
//   g = -expm1(-dt / tau)
//   soc' = soc - eta * I * dt / (3600 * q_ah * cap_scale)
//   v' = a * v + R * g * I
// for each RC branch.
//
// terminal_voltage looks up ocv and r0 at the given soc and temperature:
//   V = ocv - r0 * r0_scale * I - v1 - v2
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
