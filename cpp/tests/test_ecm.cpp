#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "cellsight/ecm.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using cellsight::EcmParams;
using cellsight::EcmState;
using cellsight::Lut2D;
using cellsight::ecm_step;
using cellsight::terminal_voltage;
using cellsight::validate;

namespace {

Lut2D constant_table(double value)
{
    Eigen::MatrixXd table(2, 2);
    table.setConstant(value);
    return Lut2D({0.0, 1.0}, {0.0, 45.0}, std::move(table));
}

// One entry differs from the fill so a single bad cell is visible to validate.
Lut2D table_with_entry(double fill, double bad, int row, int col)
{
    Eigen::MatrixXd table(2, 2);
    table.setConstant(fill);
    table(row, col) = bad;
    return Lut2D({0.0, 1.0}, {0.0, 45.0}, std::move(table));
}

EcmParams make_cell(double q_ah, double eta_charge = 1.0)
{
    return EcmParams{
        constant_table(3.7),
        constant_table(0.01),
        constant_table(0.01),
        constant_table(1000.0),
        constant_table(0.02),
        constant_table(5000.0),
        q_ah,
        eta_charge,
    };
}

}  // namespace

TEST_CASE("discharge")
{
    // 2.5 A for 3600 s is one hour, which removes 2.5 Ah.
    EcmParams params = make_cell(2.5);
    EcmState state;
    state.soc = 1.0;
    state.v1 = 0.0;
    state.v2 = 0.0;

    constexpr double current_a = 2.5;
    constexpr double temp_c = 25.0;
    constexpr double dt_s = 1.0;
    constexpr int steps = 3600;

    for (int step = 0; step < steps; ++step) {
        state = ecm_step(params, state, current_a, temp_c, dt_s);
    }

    REQUIRE(state.soc == Catch::Approx(0.0).epsilon(0.0).margin(1e-10));
}

TEST_CASE("charging efficiency")
{
    EcmParams params = make_cell(2.5, 0.98);
    EcmState state;
    state.soc = 0.0;
    state.v1 = 0.0;
    state.v2 = 0.0;

    constexpr double current_a = -2.5;
    constexpr double temp_c = 25.0;
    constexpr double dt_s = 1.0;
    constexpr int steps = 3600;

    for (int step = 0; step < steps; ++step) {
        state = ecm_step(params, state, current_a, temp_c, dt_s);
    }

    REQUIRE(state.soc == Catch::Approx(0.98).epsilon(0.0).margin(1e-10));
}

TEST_CASE("capacity scaling")
{
    EcmParams params = make_cell(2.5);
    EcmState state;
    state.soc = 1.0;
    state.v1 = 0.0;
    state.v2 = 0.0;

    constexpr double current_a = 2.5;
    constexpr double cap_scale = 0.5;
    constexpr double temp_c = 25.0;
    constexpr double dt_s = 1.0;
    constexpr int steps = 1800;

    for (int step = 0; step < steps; ++step) {
        state = ecm_step(params, state, current_a, temp_c, dt_s, cap_scale);
    }

    REQUIRE(state.soc == Catch::Approx(0.0).epsilon(0.0).margin(1e-10));
}

TEST_CASE("zero-current charge conservation")
{
    EcmParams params = make_cell(2.5);
    EcmState state;
    state.soc = 0.6;
    state.v1 = 0.0;
    state.v2 = 0.0;

    for (double dt_s : {1.0, 10.0, 100.0}) {
        state = ecm_step(params, state, 0.0, 25.0, dt_s);
        REQUIRE(state.soc == 0.6);
    }
}

TEST_CASE("RC voltage buildup")
{
    constexpr double r1 = 0.05;
    constexpr double c1 = 2000.0;
    constexpr double current_a = 4.0;
    constexpr double temp_c = 25.0;
    constexpr double t_end_s = 2000.0;

    EcmParams params = make_cell(2.5);
    params.r1 = constant_table(r1);
    params.c1 = constant_table(c1);

    const double expected =
        r1 * current_a * (1.0 - std::exp(-t_end_s / (r1 * c1)));

    for (double dt_s : {1.0, 10.0, 100.0}) {
        EcmState state;
        state.soc = 0.5;
        state.v1 = 0.0;
        state.v2 = 0.0;

        const int steps = static_cast<int>(t_end_s / dt_s);
        for (int step = 0; step < steps; ++step) {
            state = ecm_step(params, state, current_a, temp_c, dt_s);
        }

        REQUIRE(state.v1 == Catch::Approx(expected).epsilon(0.0).margin(1e-12));
    }
}

TEST_CASE("relaxation")
{
    constexpr double r1 = 0.01;
    constexpr double c1 = 1000.0;
    constexpr double r2 = 0.02;
    constexpr double c2 = 5000.0;
    constexpr double dt_s = 25.0;
    constexpr double v0 = 0.05;
    constexpr double soc0 = 0.4;

    EcmParams params = make_cell(2.5);
    params.r1 = constant_table(r1);
    params.c1 = constant_table(c1);
    params.r2 = constant_table(r2);
    params.c2 = constant_table(c2);

    EcmState state;
    state.soc = soc0;
    state.v1 = v0;
    state.v2 = v0;

    state = ecm_step(params, state, 0.0, 25.0, dt_s);

    REQUIRE(state.v1 ==
            Catch::Approx(v0 * std::exp(-dt_s / (r1 * c1))).epsilon(0.0).margin(1e-12));
    REQUIRE(state.v2 ==
            Catch::Approx(v0 * std::exp(-dt_s / (r2 * c2))).epsilon(0.0).margin(1e-12));
    REQUIRE(state.soc == soc0);
}

TEST_CASE("terminal voltage")
{
    constexpr double ocv = 3.85;
    constexpr double r0 = 0.03;
    constexpr double r0_scale = 2.0;
    constexpr double cap_scale = 0.7;
    constexpr double v1 = 0.02;
    constexpr double v2 = 0.05;
    constexpr double temp_c = 25.0;

    EcmParams params = make_cell(2.5);
    params.ocv = constant_table(ocv);
    params.r0 = constant_table(r0);

    EcmState state;
    state.soc = 0.5;
    state.v1 = v1;
    state.v2 = v2;
    const EcmState before = state;

    const double discharge_a = 4.0;
    const double discharge_v = terminal_voltage(
        params, state, discharge_a, temp_c, cap_scale, r0_scale);
    REQUIRE(discharge_v == Catch::Approx(ocv - r0 * r0_scale * discharge_a - v1 - v2)
                               .epsilon(0.0).margin(1e-12));

    const double charge_a = -4.0;
    const double charge_v = terminal_voltage(
        params, state, charge_a, temp_c, cap_scale, r0_scale);
    REQUIRE(charge_v ==
            Catch::Approx(ocv - r0 * r0_scale * charge_a - v1 - v2).epsilon(0.0).margin(1e-12));

    REQUIRE(state.soc == before.soc);
    REQUIRE(state.v1 == before.v1);
    REQUIRE(state.v2 == before.v2);
}

TEST_CASE("invalid inputs")
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    EcmParams good = make_cell(2.5, 0.98);
    EcmState state;
    state.soc = 0.5;
    state.v1 = 0.01;
    state.v2 = 0.02;

    REQUIRE_NOTHROW(validate(good));

    SECTION("capacity")
    {
        for (double q_ah : {0.0, -1.0, nan, inf, -inf}) {
            EcmParams bad = good;
            bad.q_ah = q_ah;
            REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);
        }
    }

    SECTION("efficiency")
    {
        for (double eta : {0.0, -0.1, 1.1, nan, inf, -inf}) {
            EcmParams bad = good;
            bad.eta_charge = eta;
            REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);
        }
    }

    SECTION("table entries")
    {
        EcmParams zero_r0 = good;
        zero_r0.r0 = constant_table(0.0);
        REQUIRE_NOTHROW(validate(zero_r0));

        EcmParams bad = good;
        bad.ocv = table_with_entry(3.7, nan, 1, 0);
        REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);
        bad = good;
        bad.ocv = table_with_entry(3.7, inf, 0, 1);
        REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);

        bad = good;
        bad.r0 = table_with_entry(0.01, -0.01, 1, 0);
        REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);
        bad = good;
        bad.r0 = table_with_entry(0.01, nan, 0, 1);
        REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);

        for (double bad_value : {0.0, -0.01, nan, inf}) {
            bad = good;
            bad.r1 = table_with_entry(0.01, bad_value, 1, 0);
            REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);

            bad = good;
            bad.c1 = table_with_entry(1000.0, bad_value, 1, 0);
            REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);

            bad = good;
            bad.r2 = table_with_entry(0.02, bad_value, 0, 1);
            REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);

            bad = good;
            bad.c2 = table_with_entry(5000.0, bad_value, 0, 1);
            REQUIRE_THROWS_AS(validate(bad), std::invalid_argument);
        }
    }

    SECTION("time step")
    {
        for (double dt_s : {0.0, -1.0, nan, inf, -inf}) {
            REQUIRE_THROWS_AS(ecm_step(good, state, 1.0, 25.0, dt_s),
                              std::invalid_argument);
        }
    }

    SECTION("scales")
    {
        REQUIRE_THROWS_AS(terminal_voltage(good, state, 1.0, 25.0, 0.0, 1.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(terminal_voltage(good, state, 1.0, 25.0, -0.5, 1.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(terminal_voltage(good, state, 1.0, 25.0, 1.0, 0.0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(terminal_voltage(good, state, 1.0, 25.0, 1.0, -0.5),
                          std::invalid_argument);

        for (double scale : {0.0, -0.5, nan, inf, -inf}) {
            REQUIRE_THROWS_AS(ecm_step(good, state, 1.0, 25.0, 1.0, scale, 1.0),
                              std::invalid_argument);
            REQUIRE_THROWS_AS(ecm_step(good, state, 1.0, 25.0, 1.0, 1.0, scale),
                              std::invalid_argument);
            REQUIRE_THROWS_AS(terminal_voltage(good, state, 1.0, 25.0, scale, 1.0),
                              std::invalid_argument);
            REQUIRE_THROWS_AS(terminal_voltage(good, state, 1.0, 25.0, 1.0, scale),
                              std::invalid_argument);
        }
    }

    SECTION("non-finite inputs")
    {
        for (double bad_value : {nan, inf, -inf}) {
            REQUIRE_THROWS_AS(ecm_step(good, state, bad_value, 25.0, 1.0),
                              std::invalid_argument);
            REQUIRE_THROWS_AS(ecm_step(good, state, 1.0, bad_value, 1.0),
                              std::invalid_argument);
            REQUIRE_THROWS_AS(terminal_voltage(good, state, bad_value, 25.0),
                              std::invalid_argument);
            REQUIRE_THROWS_AS(terminal_voltage(good, state, 1.0, bad_value),
                              std::invalid_argument);

            EcmState bad_state = state;
            bad_state.soc = bad_value;
            REQUIRE_THROWS_AS(ecm_step(good, bad_state, 1.0, 25.0, 1.0),
                              std::invalid_argument);
            REQUIRE_THROWS_AS(terminal_voltage(good, bad_state, 1.0, 25.0),
                              std::invalid_argument);

            bad_state = state;
            bad_state.v1 = bad_value;
            REQUIRE_THROWS_AS(ecm_step(good, bad_state, 1.0, 25.0, 1.0),
                              std::invalid_argument);
            REQUIRE_THROWS_AS(terminal_voltage(good, bad_state, 1.0, 25.0),
                              std::invalid_argument);

            bad_state = state;
            bad_state.v2 = bad_value;
            REQUIRE_THROWS_AS(ecm_step(good, bad_state, 1.0, 25.0, 1.0),
                              std::invalid_argument);
            REQUIRE_THROWS_AS(terminal_voltage(good, bad_state, 1.0, 25.0),
                              std::invalid_argument);
        }
    }
}





