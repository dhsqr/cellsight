#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "cellsight/simulate.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using cellsight::EcmParams;
using cellsight::EcmState;
using cellsight::Lut2D;
using cellsight::ecm_step;
using cellsight::simulate;
using cellsight::terminal_voltage;

namespace {

Lut2D constant_table(double value)
{
    Eigen::MatrixXd table(2, 2);
    table.setConstant(value);
    return Lut2D({0.0, 1.0}, {0.0, 45.0}, std::move(table));
}

// Value is a + b * soc at both temperatures, so lookup is exact inside
// the SOC edges and does not depend on temperature.
Lut2D soc_line(double intercept, double slope)
{
    Eigen::MatrixXd table(2, 2);
    table << intercept, intercept,
        intercept + slope, intercept + slope;
    return Lut2D({0.0, 1.0}, {0.0, 40.0}, std::move(table));
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

void require_close(double actual, double expected, double margin)
{
    REQUIRE(actual == Catch::Approx(expected).epsilon(0.0).margin(margin));
}

struct Reference {
    std::vector<double> voltage;
    EcmState state;
};

// Same recording rule as simulate: voltage before the step, then advance.
Reference run_one_cell(const EcmParams& params,
                       const Eigen::VectorXd& current_a,
                       const Eigen::VectorXd& temp_c,
                       double soc0,
                       double cap_scale,
                       double r0_scale,
                       double dt_s,
                       int record_every)
{
    EcmState state;
    state.soc = soc0;
    state.v1 = 0.0;
    state.v2 = 0.0;

    Reference out;
    for (Eigen::Index step = 0; step < current_a.size(); ++step) {
        if (step % record_every == 0) {
            out.voltage.push_back(terminal_voltage(
                params, state, current_a(step), temp_c(step), cap_scale,
                r0_scale));
        }
        state = ecm_step(params, state, current_a(step), temp_c(step), dt_s,
                         cap_scale, r0_scale);
    }
    out.state = state;
    return out;
}

}  // namespace

TEST_CASE("batch matches single-cell loop")
{
    EcmParams params = make_cell(2.5, 0.97);

    Eigen::VectorXd current_a(6);
    Eigen::VectorXd temp_c(6);
    current_a << 1.5, -0.4, 0.0, 2.0, -1.0, 0.5;
    temp_c << 10.0, 25.0, 25.0, 40.0, 15.0, 30.0;

    Eigen::VectorXd soc0(3);
    Eigen::VectorXd cap_scale(3);
    Eigen::VectorXd r0_scale(3);
    soc0 << 0.2, 0.8, 1.0;
    cap_scale << 1.0, 0.5, 1.25;
    r0_scale << 1.0, 2.0, 0.8;

    constexpr double dt_s = 2.0;
    constexpr int record_every = 1;

    const auto batch = simulate(params, current_a, temp_c, soc0, cap_scale,
                                r0_scale, dt_s, record_every);

    REQUIRE(batch.voltage.rows() == current_a.size());
    REQUIRE(batch.voltage.cols() == soc0.size());
    REQUIRE(batch.final_state.rows() == soc0.size());
    REQUIRE(batch.final_state.cols() == 3);

    for (Eigen::Index cell = 0; cell < soc0.size(); ++cell) {
        const Reference one = run_one_cell(
            params, current_a, temp_c, soc0(cell), cap_scale(cell),
            r0_scale(cell), dt_s, record_every);

        REQUIRE(static_cast<Eigen::Index>(one.voltage.size()) ==
                batch.voltage.rows());
        for (Eigen::Index row = 0; row < batch.voltage.rows(); ++row) {
            require_close(batch.voltage(row, cell), one.voltage[static_cast<std::size_t>(row)],
                          1e-12);
        }
        require_close(batch.final_state(cell, 0), one.state.soc, 1e-10);
        require_close(batch.final_state(cell, 1), one.state.v1, 1e-12);
        require_close(batch.final_state(cell, 2), one.state.v2, 1e-12);
    }
}

TEST_CASE("recording intervals and shapes")
{
    EcmParams params = make_cell(2.5);
    constexpr double dt_s = 1.0;

    Eigen::VectorXd current_a(10);
    Eigen::VectorXd temp_c(10);
    for (Eigen::Index step = 0; step < current_a.size(); ++step) {
        current_a(step) = 0.25 * static_cast<double>(step + 1);
        temp_c(step) = 20.0 + static_cast<double>(step);
    }

    Eigen::VectorXd soc0(2);
    Eigen::VectorXd cap_scale(2);
    Eigen::VectorXd r0_scale(2);
    soc0 << 0.9, 0.4;
    cap_scale << 1.0, 0.8;
    r0_scale << 1.0, 1.5;

    // T = 10. Steps 0, 3, 6, 9 are recorded. Final state is after all 10.
    {
        constexpr int record_every = 3;
        const auto batch = simulate(params, current_a, temp_c, soc0, cap_scale,
                                    r0_scale, dt_s, record_every);
        REQUIRE(batch.voltage.rows() == 4);
        REQUIRE(batch.voltage.cols() == 2);
        REQUIRE(batch.final_state.rows() == 2);
        REQUIRE(batch.final_state.cols() == 3);

        for (Eigen::Index cell = 0; cell < soc0.size(); ++cell) {
            const Reference one = run_one_cell(
                params, current_a, temp_c, soc0(cell), cap_scale(cell),
                r0_scale(cell), dt_s, record_every);
            REQUIRE(one.voltage.size() == 4);
            for (std::size_t row = 0; row < one.voltage.size(); ++row) {
                require_close(batch.voltage(static_cast<Eigen::Index>(row), cell),
                              one.voltage[row], 1e-12);
            }
            require_close(batch.final_state(cell, 0), one.state.soc, 1e-10);
            require_close(batch.final_state(cell, 1), one.state.v1, 1e-12);
            require_close(batch.final_state(cell, 2), one.state.v2, 1e-12);
        }
    }

    // record_every == T stores only the initial voltage. State still advances.
    {
        constexpr int record_every = 10;
        const auto batch = simulate(params, current_a, temp_c, soc0, cap_scale,
                                    r0_scale, dt_s, record_every);
        REQUIRE(batch.voltage.rows() == 1);
        REQUIRE(batch.voltage.cols() == 2);

        for (Eigen::Index cell = 0; cell < soc0.size(); ++cell) {
            const Reference one = run_one_cell(
                params, current_a, temp_c, soc0(cell), cap_scale(cell),
                r0_scale(cell), dt_s, record_every);
            REQUIRE(one.voltage.size() == 1);
            require_close(batch.voltage(0, cell), one.voltage[0], 1e-12);
            require_close(batch.final_state(cell, 0), one.state.soc, 1e-10);
            REQUIRE(batch.final_state(cell, 0) != soc0(cell));
        }
    }

    // record_every > T still records step 0 and still runs every step.
    {
        constexpr int record_every = 25;
        const auto batch = simulate(params, current_a, temp_c, soc0, cap_scale,
                                    r0_scale, dt_s, record_every);
        REQUIRE(batch.voltage.rows() == 1);
        REQUIRE(batch.voltage.cols() == 2);
        const Reference one = run_one_cell(
            params, current_a, temp_c, soc0(0), cap_scale(0), r0_scale(0), dt_s,
            record_every);
        require_close(batch.voltage(0, 0), one.voltage[0], 1e-12);
        require_close(batch.final_state(0, 0), one.state.soc, 1e-10);
        require_close(batch.final_state(0, 1), one.state.v1, 1e-12);
        require_close(batch.final_state(0, 2), one.state.v2, 1e-12);
    }
}

TEST_CASE("simulate rejects invalid inputs")
{
    const EcmParams good = make_cell(2.5, 0.98);
    Eigen::VectorXd current_a(4);
    Eigen::VectorXd temp_c(4);
    current_a << 1.0, -0.5, 0.0, 0.25;
    temp_c << 25.0, 25.0, 30.0, 20.0;
    Eigen::VectorXd soc0(2);
    Eigen::VectorXd cap_scale(2);
    Eigen::VectorXd r0_scale(2);
    soc0 << 0.5, 0.7;
    cap_scale << 1.0, 0.9;
    r0_scale << 1.0, 1.1;

    REQUIRE_NOTHROW(simulate(good, current_a, temp_c, soc0, cap_scale, r0_scale,
                             1.0, 1));

    SECTION("lengths")
    {
        Eigen::VectorXd short_temp(3);
        short_temp << 25.0, 25.0, 25.0;
        REQUIRE_THROWS_AS(simulate(good, current_a, short_temp, soc0, cap_scale,
                                   r0_scale, 1.0, 1),
                          std::invalid_argument);

        Eigen::VectorXd one_scale(1);
        one_scale << 1.0;
        REQUIRE_THROWS_AS(simulate(good, current_a, temp_c, soc0, one_scale,
                                   r0_scale, 1.0, 1),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(simulate(good, current_a, temp_c, soc0, cap_scale,
                                   one_scale, 1.0, 1),
                          std::invalid_argument);
    }

    SECTION("empty")
    {
        Eigen::VectorXd no_steps;
        Eigen::VectorXd no_cells;
        REQUIRE_THROWS_AS(simulate(good, no_steps, no_steps, soc0, cap_scale,
                                   r0_scale, 1.0, 1),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(simulate(good, current_a, temp_c, no_cells, no_cells,
                                   no_cells, 1.0, 1),
                          std::invalid_argument);
    }

    SECTION("time step and record interval")
    {
        for (double dt_s : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()}) {
            REQUIRE_THROWS_AS(simulate(good, current_a, temp_c, soc0, cap_scale,
                                       r0_scale, dt_s, 1),
                              std::invalid_argument);
        }
        REQUIRE_THROWS_AS(simulate(good, current_a, temp_c, soc0, cap_scale,
                                   r0_scale, 1.0, 0),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(simulate(good, current_a, temp_c, soc0, cap_scale,
                                   r0_scale, 1.0, -2),
                          std::invalid_argument);
    }

    SECTION("non-finite profiles and non-positive scales")
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();

        Eigen::VectorXd bad_current = current_a;
        bad_current(2) = nan;
        REQUIRE_THROWS_AS(simulate(good, bad_current, temp_c, soc0, cap_scale,
                                   r0_scale, 1.0, 1),
                          std::invalid_argument);

        Eigen::VectorXd bad_temp = temp_c;
        bad_temp(0) = inf;
        REQUIRE_THROWS_AS(simulate(good, current_a, bad_temp, soc0, cap_scale,
                                   r0_scale, 1.0, 1),
                          std::invalid_argument);

        Eigen::VectorXd bad_soc = soc0;
        bad_soc(1) = -inf;
        REQUIRE_THROWS_AS(simulate(good, current_a, temp_c, bad_soc, cap_scale,
                                   r0_scale, 1.0, 1),
                          std::invalid_argument);

        for (double scale : {0.0, -0.5, nan}) {
            Eigen::VectorXd bad_cap = cap_scale;
            bad_cap(0) = scale;
            REQUIRE_THROWS_AS(simulate(good, current_a, temp_c, soc0, bad_cap,
                                       r0_scale, 1.0, 1),
                              std::invalid_argument);

            Eigen::VectorXd bad_r0 = r0_scale;
            bad_r0(1) = scale;
            REQUIRE_THROWS_AS(simulate(good, current_a, temp_c, soc0, cap_scale,
                                       bad_r0, 1.0, 1),
                              std::invalid_argument);
        }
    }

    SECTION("parameters")
    {
        EcmParams bad = good;
        bad.q_ah = 0.0;
        REQUIRE_THROWS_AS(simulate(bad, current_a, temp_c, soc0, cap_scale,
                                   r0_scale, 1.0, 1),
                          std::invalid_argument);
    }
}

TEST_CASE("sloped table uses old soc")
{
    // OCV = 3 + soc, R0 = 0.02 + 0.02 * soc, R1 = 0.05 + 0.10 * soc.
    // Both temperatures hold the same value, so the lookup is the line.
    // Expected numbers use that line at the SOC present at the start of
    // the step. They do not call ecm_step or terminal_voltage.
    constexpr double q_ah = 2.5;
    constexpr double eta_charge = 0.5;
    constexpr double soc_start = 0.4;
    constexpr double current_amp = 5.0;
    constexpr double dt_s = 20.0;
    constexpr double cap = 1.0;
    constexpr double r0_scale = 2.0;
    constexpr double c1 = 1000.0;
    constexpr double r2 = 0.02;
    constexpr double c2 = 5000.0;

    EcmParams params{
        soc_line(3.0, 1.0),
        soc_line(0.02, 0.02),
        soc_line(0.05, 0.10),
        constant_table(c1),
        constant_table(r2),
        constant_table(c2),
        q_ah,
        eta_charge,
    };

    Eigen::VectorXd current_a(2);
    Eigen::VectorXd temp_c(2);
    Eigen::VectorXd soc0(1);
    Eigen::VectorXd cap_scale(1);
    Eigen::VectorXd r0s(1);
    current_a << current_amp, current_amp;
    temp_c << 25.0, 25.0;
    soc0 << soc_start;
    cap_scale << cap;
    r0s << r0_scale;

    const auto batch = simulate(params, current_a, temp_c, soc0, cap_scale, r0s,
                                dt_s, 1);

    const auto ocv = [](double soc) { return 3.0 + soc; };
    const auto r0 = [](double soc) { return 0.02 + 0.02 * soc; };
    const auto r1 = [](double soc) { return 0.05 + 0.10 * soc; };
    const auto voltage = [&](double soc) {
        return ocv(soc) - r0(soc) * r0_scale * current_amp;
    };
    const auto branch = [](double v, double resistance, double capacitance) {
        const double a = std::exp(-dt_s / (resistance * capacitance));
        const double g = -std::expm1(-dt_s / (resistance * capacitance));
        return a * v + resistance * g * current_amp;
    };
    const double removed = current_amp * dt_s / (3600.0 * q_ah * cap);

    const double soc_step1 = soc_start - removed;
    const double soc_step2 = soc_step1 - removed;
    // Branch voltages carried into a step were produced by the previous step.
    const double v1_after_1 = branch(0.0, r1(soc_start), c1);
    const double v1_after_2 = branch(v1_after_1, r1(soc_step1), c1);
    const double v2_after_1 = branch(0.0, r2, c2);
    const double v2_after_2 = branch(v2_after_1, r2, c2);

    // Step 0 starts from rest, so v1 = v2 = 0. Step 1 still looks up OCV and
    // R0 at soc_step1, then subtracts the branch voltages already stored.
    const double recorded_0 = voltage(soc_start);
    const double recorded_1 = voltage(soc_step1) - v1_after_1 - v2_after_1;
    const double recorded_1_if_new_soc =
        voltage(soc_step2) - v1_after_1 - v2_after_1;
    const double v1_if_lookup_used_new_soc =
        branch(branch(0.0, r1(soc_step1), c1), r1(soc_step2), c1);

    // The old-SOC and new-SOC answers differ by much more than the tolerance.
    REQUIRE(std::abs(recorded_0 - voltage(soc_step1)) > 1e-4);
    REQUIRE(std::abs(recorded_1 - recorded_1_if_new_soc) > 1e-4);
    REQUIRE(std::abs(v1_after_2 - v1_if_lookup_used_new_soc) > 1e-4);

    REQUIRE(batch.voltage.rows() == 2);
    REQUIRE(batch.voltage.cols() == 1);
    require_close(batch.voltage(0, 0), recorded_0, 1e-12);
    require_close(batch.voltage(1, 0), recorded_1, 1e-12);
    require_close(batch.final_state(0, 0), soc_step2, 1e-12);
    require_close(batch.final_state(0, 1), v1_after_2, 1e-12);
    require_close(batch.final_state(0, 2), v2_after_2, 1e-12);

    REQUIRE(std::abs(batch.voltage(0, 0) - voltage(soc_step1)) > 1e-4);
    REQUIRE(std::abs(batch.voltage(1, 0) - recorded_1_if_new_soc) > 1e-4);
    REQUIRE(std::abs(batch.final_state(0, 1) - v1_if_lookup_used_new_soc) > 1e-4);
}
