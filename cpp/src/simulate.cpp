#include "cellsight/simulate.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cellsight {
namespace {

void require(bool ok, const char* msg)
{
    if (!ok) {
        throw std::invalid_argument(msg);
    }
}

void require_finite(const Eigen::VectorXd& values, const char* name)
{
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        require(std::isfinite(values(i)), name);
    }
}

void require_positive(const Eigen::VectorXd& values, const char* name)
{
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        require(std::isfinite(values(i)) && values(i) > 0.0, name);
    }
}

// False when a * b is negative or does not fit in Eigen::Index.
bool checked_mul(Eigen::Index a, Eigen::Index b, Eigen::Index& out)
{
    if (a < 0 || b < 0) {
        return false;
    }
    if (a != 0 && b > std::numeric_limits<Eigen::Index>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

// False when n elements of double do not fit in size_t.
bool checked_bytes(Eigen::Index n_elements, std::size_t& bytes)
{
    if (n_elements < 0) {
        return false;
    }
    const auto count = static_cast<std::size_t>(n_elements);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        return false;
    }
    bytes = count * sizeof(double);
    return true;
}

}  // namespace

SimulateResult simulate(const EcmParams& params,
                        const Eigen::VectorXd& current_a,
                        const Eigen::VectorXd& temp_c,
                        const Eigen::VectorXd& soc0,
                        const Eigen::VectorXd& cap_scale,
                        const Eigen::VectorXd& r0_scale,
                        double dt_s,
                        int record_every)
{
    require(record_every >= 1, "record_every");
    require(current_a.size() == temp_c.size(), "length");
    require(soc0.size() == cap_scale.size() && soc0.size() == r0_scale.size(),
            "length");
    require(current_a.size() > 0 && soc0.size() > 0, "empty");
    require(std::isfinite(dt_s) && dt_s > 0.0, "dt");

    require_finite(current_a, "current");
    require_finite(temp_c, "temp");
    require_finite(soc0, "soc0");
    require_positive(cap_scale, "cap_scale");
    require_positive(r0_scale, "r0_scale");

    // Parameter tables are scanned once here, not on every step.
    validate(params);

    const Eigen::Index n_steps = current_a.size();
    const Eigen::Index n_cells = soc0.size();
    const Eigen::Index every = static_cast<Eigen::Index>(record_every);

    Eigen::Index n_records = n_steps / every;
    if (n_steps % every != 0) {
        require(n_records < std::numeric_limits<Eigen::Index>::max(),
                "allocation");
        ++n_records;
    }

    Eigen::Index n_voltage = 0;
    Eigen::Index n_state = 0;
    require(checked_mul(n_records, n_cells, n_voltage), "allocation");
    require(checked_mul(n_cells, 3, n_state), "allocation");

    std::size_t voltage_bytes = 0;
    std::size_t state_bytes = 0;
    require(checked_bytes(n_voltage, voltage_bytes), "allocation");
    require(checked_bytes(n_state, state_bytes), "allocation");
    require(voltage_bytes <= std::numeric_limits<std::size_t>::max() - state_bytes,
            "allocation");

    SimulateResult out;
    out.voltage.resize(n_records, n_cells);
    out.final_state.resize(n_cells, 3);

    for (Eigen::Index cell = 0; cell < n_cells; ++cell) {
        EcmState state;
        state.soc = soc0(cell);
        state.v1 = 0.0;
        state.v2 = 0.0;

        Eigen::Index record = 0;
        for (Eigen::Index step = 0; step < n_steps; ++step) {
            if (step % every == 0) {
                out.voltage(record, cell) = terminal_voltage(
                    params, state, current_a(step), temp_c(step),
                    cap_scale(cell), r0_scale(cell));
                ++record;
            }
            state = ecm_step(params, state, current_a(step), temp_c(step), dt_s,
                             cap_scale(cell), r0_scale(cell));
        }

        out.final_state(cell, 0) = state.soc;
        out.final_state(cell, 1) = state.v1;
        out.final_state(cell, 2) = state.v2;
    }

    return out;
}

}  // namespace cellsight
