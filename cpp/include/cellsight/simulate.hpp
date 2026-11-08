#pragma once

#include <cellsight/ecm.hpp>

#include <Eigen/Dense>

namespace cellsight {

// N independent cells that share one current profile and one temperature
// profile. This is not an electrically connected pack.
//
// Each cell starts at {soc0, 0, 0}. At step k the terminal voltage is
// recorded before ecm_step, and only when k % record_every == 0.
// record_every does not skip steps. final_state always has one row per
// cell after all T steps, with columns soc, v1, v2.
//
// voltage is (n_records, n_cells), where
// n_records = ceil(T / record_every). final_state is (n_cells, 3).
//
// Both products, n_records * n_cells and n_cells * 3, must fit in
// Eigen::Index. Their byte sizes, including both matrices together, must
// fit in size_t. simulate throws std::invalid_argument before allocating
// when that check fails. It does not truncate the workload.
struct SimulateResult {
    Eigen::MatrixXd voltage;
    Eigen::MatrixXd final_state;
};

SimulateResult simulate(
    const EcmParams& params,
    const Eigen::VectorXd& current_a,
    const Eigen::VectorXd& temp_c,
    const Eigen::VectorXd& soc0,
    const Eigen::VectorXd& cap_scale,
    const Eigen::VectorXd& r0_scale,
    double dt_s,
    int record_every);

}  // namespace cellsight
