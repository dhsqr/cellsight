#include "cellsight/ecm.hpp"
#include <cmath>
#include <stdexcept>

namespace cellsight {
namespace {

void require(bool ok, const char* msg)
{
    if (!ok) {
        throw std::invalid_argument(msg);
    }
}

// allow_zero: finite and >= 0. Otherwise finite and > 0.
void check_table(const Lut2D& t, const char* name, bool allow_zero)
{
    const Eigen::MatrixXd& values = t.table();
    for (Eigen::Index i = 0; i < values.size(); ++i) {
        const double value = values.data()[i];
        require(std::isfinite(value), name);
        require(allow_zero ? value >= 0.0 : value > 0.0, name);
    }
}

}  // namespace

void validate(const EcmParams& p)
{
    const Eigen::MatrixXd& ocv = p.ocv.table();
    for (Eigen::Index i = 0; i < ocv.size(); ++i) {
        require(std::isfinite(ocv.data()[i]), "ocv");
    }

    check_table(p.r0, "r0", true);
    check_table(p.r1, "r1", false);
    check_table(p.c1, "c1", false);
    check_table(p.r2, "r2", false);
    check_table(p.c2, "c2", false);

    require(std::isfinite(p.q_ah) && p.q_ah > 0.0, "q_ah");
    require(std::isfinite(p.eta_charge) && p.eta_charge > 0.0 &&
                p.eta_charge <= 1.0,
            "eta_charge");
}

double terminal_voltage(const EcmParams& p,
                        const EcmState& x,
                        double i_a,
                        double temp_c,
                        double cap_scale,
                        double r0_scale)
{
    require(std::isfinite(i_a), "current");
    require(std::isfinite(temp_c), "temp");
    require(std::isfinite(x.soc) && std::isfinite(x.v1) && std::isfinite(x.v2),
            "state");
    require(std::isfinite(cap_scale) && cap_scale > 0.0, "cap_scale");
    require(std::isfinite(r0_scale) && r0_scale > 0.0, "r0_scale");

    const double ocv = p.ocv(x.soc, temp_c);
    const double r0 = p.r0(x.soc, temp_c);
    return ocv - r0 * r0_scale * i_a - x.v1 - x.v2;
}

EcmState ecm_step(const EcmParams& p,
                  const EcmState& x,
                  double i_a,
                  double temp_c,
                  double dt_s,
                  double cap_scale,
                  double r0_scale)
{
    require(std::isfinite(i_a), "current");
    require(std::isfinite(temp_c), "temp");
    require(std::isfinite(x.soc) && std::isfinite(x.v1) && std::isfinite(x.v2),
            "state");
    require(std::isfinite(dt_s) && dt_s > 0.0, "dt");
    require(std::isfinite(cap_scale) && cap_scale > 0.0, "cap_scale");
    require(std::isfinite(r0_scale) && r0_scale > 0.0, "r0_scale");

    const double r1 = p.r1(x.soc, temp_c);
    const double c1 = p.c1(x.soc, temp_c);
    const double r2 = p.r2(x.soc, temp_c);
    const double c2 = p.c2(x.soc, temp_c);

    const double tau1 = r1 * c1;
    const double a1 = std::exp(-dt_s / tau1);
    const double g1 = -std::expm1(-dt_s / tau1);
    const double tau2 = r2 * c2;
    const double a2 = std::exp(-dt_s / tau2);
    const double g2 = -std::expm1(-dt_s / tau2);

    const double eta = (i_a < 0.0) ? p.eta_charge : 1.0;

    EcmState next;
    next.soc = x.soc - eta * i_a * dt_s / (3600.0 * p.q_ah * cap_scale);
    next.v1 = a1 * x.v1 + r1 * g1 * i_a;
    next.v2 = a2 * x.v2 + r2 * g2 * i_a;
    return next;
}

}  // namespace cellsight
