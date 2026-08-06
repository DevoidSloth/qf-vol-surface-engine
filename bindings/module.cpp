// pybind11 bindings for the C++ core.
//
// This file is deliberately thin. It contains no pricing logic, no defaults that
// differ from the C++ ones, and no convenience reinterpretation of arguments --
// anything of that kind belongs in python/vsepy/ where it is visible, or in the
// core where it is tested. A binding layer that quietly does something the
// library does not is how two implementations of the same model end up
// disagreeing in the fourth decimal place.
//
// Scalar functions are wrapped with py::vectorize, so they accept NumPy arrays
// and broadcast. That matters for the surface layer, which fits tens of
// thousands of quotes at a time and would otherwise pay Python call overhead per
// quote.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "vse/aad.hpp"
#include "vse/bates.hpp"
#include "vse/binomial.hpp"
#include "vse/black.hpp"
#include "vse/calibrate_heston.hpp"
#include "vse/calibrate_svi.hpp"
#include "vse/chain.hpp"
#include "vse/common.hpp"
#include "vse/heston.hpp"
#include "vse/implied_vol.hpp"
#include "vse/lsmc.hpp"
#include "vse/mc.hpp"
#include "vse/mc_aad.hpp"
#include "vse/normal.hpp"
#include "vse/pde.hpp"
#include "vse/pde_heston.hpp"
#include "vse/sabr.hpp"
#include "vse/smile_repair.hpp"
#include "vse/svi.hpp"
#include "vse/synthetic.hpp"

namespace py = pybind11;
using namespace vse;

namespace {

/// Expose a plain-data struct as a read-only Python object with a repr.
template <class T>
py::class_<T> value_type(py::module_& m, const char* name, const char* doc) {
    return py::class_<T>(m, name, doc);
}

}  // namespace

PYBIND11_MODULE(_vse, m) {
    m.doc() = "Arbitrage-free volatility surface and option pricing core (C++20)";
    m.attr("__version__") = "0.1.0";

    py::register_exception<DomainError>(m, "DomainError", PyExc_ValueError);
    py::register_exception<ConvergenceError>(m, "ConvergenceError", PyExc_RuntimeError);

    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put)
        .export_values();
    py::enum_<Exercise>(m, "Exercise")
        .value("European", Exercise::European)
        .value("American", Exercise::American);
    py::enum_<HestonScheme>(m, "HestonScheme")
        .value("EulerFullTruncation", HestonScheme::EulerFullTruncation)
        .value("AndersenQE", HestonScheme::AndersenQE);
    py::enum_<Sampling>(m, "Sampling")
        .value("PseudoRandom", Sampling::PseudoRandom)
        .value("SobolBridge", Sampling::SobolBridge);

    // -----------------------------------------------------------------------
    // Normal / error functions
    // -----------------------------------------------------------------------
    m.def("norm_pdf", py::vectorize(norm_pdf), py::arg("x"));
    m.def("norm_cdf", py::vectorize(static_cast<Real (*)(Real)>(norm_cdf)), py::arg("x"),
          "Standard normal CDF, relative-accurate into the lower tail.");
    m.def("norm_inv_cdf", py::vectorize(norm_inv_cdf), py::arg("p"));
    m.def("erf", py::vectorize(erf_), py::arg("x"));
    m.def("erfc", py::vectorize(erfc_), py::arg("x"));
    m.def("erfcx", py::vectorize(erfcx), py::arg("x"),
          "Scaled complementary error function, exp(x^2) erfc(x).");
    m.def("mills_ratio", py::vectorize(mills_ratio), py::arg("z"),
          "N(-z) / phi(z), relative-accurate for every real z.");

    // -----------------------------------------------------------------------
    // Black
    // -----------------------------------------------------------------------
    m.def("normalised_black", py::vectorize(normalised_black), py::arg("x"), py::arg("s"));
    m.def("normalised_vega", py::vectorize(normalised_vega), py::arg("x"), py::arg("s"));
    m.def("normalised_black_inflection", py::vectorize(normalised_black_inflection), py::arg("x"));
    m.def("black76", py::vectorize(black76), py::arg("forward"), py::arg("strike"),
          py::arg("expiry"), py::arg("sigma"), py::arg("discount"), py::arg("type"));
    m.def("bs_price", py::vectorize(bs_price), py::arg("spot"), py::arg("strike"),
          py::arg("expiry"), py::arg("rate"), py::arg("dividend"), py::arg("sigma"),
          py::arg("type"));

    value_type<Greeks>(m, "Greeks", "Analytic Black-Scholes sensitivities.")
        .def_readonly("price", &Greeks::price)
        .def_readonly("delta", &Greeks::delta)
        .def_readonly("gamma", &Greeks::gamma)
        .def_readonly("vega", &Greeks::vega, "d(price)/d(sigma), sigma in absolute units")
        .def_readonly("theta", &Greeks::theta, "d(price)/dt per calendar year")
        .def_readonly("rho", &Greeks::rho)
        .def_readonly("vanna", &Greeks::vanna)
        .def_readonly("volga", &Greeks::volga)
        .def_readonly("dual_delta", &Greeks::dual_delta, "d(price)/dK")
        .def_readonly("dual_gamma", &Greeks::dual_gamma, "d2(price)/dK2")
        .def("__repr__", [](const Greeks& g) {
            return "Greeks(price=" + std::to_string(g.price) + ", delta=" +
                   std::to_string(g.delta) + ", vega=" + std::to_string(g.vega) + ", ...)";
        });
    m.def("bs_greeks", &bs_greeks, py::arg("spot"), py::arg("strike"), py::arg("expiry"),
          py::arg("rate"), py::arg("dividend"), py::arg("sigma"), py::arg("type"));

    value_type<ImpliedVolResult>(m, "ImpliedVolResult", "")
        .def_readonly("sigma", &ImpliedVolResult::sigma)
        .def_readonly("iterations", &ImpliedVolResult::iterations)
        .def_readonly("converged", &ImpliedVolResult::converged)
        .def("__repr__", [](const ImpliedVolResult& r) {
            return "ImpliedVolResult(sigma=" + std::to_string(r.sigma) + ", iterations=" +
                   std::to_string(r.iterations) + ")";
        });
    m.def("implied_volatility", py::vectorize(implied_volatility), py::arg("price"),
          py::arg("forward"), py::arg("strike"), py::arg("expiry"), py::arg("discount"),
          py::arg("type"),
          "Implied volatility from a quoted premium. Pass the out-of-the-money member\n"
          "of the pair: the in-the-money one carries the same information behind an\n"
          "intrinsic value that destroys it.");
    m.def("implied_volatility_ex", &implied_volatility_ex, py::arg("price"), py::arg("forward"),
          py::arg("strike"), py::arg("expiry"), py::arg("discount"), py::arg("type"));
    m.def("implied_total_volatility",
          [](Real beta, Real x) { return implied_total_volatility(beta, x); },
          py::arg("beta"), py::arg("x"));

    // -----------------------------------------------------------------------
    // Surface
    // -----------------------------------------------------------------------
    value_type<SVIRaw>(m, "SVIRaw", "Raw SVI slice: a + b[rho(k-m) + sqrt((k-m)^2 + sigma^2)]")
        .def(py::init<>())
        .def_readwrite("a", &SVIRaw::a)
        .def_readwrite("b", &SVIRaw::b)
        .def_readwrite("rho", &SVIRaw::rho)
        .def_readwrite("m", &SVIRaw::m)
        .def_readwrite("sigma", &SVIRaw::sigma)
        .def("total_variance", &SVIRaw::total_variance, py::arg("k"))
        .def("implied_vol", &SVIRaw::implied_vol, py::arg("k"), py::arg("expiry"))
        .def("dw", &SVIRaw::dw, py::arg("k"))
        .def("d2w", &SVIRaw::d2w, py::arg("k"))
        .def("min_variance", &SVIRaw::min_variance)
        .def("left_slope", &SVIRaw::left_slope)
        .def("right_slope", &SVIRaw::right_slope)
        .def("is_well_formed", &SVIRaw::is_well_formed);

    value_type<SSVISlice>(m, "SSVISlice", "One SSVI slice at a given ATM total variance.")
        .def(py::init<>())
        .def_readwrite("theta", &SSVISlice::theta)
        .def_readwrite("rho", &SSVISlice::rho)
        .def_readwrite("phi", &SSVISlice::phi)
        .def("total_variance", &SSVISlice::total_variance, py::arg("k"))
        .def("implied_vol", &SSVISlice::implied_vol, py::arg("k"), py::arg("expiry"))
        .def("to_raw", &SSVISlice::to_raw);

    value_type<PowerLawPhi>(m, "PowerLawPhi", "phi(theta) = eta / (theta^gamma (1+theta)^(1-gamma))")
        .def(py::init<>())
        .def_readwrite("eta", &PowerLawPhi::eta)
        .def_readwrite("gamma", &PowerLawPhi::gamma)
        .def("__call__", &PowerLawPhi::operator(), py::arg("theta"));

    value_type<SSVISurface>(m, "SSVISurface", "")
        .def(py::init<>())
        .def_readwrite("phi", &SSVISurface::phi)
        .def_readwrite("rho", &SSVISurface::rho)
        .def_readwrite("expiries", &SSVISurface::expiries)
        .def_readwrite("theta", &SSVISurface::theta)
        .def("slice_at", &SSVISurface::slice_at, py::arg("expiry"))
        .def("total_variance", &SSVISurface::total_variance, py::arg("k"), py::arg("expiry"))
        .def("implied_vol", &SSVISurface::implied_vol, py::arg("k"), py::arg("expiry"));

    value_type<ESSVISurface>(m, "ESSVISurface", "")
        .def(py::init<>())
        .def_readwrite("expiries", &ESSVISurface::expiries)
        .def_readwrite("theta", &ESSVISurface::theta)
        .def_readwrite("rho", &ESSVISurface::rho)
        .def_readwrite("psi", &ESSVISurface::psi)
        .def("slice_at_index", &ESSVISurface::slice_at_index, py::arg("index"))
        .def("total_variance", &ESSVISurface::total_variance, py::arg("k"), py::arg("expiry"))
        .def("calendar_conditions_hold", &ESSVISurface::calendar_conditions_hold,
             py::arg("tolerance") = 1e-12);

    value_type<ButterflyReport>(m, "ButterflyReport", "")
        .def_readonly("free", &ButterflyReport::free)
        .def_readonly("min_g", &ButterflyReport::min_g)
        .def_readonly("k_at_min", &ButterflyReport::k_at_min)
        .def_readonly("min_density", &ButterflyReport::min_density)
        .def_readonly("density_integral", &ButterflyReport::density_integral)
        .def_readonly("violations", &ButterflyReport::violations);
    value_type<CalendarReport>(m, "CalendarReport", "")
        .def_readonly("free", &CalendarReport::free)
        .def_readonly("worst_decrease", &CalendarReport::worst_decrease)
        .def_readonly("k_at_worst", &CalendarReport::k_at_worst)
        .def_readonly("t_at_worst", &CalendarReport::t_at_worst)
        .def_readonly("violations", &CalendarReport::violations);
    value_type<SSVIConditionReport>(m, "SSVIConditionReport", "Gatheral-Jacquier Theorem 4.2")
        .def_readonly("butterfly_free", &SSVIConditionReport::butterfly_free)
        .def_readonly("calendar_free", &SSVIConditionReport::calendar_free)
        .def_readonly("bf_condition_1", &SSVIConditionReport::bf_condition_1)
        .def_readonly("bf_condition_2", &SSVIConditionReport::bf_condition_2)
        .def_readonly("cal_lower", &SSVIConditionReport::cal_lower)
        .def_readonly("cal_upper_bound", &SSVIConditionReport::cal_upper_bound);

    m.def("durrleman_g", [](const SVIRaw& s, Real k) { return durrleman_g(s, k); },
          py::arg("slice"), py::arg("k"),
          "Gatheral's g(k); non-negative everywhere iff the density is.");
    m.def("risk_neutral_density",
          [](const SVIRaw& s, Real k) { return risk_neutral_density(s, k); },
          py::arg("slice"), py::arg("k"));
    m.def("durrleman_g_ssvi", [](const SSVISlice& s, Real k) { return durrleman_g(s, k); },
          py::arg("slice"), py::arg("k"));
    m.def("risk_neutral_density_ssvi",
          [](const SSVISlice& s, Real k) { return risk_neutral_density(s, k); },
          py::arg("slice"), py::arg("k"));
    m.def("check_butterfly",
          [](const SVIRaw& s, Real expiry, Real half_width, int n) {
              return check_butterfly(s, expiry, half_width, n);
          },
          py::arg("slice"), py::arg("expiry"), py::arg("k_half_width") = 0.0,
          py::arg("points") = 2001);
    m.def("check_butterfly_ssvi",
          [](const SSVISlice& s, Real expiry, Real half_width, int n) {
              return check_butterfly(s, expiry, half_width, n);
          },
          py::arg("slice"), py::arg("expiry"), py::arg("k_half_width") = 0.0,
          py::arg("points") = 2001);
    m.def("check_calendar",
          [](const SSVISurface& s, const std::vector<Real>& times, Real hw, int n) {
              return check_calendar(s, times, hw, n);
          },
          py::arg("surface"), py::arg("times"), py::arg("k_half_width") = 1.5,
          py::arg("points") = 401);

    // -----------------------------------------------------------------------
    // Chain processing
    // -----------------------------------------------------------------------
    value_type<RawQuote>(m, "RawQuote", "")
        .def(py::init<>())
        .def_readwrite("strike", &RawQuote::strike)
        .def_readwrite("bid", &RawQuote::bid)
        .def_readwrite("ask", &RawQuote::ask)
        .def_readwrite("volume", &RawQuote::volume)
        .def_readwrite("open_interest", &RawQuote::open_interest)
        .def_readwrite("type", &RawQuote::type)
        .def("mid", &RawQuote::mid)
        .def("spread", &RawQuote::spread);

    value_type<ForwardFit>(m, "ForwardFit", "")
        .def_readonly("forward", &ForwardFit::forward)
        .def_readonly("discount", &ForwardFit::discount)
        .def_readonly("implied_rate", &ForwardFit::implied_rate)
        .def_readonly("pairs_used", &ForwardFit::pairs_used)
        .def_readonly("residual_rms", &ForwardFit::residual_rms)
        .def_readonly("worst_residual", &ForwardFit::worst_residual)
        .def_readonly("ok", &ForwardFit::ok);

    value_type<SurfacePoint>(m, "SurfacePoint", "")
        .def_readonly("log_moneyness", &SurfacePoint::log_moneyness)
        .def_readonly("strike", &SurfacePoint::strike)
        .def_readonly("implied_vol", &SurfacePoint::implied_vol)
        .def_readonly("total_variance", &SurfacePoint::total_variance)
        .def_readonly("vega", &SurfacePoint::vega)
        .def_readonly("spread_vol", &SurfacePoint::spread_vol)
        .def_readonly("weight", &SurfacePoint::weight)
        .def_readonly("type", &SurfacePoint::type);

    value_type<FilterConfig>(m, "FilterConfig", "")
        .def(py::init<>())
        .def_readwrite("min_price", &FilterConfig::min_price)
        .def_readwrite("max_relative_spread", &FilterConfig::max_relative_spread)
        .def_readwrite("max_abs_log_moneyness", &FilterConfig::max_abs_log_moneyness)
        .def_readwrite("min_volume", &FilterConfig::min_volume)
        .def_readwrite("min_open_interest", &FilterConfig::min_open_interest)
        .def_readwrite("require_two_sided", &FilterConfig::require_two_sided);

    value_type<SliceBuildReport>(m, "SliceBuildReport", "")
        .def_readonly("input_quotes", &SliceBuildReport::input_quotes)
        .def_readonly("dropped_in_the_money", &SliceBuildReport::dropped_in_the_money)
        .def_readonly("dropped_one_sided", &SliceBuildReport::dropped_one_sided)
        .def_readonly("dropped_cheap", &SliceBuildReport::dropped_cheap)
        .def_readonly("dropped_wide", &SliceBuildReport::dropped_wide)
        .def_readonly("dropped_moneyness", &SliceBuildReport::dropped_moneyness)
        .def_readonly("dropped_liquidity", &SliceBuildReport::dropped_liquidity)
        .def_readonly("dropped_arbitrage", &SliceBuildReport::dropped_arbitrage)
        .def_readonly("kept", &SliceBuildReport::kept)
        .def("dropped_total", &SliceBuildReport::dropped_total)
        .def("balances", &SliceBuildReport::balances,
             "Every input quote landed in exactly one bucket.");

    m.def("implied_forward_from_parity", &implied_forward_from_parity, py::arg("calls"),
          py::arg("puts"), py::arg("expiry"), py::arg("anchor"),
          py::arg("moneyness_window") = 0.10,
          "Recover F and P(0,T) together by regressing C - P on K.");
    m.def("build_slice",
          [](const std::vector<RawQuote>& q, Real f, Real t, Real df, const FilterConfig& c) {
              SliceBuildReport report;
              auto slice = build_slice(q, f, t, df, c, &report);
              return py::make_tuple(std::move(slice), report);
          },
          py::arg("quotes"), py::arg("forward"), py::arg("expiry"), py::arg("discount"),
          py::arg("filters") = FilterConfig{},
          "Returns (points, report).");
    m.def("atm_total_variance", &atm_total_variance, py::arg("slice"));

    // -----------------------------------------------------------------------
    // Synthetic data
    // -----------------------------------------------------------------------
    value_type<SyntheticChainConfig>(m, "SyntheticChainConfig", "")
        .def(py::init<>())
        .def_readwrite("spot", &SyntheticChainConfig::spot)
        .def_readwrite("quoted_rate", &SyntheticChainConfig::quoted_rate)
        .def_readwrite("borrow_spread", &SyntheticChainConfig::borrow_spread)
        .def_readwrite("dividend_yield", &SyntheticChainConfig::dividend_yield)
        .def_readwrite("tick", &SyntheticChainConfig::tick)
        .def_readwrite("seed", &SyntheticChainConfig::seed)
        .def_readwrite("round_to_tick", &SyntheticChainConfig::round_to_tick)
        .def_readwrite("add_quote_noise", &SyntheticChainConfig::add_quote_noise)
        .def_readwrite("quote_noise_vol_points", &SyntheticChainConfig::quote_noise_vol_points);

    value_type<SyntheticExpiry>(m, "SyntheticExpiry", "")
        .def_readonly("expiry", &SyntheticExpiry::expiry)
        .def_readonly("true_forward", &SyntheticExpiry::true_forward)
        .def_readonly("true_discount", &SyntheticExpiry::true_discount)
        .def_readonly("quotes", &SyntheticExpiry::quotes);
    value_type<SyntheticTruth>(m, "SyntheticTruth", "")
        .def_readonly("surface", &SyntheticTruth::surface)
        .def_readonly("expiries", &SyntheticTruth::expiries)
        .def_readonly("forwards", &SyntheticTruth::forwards)
        .def_readonly("discounts", &SyntheticTruth::discounts);
    value_type<SyntheticChain>(m, "SyntheticChain", "")
        .def_readonly("truth", &SyntheticChain::truth)
        .def_readonly("expiries", &SyntheticChain::expiries);

    m.def("generate_synthetic_chain", &generate_synthetic_chain,
          py::arg("expiries") = std::vector<Real>{7.0 / 365, 30.0 / 365, 60.0 / 365,
                                                  91.0 / 365, 182.0 / 365, 1.0, 2.0},
          py::arg("config") = SyntheticChainConfig{},
          "MANUFACTURED quotes with a known ground truth. Not market data.");
    m.def("split_by_type",
          [](const std::vector<RawQuote>& q) {
              std::vector<RawQuote> calls, puts;
              split_by_type(q, calls, puts);
              return py::make_tuple(calls, puts);
          },
          py::arg("quotes"));

    // -----------------------------------------------------------------------
    // Calibration
    // -----------------------------------------------------------------------
    value_type<SVIFitResult>(m, "SVIFitResult", "")
        .def_readonly("params", &SVIFitResult::params)
        .def_readonly("rmse_vol", &SVIFitResult::rmse_vol)
        .def_readonly("max_error_vol", &SVIFitResult::max_error_vol)
        .def_readonly("iterations", &SVIFitResult::iterations)
        .def_readonly("converged", &SVIFitResult::converged)
        .def_readonly("butterfly", &SVIFitResult::butterfly)
        .def_readonly("message", &SVIFitResult::message);
    value_type<SSVIFitResult>(m, "SSVIFitResult", "")
        .def_readonly("surface", &SSVIFitResult::surface)
        .def_readonly("rmse_vol", &SSVIFitResult::rmse_vol)
        .def_readonly("max_error_vol", &SSVIFitResult::max_error_vol)
        .def_readonly("quotes", &SSVIFitResult::quotes)
        .def_readonly("converged", &SSVIFitResult::converged)
        .def_readonly("conditions", &SSVIFitResult::conditions)
        .def_readonly("calendar", &SSVIFitResult::calendar)
        .def_readonly("butterfly", &SSVIFitResult::butterfly)
        .def_readonly("message", &SSVIFitResult::message);
    value_type<ESSVIFitResult>(m, "ESSVIFitResult", "")
        .def_readonly("surface", &ESSVIFitResult::surface)
        .def_readonly("rmse_vol", &ESSVIFitResult::rmse_vol)
        .def_readonly("max_error_vol", &ESSVIFitResult::max_error_vol)
        .def_readonly("quotes", &ESSVIFitResult::quotes)
        .def_readonly("calendar_conditions_hold", &ESSVIFitResult::calendar_conditions_hold)
        .def_readonly("butterfly", &ESSVIFitResult::butterfly)
        .def_readonly("message", &ESSVIFitResult::message);

    m.def("fit_svi_slice",
          [](const std::vector<SurfacePoint>& pts, Real t) { return fit_svi_slice(pts, t); },
          py::arg("points"), py::arg("expiry"));
    m.def("fit_ssvi",
          [](const std::vector<std::vector<SurfacePoint>>& s, const std::vector<Real>& e,
             const std::vector<Real>& th) { return fit_ssvi(s, e, th); },
          py::arg("slices"), py::arg("expiries"), py::arg("theta"));
    m.def("fit_essvi",
          [](const std::vector<std::vector<SurfacePoint>>& s, const std::vector<Real>& e,
             const std::vector<Real>& th) { return fit_essvi(s, e, th); },
          py::arg("slices"), py::arg("expiries"), py::arg("theta"));

    // -----------------------------------------------------------------------
    // Heston, Bates, SABR
    // -----------------------------------------------------------------------
    value_type<HestonParams>(m, "HestonParams", "")
        .def(py::init<>())
        .def(py::init([](Real v0, Real kappa, Real theta, Real sigma, Real rho) {
                 return HestonParams{v0, kappa, theta, sigma, rho};
             }),
             py::arg("v0"), py::arg("kappa"), py::arg("theta"), py::arg("sigma"), py::arg("rho"))
        .def_readwrite("v0", &HestonParams::v0)
        .def_readwrite("kappa", &HestonParams::kappa)
        .def_readwrite("theta", &HestonParams::theta)
        .def_readwrite("sigma", &HestonParams::sigma)
        .def_readwrite("rho", &HestonParams::rho)
        .def("feller_ratio", &HestonParams::feller_ratio)
        .def("satisfies_feller", &HestonParams::satisfies_feller)
        .def("__repr__", [](const HestonParams& p) {
            return "HestonParams(v0=" + std::to_string(p.v0) + ", kappa=" +
                   std::to_string(p.kappa) + ", theta=" + std::to_string(p.theta) +
                   ", sigma=" + std::to_string(p.sigma) + ", rho=" + std::to_string(p.rho) + ")";
        });

    m.def("heston_call_lewis", &heston_call_lewis, py::arg("params"), py::arg("forward"),
          py::arg("strike"), py::arg("expiry"), py::arg("order") = 32, py::arg("panels") = 16,
          "Undiscounted European call by the Lewis integral.");
    m.def("heston_price", &heston_price, py::arg("params"), py::arg("forward"),
          py::arg("strike"), py::arg("expiry"), py::arg("discount"), py::arg("type"),
          py::arg("order") = 32, py::arg("panels") = 16);
    m.def("heston_cf",
          [](const HestonParams& p, Real t, std::complex<Real> u) { return heston_cf(p, t, u); },
          py::arg("params"), py::arg("expiry"), py::arg("u"));

    value_type<HestonPriceGradient>(m, "HestonPriceGradient", "")
        .def_readonly("price", &HestonPriceGradient::price)
        .def_property_readonly("gradient", [](const HestonPriceGradient& g) {
            return std::vector<Real>(g.gradient.begin(), g.gradient.end());
        });
    m.def("heston_call_and_gradient", &heston_call_and_gradient, py::arg("params"),
          py::arg("forward"), py::arg("strike"), py::arg("expiry"), py::arg("order") = 32,
          py::arg("panels") = 16,
          "Price and its exact gradient in (v0, kappa, theta, sigma, rho).");

    value_type<CarrMadanConfig>(m, "CarrMadanConfig", "")
        .def(py::init<>())
        .def_readwrite("n", &CarrMadanConfig::n)
        .def_readwrite("eta", &CarrMadanConfig::eta)
        .def_readwrite("alpha", &CarrMadanConfig::alpha);
    value_type<CarrMadanGrid>(m, "CarrMadanGrid", "")
        .def_readonly("log_strikes", &CarrMadanGrid::log_strikes)
        .def_readonly("calls", &CarrMadanGrid::calls)
        .def_readonly("strike_spacing", &CarrMadanGrid::strike_spacing)
        .def("call_at", &CarrMadanGrid::call_at, py::arg("strike"));
    m.def("heston_carr_madan", &heston_carr_madan, py::arg("params"), py::arg("forward"),
          py::arg("expiry"), py::arg("config") = CarrMadanConfig{});

    value_type<CalibrationQuote>(m, "CalibrationQuote", "")
        .def(py::init([](Real f, Real k, Real t, Real v, Real w) {
                 return CalibrationQuote{f, k, t, v, w};
             }),
             py::arg("forward"), py::arg("strike"), py::arg("expiry"), py::arg("implied_vol"),
             py::arg("weight") = 1.0)
        .def_readwrite("forward", &CalibrationQuote::forward)
        .def_readwrite("strike", &CalibrationQuote::strike)
        .def_readwrite("expiry", &CalibrationQuote::expiry)
        .def_readwrite("implied_vol", &CalibrationQuote::implied_vol)
        .def_readwrite("weight", &CalibrationQuote::weight);
    value_type<HestonFitResult>(m, "HestonFitResult", "")
        .def_readonly("params", &HestonFitResult::params)
        .def_readonly("rmse_vol", &HestonFitResult::rmse_vol)
        .def_readonly("max_error_vol", &HestonFitResult::max_error_vol)
        .def_readonly("quotes", &HestonFitResult::quotes)
        .def_readonly("iterations", &HestonFitResult::iterations)
        .def_readonly("slice_builds", &HestonFitResult::slice_builds)
        .def_readonly("converged", &HestonFitResult::converged)
        .def_readonly("message", &HestonFitResult::message);
    m.def("calibrate_heston",
          [](const std::vector<CalibrationQuote>& q, const HestonParams& start) {
              return calibrate_heston(q, start);
          },
          py::arg("quotes"), py::arg("start") = HestonParams{});
    m.def("quotes_from_slices", &quotes_from_slices, py::arg("slices"), py::arg("expiries"),
          py::arg("forwards"));

    value_type<BatesParams>(m, "BatesParams", "")
        .def(py::init<>())
        .def_readwrite("heston", &BatesParams::heston)
        .def_readwrite("lambda_", &BatesParams::lambda)
        .def_readwrite("jump_mean", &BatesParams::jump_mean)
        .def_readwrite("jump_vol", &BatesParams::jump_vol)
        .def("expected_jump", &BatesParams::expected_jump);
    m.def("bates_price", &bates_price, py::arg("params"), py::arg("forward"), py::arg("strike"),
          py::arg("expiry"), py::arg("discount"), py::arg("type"), py::arg("order") = 32,
          py::arg("panels") = 16);

    value_type<SABRParams>(m, "SABRParams", "")
        .def(py::init([](Real alpha, Real beta, Real rho, Real nu, Real shift) {
                 return SABRParams{alpha, beta, rho, nu, shift};
             }),
             py::arg("alpha") = 0.2, py::arg("beta") = 0.5, py::arg("rho") = -0.3,
             py::arg("nu") = 0.4, py::arg("shift") = 0.0)
        .def_readwrite("alpha", &SABRParams::alpha)
        .def_readwrite("beta", &SABRParams::beta)
        .def_readwrite("rho", &SABRParams::rho)
        .def_readwrite("nu", &SABRParams::nu)
        .def_readwrite("shift", &SABRParams::shift)
        .def("__repr__", [](const SABRParams& p) {
            return "SABRParams(alpha=" + std::to_string(p.alpha) + ", beta=" +
                   std::to_string(p.beta) + ", rho=" + std::to_string(p.rho) +
                   ", nu=" + std::to_string(p.nu) + ")";
        });
    value_type<SABRArbitrageReport>(m, "SABRArbitrageReport", "")
        .def_readonly("free", &SABRArbitrageReport::free)
        .def_readonly("arbitrage_boundary", &SABRArbitrageReport::arbitrage_boundary)
        .def_readonly("min_density", &SABRArbitrageReport::min_density)
        .def_readonly("strike_at_min", &SABRArbitrageReport::strike_at_min)
        .def_readonly("violations", &SABRArbitrageReport::violations)
        .def_readonly("points", &SABRArbitrageReport::points);
    m.def("sabr_lognormal_vol", py::vectorize(sabr_lognormal_vol_scalar), py::arg("alpha"),
          py::arg("beta"), py::arg("rho"), py::arg("nu"), py::arg("shift"), py::arg("forward"),
          py::arg("strike"), py::arg("expiry"));
    m.def("sabr_normal_vol", py::vectorize(sabr_normal_vol_scalar), py::arg("alpha"),
          py::arg("beta"), py::arg("rho"), py::arg("nu"), py::arg("shift"), py::arg("forward"),
          py::arg("strike"), py::arg("expiry"));
    m.def("sabr_alpha_from_atm", &sabr_alpha_from_atm, py::arg("atm_vol"), py::arg("forward"),
          py::arg("expiry"), py::arg("beta"), py::arg("rho"), py::arg("nu"),
          py::arg("shift") = 0.0);
    value_type<SmileRepairReport>(m, "SmileRepairReport", "")
        .def_readonly("min_density_before", &SmileRepairReport::min_density_before)
        .def_readonly("min_density_after", &SmileRepairReport::min_density_after)
        .def_readonly("violations_before", &SmileRepairReport::violations_before)
        .def_readonly("violations_after", &SmileRepairReport::violations_after)
        .def_readonly("max_vol_change", &SmileRepairReport::max_vol_change)
        .def_readonly("max_vol_change_strike", &SmileRepairReport::max_vol_change_strike)
        .def_readonly("max_price_change", &SmileRepairReport::max_price_change)
        .def_readonly("density_tolerance", &SmileRepairReport::density_tolerance)
        .def_readonly("mass_before", &SmileRepairReport::mass_before)
        .def_readonly("mass_after", &SmileRepairReport::mass_after)
        .def_readonly("points", &SmileRepairReport::points)
        .def_readonly("repaired", &SmileRepairReport::repaired);
    value_type<RepairedSmile>(m, "RepairedSmile",
                              "A smile projected onto the nearest arbitrage-free one.")
        .def_readonly("strikes", &RepairedSmile::strikes)
        .def_readonly("otm_price", &RepairedSmile::otm_price)
        .def_readonly("implied_vol", &RepairedSmile::implied_vol)
        .def_readonly("input_vol", &RepairedSmile::input_vol)
        .def_readonly("density", &RepairedSmile::density)
        .def_readonly("forward", &RepairedSmile::forward)
        .def_readonly("expiry", &RepairedSmile::expiry)
        .def_readonly("report", &RepairedSmile::report)
        .def("vol_at", &RepairedSmile::vol_at, py::arg("strike"));
    m.def("repair_sabr", &repair_sabr, py::arg("params"), py::arg("forward"), py::arg("expiry"),
          py::arg("k_min_ratio") = 0.02, py::arg("k_max_ratio") = 3.0, py::arg("points") = 1501,
          py::arg("normal_form") = false,
          "Project a SABR smile onto the nearest one that is a distribution.");
    m.def("repair_smile",
          [](const std::function<Real(Real)>& vol_at, Real forward, Real expiry, Real lo, Real hi,
             int n) { return repair_smile(vol_at, forward, expiry, lo, hi, n); },
          py::arg("vol_at"), py::arg("forward"), py::arg("expiry"), py::arg("k_min_ratio") = 0.02,
          py::arg("k_max_ratio") = 3.0, py::arg("points") = 1501,
          "Project any smile, given as a callable from strike to implied vol.");
    m.def("isotonic_increasing",
          [](std::vector<Real> y) { isotonic_increasing(y); return y; }, py::arg("values"),
          "L2 projection onto the non-decreasing sequences (pool adjacent violators).");

    m.def("sabr_density_scan", &sabr_density_scan, py::arg("params"), py::arg("forward"),
          py::arg("expiry"), py::arg("k_min_ratio") = 0.02, py::arg("k_max_ratio") = 3.0,
          py::arg("points") = 1500, py::arg("normal_form") = false);

    // -----------------------------------------------------------------------
    // PDE, Monte Carlo, LSMC
    // -----------------------------------------------------------------------
    value_type<PDEConfig>(m, "PDEConfig", "")
        .def(py::init<>())
        .def_readwrite("space_steps", &PDEConfig::space_steps)
        .def_readwrite("time_steps", &PDEConfig::time_steps)
        .def_readwrite("grid_width_sd", &PDEConfig::grid_width_sd)
        .def_readwrite("rannacher_steps", &PDEConfig::rannacher_steps)
        .def_readwrite("theta", &PDEConfig::theta)
        .def_readwrite("use_brennan_schwartz", &PDEConfig::use_brennan_schwartz);
    value_type<PDEResult>(m, "PDEResult", "")
        .def_readonly("price", &PDEResult::price)
        .def_readonly("delta", &PDEResult::delta)
        .def_readonly("gamma", &PDEResult::gamma)
        .def_readonly("psor_iterations", &PDEResult::psor_iterations)
        .def_readonly("spots", &PDEResult::spots)
        .def_readonly("values", &PDEResult::values);
    m.def("pde_vanilla", &pde_vanilla, py::arg("spot"), py::arg("strike"), py::arg("expiry"),
          py::arg("rate"), py::arg("dividend"), py::arg("sigma"), py::arg("type"),
          py::arg("exercise") = Exercise::European, py::arg("config") = PDEConfig{});

    value_type<HestonPDEConfig>(m, "HestonPDEConfig", "")
        .def(py::init<>())
        .def_readwrite("spot_steps", &HestonPDEConfig::spot_steps)
        .def_readwrite("var_steps", &HestonPDEConfig::var_steps)
        .def_readwrite("time_steps", &HestonPDEConfig::time_steps)
        .def_readwrite("craig_sneyd", &HestonPDEConfig::craig_sneyd);
    value_type<HestonPDEResult>(m, "HestonPDEResult", "")
        .def_readonly("price", &HestonPDEResult::price)
        .def_readonly("delta", &HestonPDEResult::delta)
        .def_readonly("gamma", &HestonPDEResult::gamma);
    m.def("heston_pde", &heston_pde, py::arg("params"), py::arg("spot"), py::arg("strike"),
          py::arg("expiry"), py::arg("rate"), py::arg("dividend"), py::arg("type"),
          py::arg("config") = HestonPDEConfig{});

    m.def("binomial_crr", &binomial_crr, py::arg("spot"), py::arg("strike"), py::arg("expiry"),
          py::arg("rate"), py::arg("dividend"), py::arg("sigma"), py::arg("type"),
          py::arg("american"), py::arg("steps"));
    m.def("binomial_leisen_reimer", &binomial_leisen_reimer, py::arg("spot"), py::arg("strike"),
          py::arg("expiry"), py::arg("rate"), py::arg("dividend"), py::arg("sigma"),
          py::arg("type"), py::arg("american"), py::arg("steps"));

    value_type<MCConfig>(m, "MCConfig", "")
        .def(py::init<>())
        .def_readwrite("paths", &MCConfig::paths)
        .def_readwrite("steps", &MCConfig::steps)
        .def_readwrite("scheme", &MCConfig::scheme)
        .def_readwrite("sampling", &MCConfig::sampling)
        .def_readwrite("antithetic", &MCConfig::antithetic)
        .def_readwrite("control_variate", &MCConfig::control_variate)
        .def_readwrite("martingale_correction", &MCConfig::martingale_correction)
        .def_readwrite("conditional", &MCConfig::conditional)
        .def_readwrite("seed", &MCConfig::seed)
        .def_readwrite("qmc_replications", &MCConfig::qmc_replications);
    value_type<MCResult>(m, "MCResult", "")
        .def_readonly("price", &MCResult::price)
        .def_readonly("standard_error", &MCResult::standard_error)
        .def_readonly("raw_price", &MCResult::raw_price)
        .def_readonly("raw_standard_error", &MCResult::raw_standard_error)
        .def_readonly("control_r_squared", &MCResult::control_r_squared)
        .def_readonly("forward_error", &MCResult::forward_error)
        .def_readonly("paths", &MCResult::paths)
        .def_readonly("replications", &MCResult::replications);
    m.def("heston_mc", &heston_mc, py::arg("params"), py::arg("spot"), py::arg("strike"),
          py::arg("expiry"), py::arg("rate"), py::arg("dividend"), py::arg("type"),
          py::arg("config") = MCConfig{});

    value_type<MCGreeksResult>(m, "MCGreeksResult", "")
        .def_readonly("price", &MCGreeksResult::price)
        .def_readonly("standard_error", &MCGreeksResult::standard_error)
        .def_readonly("paths", &MCGreeksResult::paths)
        .def_readonly("tape_nodes_per_path", &MCGreeksResult::tape_nodes_per_path)
        .def_property_readonly("gradient", [](const MCGreeksResult& r) {
            return std::vector<Real>(r.gradient.begin(), r.gradient.end());
        })
        .def_property_readonly("gradient_se", [](const MCGreeksResult& r) {
            return std::vector<Real>(r.gradient_se.begin(), r.gradient_se.end());
        });
    m.def("heston_mc_greeks_aad", &heston_mc_greeks_aad, py::arg("params"), py::arg("spot"),
          py::arg("strike"), py::arg("expiry"), py::arg("rate"), py::arg("dividend"),
          py::arg("type"), py::arg("config") = MCConfig{});
    m.def("heston_mc_greeks_bump",
          [](const HestonParams& p, Real s, Real k, Real t, Real r, Real q, OptionType type,
             const MCConfig& c) { return heston_mc_greeks_bump(p, s, k, t, r, q, type, c); },
          py::arg("params"), py::arg("spot"), py::arg("strike"), py::arg("expiry"),
          py::arg("rate"), py::arg("dividend"), py::arg("type"), py::arg("config") = MCConfig{});
    m.def("heston_risk_factor_names", [] {
        const auto& n = heston_risk_factor_names();
        return std::vector<std::string>(n.begin(), n.end());
    });

    value_type<LSMCConfig>(m, "LSMCConfig", "")
        .def(py::init<>())
        .def_readwrite("paths", &LSMCConfig::paths)
        .def_readwrite("training_paths", &LSMCConfig::training_paths)
        .def_readwrite("exercise_dates", &LSMCConfig::exercise_dates)
        .def_readwrite("basis_degree", &LSMCConfig::basis_degree)
        .def_readwrite("run_dual", &LSMCConfig::run_dual)
        .def_readwrite("dual_outer_paths", &LSMCConfig::dual_outer_paths)
        .def_readwrite("dual_inner_paths", &LSMCConfig::dual_inner_paths)
        .def_readwrite("seed", &LSMCConfig::seed);
    value_type<LSMCResult>(m, "LSMCResult", "")
        .def_readonly("lower", &LSMCResult::lower)
        .def_readonly("lower_se", &LSMCResult::lower_se)
        .def_readonly("upper", &LSMCResult::upper)
        .def_readonly("upper_se", &LSMCResult::upper_se)
        .def_readonly("duality_gap", &LSMCResult::duality_gap)
        .def_readonly("point_estimate", &LSMCResult::point_estimate)
        .def_readonly("dual_run", &LSMCResult::dual_run)
        .def_readonly("control_correlation", &LSMCResult::control_correlation);
    m.def("lsmc_american", &lsmc_american, py::arg("spot"), py::arg("strike"), py::arg("expiry"),
          py::arg("rate"), py::arg("dividend"), py::arg("sigma"), py::arg("type"),
          py::arg("config") = LSMCConfig{});
}
