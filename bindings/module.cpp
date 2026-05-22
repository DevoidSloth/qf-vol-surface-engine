// pybind11 bindings for the C++ core.
//
// This file is deliberately thin. It contains no pricing logic, no defaults that
// differ from the C++ ones, and no convenience reinterpretation of arguments --
// anything of that kind belongs in python/vsepy/ where it is visible, or in the
// core where it is tested. A binding layer that quietly does something the
// library does not is how two implementations of the same model end up
// disagreeing in the fourth decimal place.
//
// Scalar functions are wrapped with py::vectorize, so every one of them accepts
// NumPy arrays and broadcasts. That matters for the surface layer, which fits
// tens of thousands of quotes at a time and would otherwise pay Python call
// overhead per quote.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "vse/black.hpp"
#include "vse/common.hpp"
#include "vse/implied_vol.hpp"
#include "vse/normal.hpp"

namespace py = pybind11;
using namespace vse;

PYBIND11_MODULE(_vse, m) {
    m.doc() = "Arbitrage-free volatility surface and option pricing core (C++20)";
    m.attr("__version__") = "0.1.0";

    py::register_exception<DomainError>(m, "DomainError", PyExc_ValueError);
    py::register_exception<ConvergenceError>(m, "ConvergenceError", PyExc_RuntimeError);

    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put)
        .export_values();

    // -----------------------------------------------------------------------
    // Normal / error functions
    // -----------------------------------------------------------------------
    m.def("norm_pdf", py::vectorize(norm_pdf), py::arg("x"),
          "Standard normal density.");
    m.def("norm_cdf", py::vectorize(static_cast<Real (*)(Real)>(norm_cdf)), py::arg("x"),
          "Standard normal CDF, relative-accurate into the lower tail.");
    m.def("norm_inv_cdf", py::vectorize(norm_inv_cdf), py::arg("p"),
          "Inverse standard normal CDF.");
    m.def("erf", py::vectorize(erf_), py::arg("x"));
    m.def("erfc", py::vectorize(erfc_), py::arg("x"));
    m.def("erfcx", py::vectorize(erfcx), py::arg("x"),
          "Scaled complementary error function, exp(x^2) erfc(x).");
    m.def("mills_ratio", py::vectorize(mills_ratio), py::arg("z"),
          "N(-z) / phi(z), valid and relative-accurate for every real z.");

    // -----------------------------------------------------------------------
    // Normalised Black
    // -----------------------------------------------------------------------
    m.def("normalised_black", py::vectorize(normalised_black), py::arg("x"), py::arg("s"),
          "Undiscounted call value in units of sqrt(F K), for x = ln(F/K) and\n"
          "s = sigma sqrt(T).");
    m.def("normalised_vega", py::vectorize(normalised_vega), py::arg("x"), py::arg("s"),
          "db/ds for the normalised Black function.");
    m.def("normalised_black_inflection", py::vectorize(normalised_black_inflection),
          py::arg("x"), "s_c = sqrt(2|x|), where d2b/ds2 vanishes.");

    // -----------------------------------------------------------------------
    // Black-76 / Black-Scholes
    // -----------------------------------------------------------------------
    m.def("black76", py::vectorize(black76), py::arg("forward"), py::arg("strike"),
          py::arg("expiry"), py::arg("sigma"), py::arg("discount"), py::arg("type"),
          "Black-76 premium on the forward.");
    m.def("bs_price", py::vectorize(bs_price), py::arg("spot"), py::arg("strike"),
          py::arg("expiry"), py::arg("rate"), py::arg("dividend"), py::arg("sigma"),
          py::arg("type"), "Black-Scholes premium with continuous carry.");

    py::class_<Greeks>(m, "Greeks", "Analytic Black-Scholes sensitivities.")
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
            return "Greeks(price=" + std::to_string(g.price) +
                   ", delta=" + std::to_string(g.delta) +
                   ", gamma=" + std::to_string(g.gamma) +
                   ", vega=" + std::to_string(g.vega) + ", ...)";
        });

    m.def("bs_greeks", &bs_greeks, py::arg("spot"), py::arg("strike"), py::arg("expiry"),
          py::arg("rate"), py::arg("dividend"), py::arg("sigma"), py::arg("type"),
          "Price plus nine analytic sensitivities.");

    // -----------------------------------------------------------------------
    // Implied volatility
    // -----------------------------------------------------------------------
    py::class_<ImpliedVolResult>(m, "ImpliedVolResult")
        .def_readonly("sigma", &ImpliedVolResult::sigma)
        .def_readonly("iterations", &ImpliedVolResult::iterations)
        .def_readonly("converged", &ImpliedVolResult::converged)
        .def("__repr__", [](const ImpliedVolResult& r) {
            return "ImpliedVolResult(sigma=" + std::to_string(r.sigma) +
                   ", iterations=" + std::to_string(r.iterations) +
                   ", converged=" + (r.converged ? "True" : "False") + ")";
        });

    m.def("implied_volatility",
          py::vectorize(implied_volatility),
          py::arg("price"), py::arg("forward"), py::arg("strike"), py::arg("expiry"),
          py::arg("discount"), py::arg("type"),
          "Implied volatility from a quoted premium. Pass the out-of-the-money\n"
          "member of the put/call pair: the in-the-money one carries the same\n"
          "information behind an intrinsic value that destroys it.");

    m.def("implied_volatility_ex", &implied_volatility_ex,
          py::arg("price"), py::arg("forward"), py::arg("strike"), py::arg("expiry"),
          py::arg("discount"), py::arg("type"),
          "As implied_volatility, but also returns the iteration count and the\n"
          "convergence flag.");

    m.def("implied_total_volatility",
          [](Real beta, Real x) { return implied_total_volatility(beta, x); },
          py::arg("beta"), py::arg("x"),
          "Solve b(x, s) = beta for the total volatility s = sigma sqrt(T).");
}
