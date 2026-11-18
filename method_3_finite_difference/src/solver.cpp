// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace {
enum class Method { explicit_euler, implicit_euler, crank_nicolson };

Method parse_method(const std::string& name) {
    if (name == "explicit_euler") return Method::explicit_euler;
    if (name == "implicit_euler") return Method::implicit_euler;
    if (name == "crank_nicolson") return Method::crank_nicolson;
    Rcpp::stop("Unknown finite-difference method: %s", name.c_str());
}

struct Stencil {
    double left, diffusion, right;
};

Stencil spatial_stencil(const Rcpp::NumericVector& initial, double a,
                        double D, double T, int steps, Method method) {
    if (initial.size() < 3 || initial.size() >= std::numeric_limits<int>::max())
        Rcpp::stop("The initial grid needs two endpoints and at least one interior node.");
    if (!std::isfinite(a) || !std::isfinite(D) || !std::isfinite(T) ||
        D <= 0 || T <= 0 || steps <= 0)
        Rcpp::stop("Require finite a, positive finite D and T, and positive n.");
    for (double value : initial)
        if (!std::isfinite(value)) Rcpp::stop("The initial grid must be finite.");
    // sin(pi) can leave a rounding residue at the right endpoint.
    if (std::abs(initial[0]) > 1e-12 || std::abs(initial[initial.size()-1]) > 1e-12)
        Rcpp::stop("Initial endpoints must agree with the zero boundary.");

    const double h = 1.0 / (initial.size() - 1);
    const double dt = T / steps;
    const double r = D * dt / (h*h);
    const double b = a * dt / (2*h);
    if (!std::isfinite(2*r) || !std::isfinite(b) || std::abs(a)*h > 2*D)
        Rcpp::stop("Require finite stencil coefficients and |a| h <= 2D.");
    if (method == Method::explicit_euler && r > 0.5 + 1e-14)
        Rcpp::stop("Explicit Euler requires D dt / h^2 <= 1/2; increase n.");
    return {r-b, r, r+b};
}

// The endpoints of both buffers are zero throughout the simulation.
void apply_stencil(const std::vector<double>& old, std::vector<double>& next,
                   double left, double center, double right) {
    const int N = static_cast<int>(old.size()) - 1;
    for (int j = 1; j < N; ++j)
        next[j] = left*old[j-1] + center*old[j] + right*old[j+1];
}

// Thomas elimination on the N-1 interior unknowns. The constant matrix
// is strictly diagonally dominant under the checked spatial condition.
struct TridiagonalFactors {
    double lower;
    std::vector<double> inverse_pivot, upper_ratio;

    TridiagonalFactors(int N, const Stencil& s, double theta)
        : lower(-theta*s.left), inverse_pivot(N), upper_ratio(N) {
        const double diagonal = 1 + 2*theta*s.diffusion;
        const double upper = -theta*s.right;
        for (int j = 1; j < N; ++j) {
            inverse_pivot[j] = 1.0 / (diagonal - lower*upper_ratio[j-1]);
            upper_ratio[j] = upper*inverse_pivot[j];
        }
    }

    void solve_in_place(std::vector<double>& rhs) const {
        const int N = static_cast<int>(rhs.size()) - 1;
        for (int j = 1; j < N; ++j)
            rhs[j] = (rhs[j] - lower*rhs[j-1])*inverse_pivot[j];
        for (int j = N-2; j >= 1; --j)
            rhs[j] -= upper_ratio[j]*rhs[j+1];
    }
};

std::vector<double> evolve(Method method, const Rcpp::NumericVector& initial,
                           const Stencil& s, int steps) {
    const int N = static_cast<int>(initial.size()) - 1;
    std::vector<double> u(initial.begin(), initial.end());
    u.front() = u.back() = 0.0;

    if (method == Method::explicit_euler) {
        std::vector<double> next(N+1, 0.0);
        for (int k = 0; k < steps; ++k) {
            if (k % 1024 == 0) Rcpp::checkUserInterrupt();
            apply_stencil(u, next, s.left, 1-2*s.diffusion, s.right);
            u.swap(next);
        }
        return u;
    }

    const bool crank_nicolson = method == Method::crank_nicolson;
    const double theta = crank_nicolson ? 0.5 : 1.0;
    TridiagonalFactors factors(N, s, theta);
    std::vector<double> rhs;
    if (crank_nicolson) rhs.assign(N+1, 0.0);
    for (int k = 0; k < steps; ++k) {
        if (k % 1024 == 0) Rcpp::checkUserInterrupt();
        if (crank_nicolson) {
            apply_stencil(u, rhs, 0.5*s.left, 1-s.diffusion, 0.5*s.right);
            rhs.swap(u);
        }
        // For backward Euler the old solution is already the right-hand side.
        factors.solve_in_place(u);
    }
    return u;
}
} // namespace

// [[Rcpp::export]]
Rcpp::NumericVector fd_solve_cpp(std::string method, Rcpp::NumericVector initial,
                                  double a, double D, double T, int n) {
    const Method scheme = parse_method(method);
    const Stencil stencil = spatial_stencil(initial, a, D, T, n, scheme);
    return Rcpp::wrap(evolve(scheme, initial, stencil, n));
}

// Same solver as fd_solve_cpp. Time independent solves including construction
// and release of buffers/factors. Exclude R dispatch, validation and conversion.
// [[Rcpp::export]]
double fd_benchmark_cpp(std::string method, Rcpp::NumericVector initial,
                        double a, double D, double T, int n, int batch) {
    if (batch < 1) Rcpp::stop("batch must be positive.");
    const Method scheme = parse_method(method);
    const Stencil stencil = spatial_stencil(initial, a, D, T, n, scheme);
    using Clock = std::chrono::steady_clock;
    volatile double terminal_value = 0;
    const auto start = Clock::now();
    for (int i = 0; i < batch; ++i) {
        const auto result = evolve(scheme, initial, stencil, n);
        terminal_value = result[result.size()/2]; // keep every timed solve observable
    }
    const double elapsed = std::chrono::duration<double>(Clock::now()-start).count();
    (void)terminal_value;
    return elapsed / batch;
}
