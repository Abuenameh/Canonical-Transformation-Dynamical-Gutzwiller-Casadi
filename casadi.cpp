#include <boost/thread.hpp>

using namespace boost;

#include "casadi.hpp"

double energyfunc(const vector<double>& x, vector<double>& grad, void *data) {
    DynamicsProblem* prob = static_cast<DynamicsProblem*> (data);
    return prob->E(x, grad);
}

SX JW(SX W) {
    return alpha * (W * W) / (Ng * Ng + W * W);
}

SX JWij(SX Wi, SX Wj) {
    return alpha * (Wi * Wj) / (sqrt(Ng * Ng + Wi * Wi) * sqrt(Ng * Ng + Wj * Wj));
}

SX UW(SX W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

complex<double> dot(vector<complex<double>>&v, vector<complex<double>>&w) {
    complex<double> res = 0;
    for (int i = 0; i < v.size(); i++) {
        res += conj(v[i]) * w[i];
    }
    return res;
}

complex<double> b0(vector<vector<complex<double>>>& f, int i) {
    complex<double> bi = 0;
    for (int n = 1; n <= nmax; n++) {
        bi += sqrt(1.0 * n) * f[i][n - 1] * f[i][n];
    }
    return bi;
}

complex<double> b1(vector<vector<complex<double>>>& f, int i, vector<double>& J, double U) {
    complex<double> bi = 0;

    int j1 = mod(i - 1);
    int j2 = mod(i + 1);
    for (int n = 0; n < nmax; n++) {
        for (int m = 1; m <= nmax; m++) {
            if (n != m - 1) {
                bi += -J[i] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * n + 1) * conj(f[j2][m - 1]) * f[j2][m] * (conj(f[i][n + 1]) * f[i][n + 1] - conj(f[i][n]) * f[i][n]);
                bi += -J[j1] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * n + 1) * conj(f[j1][m - 1]) * f[j1][m] * (conj(f[i][n + 1]) * f[i][n + 1] - conj(f[i][n]) * f[i][n]);

                if (m < nmax) {
                    bi += -J[i] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * m + 1) * conj(f[j2][n + 1]) * f[j2][n] * conj(f[i][m - 1]) * f[i][m + 1];
                    bi += -J[j1] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * m + 1) * conj(f[j1][n + 1]) * f[j1][n] * conj(f[i][m - 1]) * f[i][m + 1];
                }
                if (m > 1) {
                    bi += J[i] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * m - 1) * conj(f[j2][n + 1]) * f[j2][n] * conj(f[i][m - 2]) * f[i][m];
                    bi += J[j1] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * m - 1) * conj(f[j1][n + 1]) * f[j1][n] * conj(f[i][m - 2]) * f[i][m];
                }
            }
        }
    }
    return bi;
}

complex<double> bf1(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;

    if (b == i && q == n + 1 && j == k) {
        if (a != k) {
            if (m >= 2) {
                bi -= (n + 1) * sqrt(1.0 * m * (m - 1) * (p + 1)) * conj(f[i][n]) * conj(f[a][p + 1]) * conj(f[k][m - 2]) * f[i][n] * f[a][p] * f[k][m];
                bi += (n + 1) * sqrt(1.0 * m * (m - 1) * (p + 1)) * conj(f[i][n + 1]) * conj(f[a][p + 1]) * conj(f[k][m - 2]) * f[i][n + 1] * f[a][p] * f[k][m];
            }
            if (m < nmax) {
                bi += (n + 1) * sqrt(1.0 * m * (m + 1) * (p + 1)) * conj(f[i][n]) * conj(f[a][p + 1]) * conj(f[k][m - 1]) * f[i][n] * f[a][p] * f[k][m + 1];
                bi -= (n + 1) * sqrt(1.0 * m * (m + 1) * (p + 1)) * conj(f[i][n + 1]) * conj(f[a][p + 1]) * conj(f[k][m - 1]) * f[i][n + 1] * f[a][p] * f[k][m + 1];
            }
        }
        else {
            if (p == m - 1) {
                if (m < nmax) {
                    bi += m * (n + 1) * sqrt(1.0 * m + 1) * conj(f[i][n]) * conj(f[k][m]) * f[i][n] * f[k][m + 1];
                    bi -= m * (n + 1) * sqrt(1.0 * m + 1) * conj(f[i][n + 1]) * conj(f[k][m]) * f[i][n + 1] * f[k][m + 1];
                }
            }
            else if (p == m - 2) {
                bi -= (m - 1) * (n + 1) * sqrt(1.0 * m) * conj(f[i][n]) * conj(f[k][m - 1]) * f[i][n] * f[k][m];
                bi += (m - 1) * (n + 1) * sqrt(1.0 * m) * conj(f[i][n + 1]) * conj(f[k][m - 1]) * f[i][n + 1] * f[k][m];
            }
        }
    }
    return bi;
}

complex<double> bf2(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (b == k && j == k) {
        if (a != i) {
            if (q == m - 1 && m < nmax) {
                bi += sqrt(1.0 * (p + 1) * (n + 1) * m * (m + 1) * q) * conj(f[i][n + 1]) * conj(f[a][p + 1]) * conj(f[k][m - 2]) * f[i][n] * f[a][p] * f[k][m + 1];
            }
            if (q == m + 2) {
                bi -= sqrt(1.0 * (p + 1) * (n + 1) * m * (m + 1) * q) * conj(f[i][n + 1]) * conj(f[a][p + 1]) * conj(f[k][m - 1]) * f[i][n] * f[a][p] * f[k][m + 2];
            }
            if (q == m - 2) {
                bi -= sqrt(1.0 * (p + 1) * (n + 1) * (m - 1) * m * q) * conj(f[i][n + 1]) * conj(f[a][p + 1]) * conj(f[k][m - 3]) * f[i][n] * f[a][p] * f[k][m];
            }
            if (q == m + 1 && m >= 2) {
                bi += sqrt(1.0 * (p + 1) * (n + 1) * (m - 1) * m * q) * conj(f[i][n + 1]) * conj(f[a][p + 1]) * conj(f[k][m - 2]) * f[i][n] * f[a][p] * f[k][m + 1];
            }
        }
        else if (p == n + 1) {
            if (q == m - 1 && n < nmax - 1 && m < nmax) {
                bi += sqrt(1.0 * (n + 2) * (n + 1) * m * (m + 1) * (m - 1)) * conj(f[i][n + 2]) * conj(f[k][m - 2]) * f[i][n] * f[k][m + 1];
            }
            if (q == m + 2 && n < nmax - 1) {
                bi -= sqrt(1.0 * (n + 2) * (n + 1) * m * (m + 1) * (m + 2)) * conj(f[i][n + 2]) * conj(f[k][m - 1]) * f[i][n] * f[k][m + 2];
            }
            if (q == m - 2 && n < nmax - 1) {
                bi -= sqrt(1.0 * (n + 2) * (n + 1) * (m - 1) * m * (m - 2)) * conj(f[i][n + 2]) * conj(f[k][m - 3]) * f[i][n] * f[k][m];
            }
            if (q == m + 1 && n < nmax - 1 && m >= 2) {
                bi += sqrt(1.0 * (n + 2) * (n + 1) * (m - 1) * m * (m + 1)) * conj(f[i][n + 2]) * conj(f[k][m - 2]) * f[i][n] * f[k][m + 1];
            }
        }
    }
    return bi;
}

complex<double> bf3(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (i == a && j == k) {
        if (b != k) {
            if (p == n + 1 && m < nmax) {
                bi += sqrt(1.0 * q * (n + 1) * (n + 2) * m * (m + 1)) * conj(f[i][n + 2]) * conj(f[b][q - 1]) * conj(f[k][m - 1]) * f[i][n] * f[b][q] * f[k][m + 1];
            }
            if (p == n + 1 && m >= 2) {
                bi -= sqrt(1.0 * q * (n + 1) * (n + 2) * (m - 1) * m) * conj(f[i][n + 2]) * conj(f[b][q - 1]) * conj(f[k][m - 2]) * f[i][n] * f[b][q] * f[k][m];
            }
            if (p == n - 1 && m < nmax) {
                bi -= sqrt(1.0 * q * n * (n + 1) * m * (m + 1)) * conj(f[i][n + 1]) * conj(f[b][q - 1]) * conj(f[k][m - 1]) * f[i][n - 1] * f[b][q] * f[k][m + 1];
            }
            if (p == n - 1 && m >= 2) {
                bi += sqrt(1.0 * q * n * (n + 1) * (m - 1) * m) * conj(f[i][n + 1]) * conj(f[b][q - 1]) * conj(f[k][m - 2]) * f[i][n - 1] * f[b][q] * f[k][m];
            }
        }
        else {
            if (q == m + 2 && p == n + 1) {
                bi += sqrt(1.0 * (n + 1) * (n + 2) * m * (m + 1) * (m + 2)) * conj(f[i][n + 2]) * conj(f[k][m - 1]) * f[i][n] * f[k][m + 2];
            }
            if (q == m + 1 && m >= 2 && p == n + 1) {
                bi -= sqrt(1.0 * (n + 1) * (n + 2) * (m - 1) * m * (m + 1)) * conj(f[i][n + 2]) * conj(f[k][m - 2]) * f[i][n] * f[k][m + 1];
            }
            if (q == m + 2 && p == n - 1) {
                bi -= sqrt(1.0 * n * (n + 1) * m * (m + 1) * (m + 2)) * conj(f[i][n + 1]) * conj(f[k][m - 1]) * f[i][n - 1] * f[k][m + 2];
            }
            if (q == m + 1 && m >= 2 && p == n - 1) {
                bi += sqrt(1.0 * n * (n + 1) * (m - 1) * m * (m + 1)) * conj(f[i][n + 1]) * conj(f[k][m - 2]) * f[i][n - 1] * f[k][m + 1];
            }
        }
    }
    return bi;
}

complex<double> bf4(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (a == k && j == k) {
        if (b != i) {
            if (p == m - 1 && m < nmax) {
                bi += m * sqrt(1.0 * (n + 1) * q * (m + 1)) * conj(f[i][n + 1]) * conj(f[b][q - 1]) * conj(f[k][m]) * f[i][n] * f[b][q] * f[k][m + 1];
            }
            if (p == m - 2) {
                bi -= (m - 1) * sqrt(1.0 * (n + 1) * q * m) * conj(f[i][n + 1]) * conj(f[b][q - 1]) * conj(f[k][m - 1]) * f[i][n] * f[b][q] * f[k][m];
            }
            if (p == m) {
                bi -= (m + 1) * sqrt(1.0 * (n + 1) * q * m) * conj(f[i][n + 1]) * conj(f[b][q - 1]) * conj(f[k][m - 1]) * f[i][n] * f[b][q] * f[k][m];
            }
            if (p == m - 1 && m >= 2) {
                bi += m * sqrt(1.0 * (n + 1) * q * (m - 1)) * conj(f[i][n + 1]) * conj(f[b][q - 1]) * conj(f[k][m - 2]) * f[i][n] * f[b][q] * f[k][m - 1];
            }
        }
        else if (n == q - 1) {
            if (p == m - 1 && m < nmax) {
                bi += (n + 1) * m * sqrt(1.0 * (m + 1)) * conj(f[i][n + 1]) * conj(f[k][m]) * f[i][n + 1] * f[k][m + 1];
            }
            if (p == m - 2) {
                bi -= (n + 1) * (m - 1) * sqrt(1.0 * m) * conj(f[i][n + 1]) * conj(f[k][m - 1]) * f[i][n + 1] * f[k][m];
            }
            if (p == m) {
                bi -= (n + 1) * (m + 1) * sqrt(1.0 * m) * conj(f[i][n + 1]) * conj(f[k][m - 1]) * f[i][n + 1] * f[k][m];
            }
            if (p == m - 1 && m >= 2) {
                bi += (n + 1) * m * sqrt(1.0 * (m - 1)) * conj(f[i][n + 1]) * conj(f[k][m - 2]) * f[i][n + 1] * f[k][m - 1];
            }
        }
    }
    return bi;
}

complex<double> bf5(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (i == b && i == k) {
        if (j != a) {
            if (q == n + 1) {
                bi += 2 * (n + 1) * sqrt(1.0 * m * (p + 1) * (n + 1)) * conj(f[j][m - 1]) * conj(f[a][p + 1]) * conj(f[k][n]) * f[j][m] * f[a][p] * f[k][n + 1];
            }
            if (q == n) {
                bi -= (n + 1) * sqrt(1.0 * m * (p + 1) * n) * conj(f[j][m - 1]) * conj(f[a][p + 1]) * conj(f[k][n - 1]) * f[j][m] * f[a][p] * f[k][n];
            }
            if (q == n + 2) {
                bi -= (n + 1) * sqrt(1.0 * m * (p + 1) * (n + 2)) * conj(f[j][m - 1]) * conj(f[a][p + 1]) * conj(f[k][n + 1]) * f[j][m] * f[a][p] * f[k][n + 2];
            }
        }
        else if (p == m - 1) {
            if (q == n + 1) {
                bi += 2 * (n + 1) * m * sqrt(1.0 * (n + 1)) * conj(f[j][p + 1]) * conj(f[k][n]) * f[j][m] * f[k][n + 1];
            }
            if (q == n) {
                bi -= (n + 1) * m * sqrt(1.0 * n) * conj(f[j][p + 1]) * conj(f[k][n - 1]) * f[j][m] * f[k][n];
            }
            if (q == n + 2) {
                bi -= (n + 1) * m * sqrt(1.0 * (n + 2)) * conj(f[j][p + 1]) * conj(f[k][n + 1]) * f[j][m] * f[k][n + 2];
            }
        }
    }
    return bi;
}

complex<double> bf6(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (i == k && j == b) {
        if (i != a) {
            if (q == m - 1) {
                bi += (n + 1) * sqrt(1.0 * (p + 1) * (m - 1) * m) * conj(f[j][m - 2]) * conj(f[a][p + 1]) * f[j][m] * f[a][p] * (conj(f[k][n + 1]) * f[k][n + 1] - conj(f[k][n]) * f[k][n]);
            }
            if (q == m + 1) {
                bi -= (n + 1) * sqrt(1.0 * (p + 1) * (m + 1) * m) * conj(f[j][m - 1]) * conj(f[a][p + 1]) * f[j][m + 1] * f[a][p] * (conj(f[k][n + 1]) * f[k][n + 1] - conj(f[k][n]) * f[k][n]);
            }
        }
        else {
            if (p == n + 1 && q == m - 1) {
                bi += (n + 1) * sqrt(1.0 * m * (m - 1) * (n + 2)) * conj(f[j][m - 2]) * conj(f[k][n + 2]) * f[j][m] * f[k][n + 1];
            }
            if (p == n + 1 && q == m + 1) {
                bi -= (n + 1) * sqrt(1.0 * m * (m + 1) * (n + 2)) * conj(f[j][m - 1]) * conj(f[k][n + 2]) * f[j][m + 1] * f[k][n + 1];
            }
            if (p == n && q == m - 1) {
                bi -= (n + 1) * sqrt(1.0 * m * (m - 1) * (n + 1)) * conj(f[j][m - 2]) * conj(f[k][n + 1]) * f[j][m] * f[k][n];
            }
            if (p == n && q == m + 1) {
                bi += (n + 1) * sqrt(1.0 * m * (m + 1) * (n + 1)) * conj(f[j][m - 1]) * conj(f[k][n + 1]) * f[j][m + 1] * f[k][n];
            }
        }
    }
    return bi;
}

complex<double> bf7(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (i == k && i == a) {
        if (j != b) {
            if (p == n + 1) {
                bi += (n + 1) * sqrt(1.0 * m * q * (p + 1)) * conj(f[j][m - 1]) * conj(f[b][q - 1]) * conj(f[k][n + 2]) * f[j][m] * f[b][q] * f[k][n + 1];
            }
            if (p == n) {
                bi -= 2 * (n + 1) * sqrt(1.0 * m * q * (p + 1)) * conj(f[j][m - 1]) * conj(f[b][q - 1]) * conj(f[k][n + 1]) * f[j][m] * f[b][q] * f[k][n];
            }
            if (p == n - 1) {
                bi += (n + 1) * sqrt(1.0 * m * q * (p + 1)) * conj(f[j][m - 1]) * conj(f[b][q - 1]) * conj(f[k][n]) * f[j][m] * f[b][q] * f[k][n - 1];
            }
        }
        else if (m == q - 1) {
            if (p == n + 1) {
                bi += (n + 1) * sqrt(1.0 * m * (m + 1) * (n + 2)) * conj(f[j][m - 1]) * conj(f[k][n + 2]) * f[j][m + 1] * f[k][n + 1];
            }
            if (p == n) {
                bi -= 2 * (n + 1) * sqrt(1.0 * m * (m + 1) * (n + 1)) * conj(f[j][m - 1]) * conj(f[k][n + 1]) * f[j][m + 1] * f[k][n];
            }
            if (p == n - 1) {
                bi += (n + 1) * sqrt(1.0 * m * (m + 1) * n) * conj(f[j][m - 1]) * conj(f[k][n]) * f[j][m + 1] * f[k][n - 1];
            }
        }
    }
    return bi;
}

complex<double> bf8(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (i == k && m == p + 1 && j == a) {
        if (i != b) {
            bi += (n + 1) * m * sqrt(1.0 * q) * conj(f[j][m]) * conj(f[b][q - 1]) * conj(f[k][n + 1]) * f[j][m] * f[b][q] * f[k][n + 1];
            bi -= (n + 1) * m * sqrt(1.0 * q) * conj(f[j][m - 1]) * conj(f[b][q - 1]) * conj(f[k][n + 1]) * f[j][m - 1] * f[b][q] * f[k][n + 1];
            bi -= (n + 1) * m * sqrt(1.0 * q) * conj(f[j][m]) * conj(f[b][q - 1]) * conj(f[k][n]) * f[j][m] * f[b][q] * f[k][n];
            bi += (n + 1) * m * sqrt(1.0 * q) * conj(f[j][m - 1]) * conj(f[b][q - 1]) * conj(f[k][n]) * f[j][m - 1] * f[b][q] * f[k][n];
        }
        else {
            if (q == n + 2) {
                bi += (n + 1) * m * sqrt(1.0 * (n + 2)) * conj(f[k][n + 1]) * f[k][n + 2] * (conj(f[j][m]) * f[j][m] - conj(f[j][m - 1]) * f[j][m - 1]);
            }
            if (q == n + 1) {
                bi -= (n + 1) * m * sqrt(1.0 * (n + 1)) * conj(f[k][n]) * f[k][n + 1] * (conj(f[j][m]) * f[j][m] - conj(f[j][m - 1]) * f[j][m - 1]);
            }
        }
    }
    return bi;
}

complex<double> bf(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    bi += bf1(f, k, i, j, a, b, n, m, p, q);
    bi += bf2(f, k, i, j, a, b, n, m, p, q);
    bi += bf3(f, k, i, j, a, b, n, m, p, q);
    bi += bf4(f, k, i, j, a, b, n, m, p, q);
    bi += bf5(f, k, i, j, a, b, n, m, p, q);
    bi += bf6(f, k, i, j, a, b, n, m, p, q);
    bi += bf7(f, k, i, j, a, b, n, m, p, q);
    bi += bf8(f, k, i, j, a, b, n, m, p, q);
    return bi;
}

complex<double> b2(vector<vector<complex<double>>>& f, int k, vector<double>& J, double U) {
    complex<double> bi = 0;
    for (int i = 0; i < L; i++) {
        int j1 = mod(i - 1);
        int j2 = mod(i + 1);
        for (int a = 0; a < L; a++) {
            int b1 = mod(a - 1);
            int b2 = mod(a + 1);
            for (int n = 0; n < nmax; n++) {
                for (int m = 1; m <= nmax; m++) {
                    for (int p = 0; p < nmax; p++) {
                        for (int q = 1; q <= nmax; q++) {
                            if (n != m - 1 && p != q - 1) {
                                bi += J[j1] * J[b1] / (eps(U, n, m) * eps(U, p, q)) * bf(f, k, i, j1, a, b1, n, m, p, q);
                                bi += J[j1] * J[a] / (eps(U, n, m) * eps(U, p, q)) * bf(f, k, i, j1, a, b2, n, m, p, q);
                                bi += J[i] * J[b1] / (eps(U, n, m) * eps(U, p, q)) * bf(f, k, i, j2, a, b1, n, m, p, q);
                                bi += J[i] * J[a] / (eps(U, n, m) * eps(U, p, q)) * bf(f, k, i, j2, a, b2, n, m, p, q);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0.5 * bi;
}

complex<double> b3(vector<vector<complex<double>>>& f, int k, vector<double>& J, double U) {
    complex<double> bi = 0;
    for (int i = 0; i < L; i++) {
        int j1 = mod(i - 1);
        int j2 = mod(i + 1);
        for (int a = 0; a < L; a++) {
            int b1 = mod(a - 1);
            int b2 = mod(a + 1);
            for (int n = 0; n < nmax; n++) {
                for (int m = 1; m <= nmax; m++) {
                    for (int p = 0; p < nmax; p++) {
                        if (n != m - 1) {
                            bi += J[j1] * J[b1] / (eps(U, n, m) * (eps(U, n, m) + eps(U, p, p + 1))) * (bf(f, k, a, b1, i, j1, p, p + 1, n, m) - bf(f, k, i, j1, a, b1, n, m, p, p + 1));
                            bi += J[j1] * J[a] / (eps(U, n, m) * (eps(U, n, m) + eps(U, p, p + 1))) * (bf(f, k, a, b2, i, j1, p, p + 1, n, m) - bf(f, k, i, j1, a, b2, n, m, p, p + 1));
                            bi += J[i] * J[b1] / (eps(U, n, m) * (eps(U, n, m) + eps(U, p, p + 1))) * (bf(f, k, a, b1, i, j2, p, p + 1, n, m) - bf(f, k, i, j2, a, b1, n, m, p, p + 1));
                            bi += J[i] * J[a] / (eps(U, n, m) * (eps(U, n, m) + eps(U, p, p + 1))) * (bf(f, k, a, b2, i, j2, p, p + 1, n, m) - bf(f, k, i, j2, a, b2, n, m, p, p + 1));
                        }
                        for (int q = 1; q <= nmax; q++) {
                            if (n != m - 1 && p != q - 1 && n - m != p - q) {
                                bi += 0.25 * J[j1] * J[b1] / (eps(U, n, m) + eps(U, q - 1, p + 1)) * (1 / eps(U, n, m) - 1 / eps(U, q - 1, p + 1)) * (bf(f, k, a, b1, i, j1, q - 1, p + 1, n, m) - bf(f, k, i, j1, a, b1, n, m, q - 1, p + 1));
                                bi += 0.25 * J[j1] * J[a] / (eps(U, n, m) + eps(U, q - 1, p + 1)) * (1 / eps(U, n, m) - 1 / eps(U, q - 1, p + 1)) * (bf(f, k, a, b2, i, j1, q - 1, p + 1, n, m) - bf(f, k, i, j1, a, b2, n, m, q - 1, p + 1));
                                bi += 0.25 * J[i] * J[b1] / (eps(U, n, m) + eps(U, q - 1, p + 1)) * (1 / eps(U, n, m) - 1 / eps(U, q - 1, p + 1)) * (bf(f, k, a, b1, i, j2, q - 1, p + 1, n, m) - bf(f, k, i, j2, a, b1, n, m, q - 1, p + 1));
                                bi += 0.25 * J[i] * J[a] / (eps(U, n, m) + eps(U, q - 1, p + 1)) * (1 / eps(U, n, m) - 1 / eps(U, q - 1, p + 1)) * (bf(f, k, a, b2, i, j2, q - 1, p + 1, n, m) - bf(f, k, i, j2, a, b2, n, m, q - 1, p + 1));
                            }
                        }
                    }
                }
            }
        }
    }
    return bi;
}

complex<double> b(vector<vector<complex<double>>>& f, int k, vector<double>& J, double U) {
    complex<double> bi = 0;
    bi += b0(f, k);
    bi += b1(f, k, J, U);
    bi += b2(f, k, J, U);
    bi += b3(f, k, J, U);
    return bi;
}

namespace casadi {

    inline bool isnan(SX& sx) {
        return sx.at(0).isNan();
    }

    inline bool isinf(SX sx) {
        return sx.at(0).isInf();
    }
}

//boost::mutex problem_mutex;

DynamicsProblem::DynamicsProblem() {

    //    vector<vector<complex<double>>> ff({vector<complex<double>>{0.4704433287137315,0.31278265828701396,0.020491099557798563,0.5120264351039052,
    //    0.4418501306816557,0.47225796949755755},vector<complex<double>>{0.5055884439289716,0.30266074435637946,0.09512521370830883,0.24918922817582706,
    //    0.5783175475335864,0.4971735468465169},vector<complex<double>>{0.23298975595641222,0.5093931448039285,0.3201022274576789,0.32897215154641374,
    //    0.4453746331978017,0.5264862023202649},vector<complex<double>>{0.030551894237901167,0.6347337871321124,0.013578078921715465,0.6730435257425468,
    //    0.20898496392524407,0.31517127774180836},vector<complex<double>>{0.5035952546202369,0.5821938736982794,0.27559620353757475,0.29400743122032663,
    //    0.3952426430295389,0.2980465216473689}});
    ////    vector<double> JJ(5, 8.608695652173912e6);
    //    vector<double> JJ({2.359359141043023e6,4.996229498581644e6,6.901292936302477e6,8.126176253257117e6,
    //   3.452496551985471e6});
    //    double UU = 1.0633270321361059e8;
    //    complex<double> qwe = b(ff, 0, JJ, UU);
    ////    complex<double> qwe = bf1(ff, 0, 1, 0, 0, 1, 4, 4, 3, 5);
    //    cout << qwe << endl;
    //    exit(0);

    fin = SX::sym("f", 1, 1, 2 * L * dim);
    dU = SX::sym("dU", 1, 1, L);
    J = SX::sym("J", 1, 1, L);
    //    Jp = SX::sym("Jp", 1, 1, L);
    U0 = SX::sym("U0");
    mu = SX::sym("mu");
    t = SX::sym("t");
    xi = SX::sym("xi", 1, 1, L);

    Wi = SX::sym("Wi");
    Wf = SX::sym("Wf");
    tau = SX::sym("tau");
    Wt = if_else(t < tau, Wi + (Wf - Wi) * t / tau, Wf + (Wi - Wf) * (t - tau) / tau);
    //    SXFunction Wtf(vector<SX>{t}, vector<SX>{Wt});
    //    Wtf.init();
    //    Function Wtdt = Wtf.gradient(0, 0);
    //    Wtdt.init();
    //    SX Wpt = Wtdt.call(vector<SX>{t})[0];
    SX Ut = 1;
    double Ji = 0.2;
    double Jf = 0.01;
    SX Jt = if_else(t < tau, Ji + (Jf - Ji) * t / tau, Jf + (Ji - Jf)*(t - tau) / tau);
    U0 = Ut; //scale*UW(Wt);
    Jfunc = vector<SXFunction>(L);
    for (int i = 0; i < L; i++) {
        J[i] = Jt; //scale*JWij(Wt * xi[i], Wt * xi[mod(i + 1)]);
        //        J[i] = scale*JWij(Wt, Wt);
        //        Jp[i] = JWij(Wpt * xi[i], Wpt * xi[mod(i + 1)]);
        dU[i] = scale * UW(Wt * xi[i]) - U0;
    }

    //    SX Wisubs = 2e11;
    //    SX Wfsubs = 1e11;
    //    SX tausubs = 1e-6;
    //    vector<SX> xisubs(L);
    //    for (int i = 0; i < L; i++) {
    //        xisubs[i] = 1;
    //    }

    //    SX Jsubs = substitute(vector<SX>{J[0]}, vector<SX>{Wi, Wf, tau}, vector<SX>{Wisubs, Wfsubs, tausubs})[0];
    //    Jsubs = substitute(vector<SX>{Jsubs}, xi, xisubs)[0];
    //    SX Jpsubs = substitute(vector<SX>{Jp[0]}, vector<SX>{Wi, Wf, tau}, vector<SX>{Wisubs, Wfsubs, tausubs})[0];
    //    Jpsubs = substitute(vector<SX>{Jpsubs}, xi, xisubs)[0];
    //    SXFunction Jfunc = SXFunction(vector<SX>{t}, vector<SX>{Jsubs});
    //    SXFunction Jpfunc = SXFunction(vector<SX>{t}, vector<SX>{Jpsubs});
    //    Jfunc.init();
    //    Jpfunc.init();
    //    double qwe = Jfunc(DMatrix(0.5e-6))[0].getValue();
    //    double asd = Jpfunc(DMatrix(0.5e-6))[0].getValue();
    //    cout << qwe << endl << asd << endl;
    //    exit(0);

    vector<SX> params;
    params.push_back(Wi);
    params.push_back(Wf);
    params.push_back(tau);
    for (SX sx : xi) params.push_back(sx);
    params.push_back(mu);

    vector<SX> gsparams(params.begin(), params.end());
    gsparams.push_back(t);
    
    SX E = energya(true);
    SX S = canonicala();

    SXFunction Sf(vector<SX>{t}, vector<SX>{S});
    Sf.init();
    Function Sdtf = Sf.gradient(0, 0);
    Sdtf.init();
    SX Sdt = Sdtf.call(vector<SX>{t})[0];

    SX GSE = energya(true);//E;

    x = SX::sym("x", fin.size());
    p = SX::sym("p", params.size());
    gsp = SX::sym("gsp", gsparams.size());

    vector<SX> xs;
    for (int i = 0; i < x.size(); i++) {
        xs.push_back(x.at(i));
    }
    vector<SX> ps;
    for (int i = 0; i < p.size(); i++) {
        ps.push_back(p.at(i));
    }
    vector<SX> gsps;
    for (int i = 0; i < gsp.size(); i++) {
        gsps.push_back(gsp.at(i));
    }

    GSE = substitute(vector<SX>{GSE}, fin, xs)[0];
    GSE = substitute(vector<SX>{GSE}, gsparams, gsps)[0];
    simplify(GSE);

    vector<SX> qwe;
//    vector<double> qwe;
    qwe.push_back(0);
    qwe.push_back(0);
    qwe.push_back(1);
    for (int i = 0; i < L; i++) qwe.push_back(1);
    qwe.push_back(0.5);
//    qwe.push_back(0);
    
    vector<SX> asd = vector<SX>{0.43612801607392543,0.23717544586479253,0.26077849862095465,0.3200273453213472,
   0.10425529497771417,0.3824801486141887,0.2429356567353783,0.30160591780809193,
   0.3729682453346464,0.36995652280327834,0.16884355568029785,0.03306632031403832,
   0.12079205421387129,0.019706647344946406,0.440966613369467,0.4184572680568556,
   0.0016817400101205538,0.35118576227869996,0.4432247501716445,0.5158283760299978,
   0.19735658869748748,0.3777909238304533,0.458763968593948,0.03632887765359411,
   0.37947968147894784,0.2501010473047869,0.11842238203139385,0.4204266843357301,
   0.2516913796220109,0.381908180032208,0.24404811263968323,0.46479584529800944,
   0.5442060683225153,0.16893924185637502,0.01228585898011247,0.0339108004677902,
   0.37496157323943646,0.03898019300644058,0.2507702727800595,0.439775371513932,
   0.06507875464590163,0.027214305949468137,0.5565377930879596,0.0601314141641671,
   0.0769433485390323,0.2703479498191469,0.4360466037340656,0.46099339709471787,
   0.0012596639678839389,0.4472276855682792};
    
    vector<SX> asd2 = vector<SX>{1.2641379967578872e-8,1.0329124622714652e-8,-9.010011167448173e-8,
//    vector<double> asd = vector<double>{1.2641379967578872e-8,1.0329124622714652e-8,-9.010011167448173e-8,
   4.952024194390157e-9,-2.4736059994708016e-8,6.767942814862892e-8,1.5078082738426852,
   -0.7273628963610762,2.5651999213642657e-8,5.274015577978409e-8,1.6126387654535521,
   0.44937597795802225,-3.082582069228331e-8,-3.7348106231131246e-9,
   -3.107817148602482e-9,-9.18356240901078e-8,-1.3570714372580044e-8,
   -1.7348072777272648e-8,-1.0302373095677293,-1.0802183085677775,-3.075869188792871e-8,
   -2.3051053277433743e-8,-0.8278062664917237,1.242174415499053,-2.489748185603503e-8,
   -5.124088914618176e-8,1.6149957978339713e-7,1.6817491005983512e-8,
   9.052543197693599e-9,-2.0088986721592977e-8,-1.5857366031377942,1.398449913478343,
   -1.6269609273222e-8,-4.141137168722801e-8,-0.6563456506116018,-2.0098341174730647,
   3.090771469109901e-9,-1.2794123357267029e-8,-1.3301642603877747e-8,
   -3.231925612400513e-8,-3.1352157953272317e-9,-7.543204694376377e-9,
   -0.46385694011152556,-1.1905932570298374,1.6234629769130116e-8,6.488901740580542e-10,
   -1.0063744407621653,-0.7873284305173013,5.581134735279098e-9,-4.176388117800141e-9,
   2.6580125078511663e-8,-1.0313661578283649e-8,-5.9072349948463525e-8,
   9.43792972683785e-9,0.3172741995274047,-1.4135896677056718,-8.021046572151412e-9,
   -1.4296187870572331e-8,-0.447057327195672,1.3780560445876853};
//    for (int i = 0; i < xs.size()/2; i++) {
////        asd.push_back(1/sqrt(2.*dim));
//        asd.push_back((i%dim)/sqrt(2.*dim));
//        asd.push_back((i%dim + 1)/sqrt(2.*dim));
//    }

    Ef = SXFunction(nlpIn("x", x, "p", gsp), nlpOut("f", GSE));
    Ef.init();
    Egradf = Ef.gradient(NL_X, NL_F);
    Egradf.init();


    SX HSr = Sdt;
    SX HSi = -E;
//    SX HSi = Sdt;//-E+Sdt;
    HSr = substitute(vector<SX>{HSr}, fin, xs)[0];
    HSr = substitute(vector<SX>{HSr}, params, ps)[0];
    HSi = substitute(vector<SX>{HSi}, fin, xs)[0];
    HSi = substitute(vector<SX>{HSi}, params, ps)[0];
    simplify(HSr);
    simplify(HSi);

    SXFunction HSf = SXFunction(vector<SX>{x, p}, vector<SX>{HSr, HSi});
    HSf.init();
    Function HSrdff = HSf.gradient(0, 0);
    Function HSidff = HSf.gradient(0, 1);
    HSrdff.init();
    HSidff.init();

    SXFunction HSf2 = SXFunction(vector<SX>{x, p, t}, vector<SX>{HSr, HSi});
//    cout << HSf2.getNumInputs() << endl;
//    cout << HSf2.getNumOutputs() << endl;
    HSf2.init();
    Function HSrdff2 = HSf2.gradient(0, 0);
    Function HSidff2 = HSf2.gradient(0, 1);
//    cout << HSidff2.getNumInputs() << endl;
//    cout << HSidff2.getNumOutputs() << endl;
    HSrdff2.init();
    HSidff2.init();

//    HSidff2.setInput(asd, 0);
//    HSidff2.setInput(qwe, 1);
//    HSidff2.setInput(1., 2);
//    HSidff2.evaluate();
//    SX HSi2 = HSidff.getOutput(0);
////    SX HSi2 = HSidff.call(vector<vector<SX>>{asd, qwe})[1];
////    HSi2 = substitute(vector<SX>{HSi2}, vector<SX>{t}, vector<SX>{0})[0];
//    cout << HSi2 << endl;
//    exit(0);

//    SX Sdt2 = substitute(vector<SX>{Sdt}, fin, asd)[0];
//    Sdt2 = substitute(vector<SX>{Sdt2}, gsparams, qwe)[0];
//    cout << Sdt2 << endl;
//    exit(0);

    SX HSrdftmp = HSrdff.call(vector<SX>{x, p})[0];
    SX HSidftmp = HSidff.call(vector<SX>{x, p})[0];

    SX HSitmp2 = HSidftmp;
    HSitmp2 = substitute(vector<SX>{HSitmp2}, xs, asd)[0];
    HSitmp2 = substitute(vector<SX>{HSitmp2}, ps, qwe)[0];
    HSitmp2 = substitute(vector<SX>{HSitmp2}, vector<SX>{t}, vector<SX>{0})[0];
//    cout << HSitmp2 << endl;
////    cout << HSidftmp << endl;
//    exit(0);

    ode = SX::sym("ode", 2 * L * dim);
    for (int i = 0; i < L * dim; i++) {
                ode[2 * i] = 0.5 * (HSrdftmp[2 * i] - HSidftmp[2 * i + 1]);
                ode[2 * i + 1] = 0.5 * (HSidftmp[2 * i] + HSrdftmp[2 * i + 1]);
//        ode[2 * i] = 0.5 * - HSidftmp[2 * i + 1];
//        ode[2 * i + 1] = 0.5 * HSidftmp[2 * i];
    }
    ode_func = SXFunction(daeIn("x", x, "t", t, "p", p), daeOut("ode", ode));

    Function g;
    integrator = new CvodesInterface(ode_func, g);
    integrator->setOption("max_num_steps", 1000000);
//    integrator->setOption("linear_multistep_method", "adams");
//        integrator->setOption("linear_solver", "csparse");
//        integrator->setOption("linear_solver_type", "user_defined");
//        integrator = new RkIntegrator(ode_func, g);
//        integrator->setOption("number_of_finite_elements", 50000);
    integrator->init();

    lopt = new opt(LD_LBFGS, 2 * L * dim);
    lopt->set_lower_bounds(-1);
    lopt->set_upper_bounds(1);
    lopt->set_min_objective(energyfunc, this);
}

void DynamicsProblem::setParameters(double Wi_, double Wf_, double tau_, vector<double>& xi_, double mu_) {
    params.clear();
    params.push_back(Wi_);
    params.push_back(Wf_);
    params.push_back(tau_ / scale);
    for (double xii : xi_) params.push_back(xii);
    params.push_back(0.5/*scale*mu_*/);
    integrator->setOption("t0", 0);
    integrator->setOption("tf", 2 * tau_ / scale);
    integrator->init();

    gsparams = vector<double>(params.begin(), params.end());
    gsparams.push_back(0);

    tf = 2 * tau_ / scale; // - 1e-10;
//    tf -= 1e-3;

    SX Wisubs = Wi_;
    SX Wfsubs = Wf_;
    SX tausubs = tau_ / scale;
    vector<SX> xisubs(L);
    for (int i = 0; i < L; i++) {
        xisubs[i] = xi_[i];
    }

    SX Usubs = substitute(vector<SX>{U0}, vector<SX>{Wi, Wf, tau}, vector<SX>{Wisubs, Wfsubs, tausubs})[0];
    Usubs = substitute(vector<SX>{Usubs}, xi, xisubs)[0];
    vector<SX> Jsubs = substitute(J, vector<SX>{Wi, Wf, tau}, vector<SX>{Wisubs, Wfsubs, tausubs});
    Ufunc = SXFunction(vector<SX>{t}, vector<SX>{Usubs});
    Ufunc.init();
    for (int i = 0; i < L; i++) {
        Jsubs[i] = substitute(vector<SX>{Jsubs[i]}, xi, xisubs)[0];
        Jfunc[i] = SXFunction(vector<SX>{t}, vector<SX>{Jsubs[i]});
        Jfunc[i].init();
    }
    //    DMatrix z(0);
    //    double J0 = Jfunc[0](z)[0].getValue();
    //    double U0 = Ufunc(z)[0].getValue();
    //    cout << J0 << "\t" << U0 << "\t" << J0/U0 << endl;
}

void DynamicsProblem::setInitial(vector<double>& f0) {
    x0.clear();
    for (double f0i : f0) x0.push_back(f0i);
    vector<complex<double>> fi(dim);
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            fi[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
        }
        double nrm = sqrt(abs(dot(fi, fi)));
        for (int n = 0; n <= nmax; n++) {
            x0[2 * (i * dim + n)] /= nrm;
            x0[2 * (i * dim + n) + 1] /= nrm;
        }
    }
}

void DynamicsProblem::solve() {
    double E0;
    //    try {
    start_time = microsec_clock::local_time();
    enum result res = lopt->optimize(x0, E0);
    stop_time = microsec_clock::local_time();
    gsresult = to_string(res);
    //    }
    //    catch (std::exception& e) {
    //        stop_time = microsec_clock::local_time();
    //        enum result res = lopt->last_optimize_result();
    //        gsresult = to_string(res) + ": " + e.what();
    //        cerr << e.what() << endl;
    //    }

    vector<complex<double>> fi(dim);
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            fi[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
        }
        double nrm = sqrt(abs(dot(fi, fi)));
        for (int n = 0; n <= nmax; n++) {
            x0[2 * (i * dim + n)] /= nrm;
            x0[2 * (i * dim + n) + 1] /= nrm;
        }
    }
//    cout << setprecision(10) << E0 << endl;
//        cout << setprecision(10) << x0 << endl;
//        exit(0);

    time_period period(start_time, stop_time);
    gsruntime = to_simple_string(period.length());
}

void DynamicsProblem::evolve(int nsteps) {
    start_time = microsec_clock::local_time();
    ptime eval_start_time = microsec_clock::local_time();
    integrator->setInput(x0, INTEGRATOR_X0);
    integrator->setInput(params, INTEGRATOR_P);
    //    integrator->evaluate();
    //    DMatrix xf = integrator->output(INTEGRATOR_XF);
    ptime eval_stop_time = microsec_clock::local_time();
    time_period eval_period(eval_start_time, eval_stop_time);

    vector<double> grad;
    E0 = E(x0, 0);

    //    vector<double> bv;
    //    bv = vector<vector<double>>(nsteps, vector<double>());
    bv.clear();

    vector<double> Es;

    ptime int_start_time = microsec_clock::local_time();
    integrator->reset();
    //    int npoints = 50;
    for (int j = 0; j <= nsteps; j++) {
        double ti = j * tf / nsteps;
        integrator->integrate(min(ti, tf));
        DMatrix x_i = integrator->output(INTEGRATOR_XF);
        vector<vector<complex<double>>> fi(L, vector<complex<double>>(dim));
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                fi[i][n] = complex<double>(x_i[2 * (i * dim + n)].getValue(), x_i[2 * (i * dim + n) + 1].getValue());
            }
            double nrm = sqrt(abs(dot(fi[i], fi[i])));
            for (int n = 0; n <= nmax; n++) {
                fi[i][n] /= nrm;
            }
        }
        vector<double> f_i;
        for (int i = 0; i < 2 * L * dim; i++) {
            f_i.push_back(x_i[i].getValue());
        }
        double E_i = E(f_i, ti);
        //    cout << setprecision(10) << E_i << endl;
        Es.push_back(E(f_i, min(ti, tf)));
        double Ui = Ufunc(DMatrix(ti))[0].getValue();
        vector<double> Ji(L);
        for (int i = 0; i < L; i++) {
            Ji[i] = Jfunc[i](DMatrix(ti))[0].getValue();
        }
        vector<complex<double>> bsci(L);
        for (int i = 0; i < L; i++) {
            bsci[i] = b(fi, i, Ji, Ui);
        }
        vector<double> bsi(L);
        for (int i = 0; i < L; i++) {
            bsi[i] = abs(bsci[i]);
        }
        bv.push_back(bsi);
        //        bv[j] = bsi;
        //        bv.push_back(bsi[0]);
    }
    //    cout << setprecision(10) << Es << endl;
    //    exit(0);

    //    integrator->reset();
    //    integrator->evaluate();
    integrator->integrate(tf);
    DMatrix xf = integrator->output(INTEGRATOR_XF);
    vector<double> ff;
    for (int i = 0; i < 2 * L * dim; i++) {
        ff.push_back(xf[i].getValue());
    }
    E1 = E(ff, tf);
    Q = E1 - E0;

    ptime int_stop_time = microsec_clock::local_time();
    time_period int_period(int_start_time, int_stop_time);
    stop_time = microsec_clock::local_time();
    time_period period(start_time, stop_time);
    runtime = to_simple_string(period.length());

    vector<vector<complex<double>>> fiv(L, vector<complex<double>>(dim));
    vector<vector<complex<double>>> ffv(L, vector<complex<double>>(dim));
    vector<double> pi(L);
    pd = 0;
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            fiv[i][n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
            ffv[i][n] = complex<double>(xf[2 * (i * dim + n)].getValue(), xf[2 * (i * dim + n) + 1].getValue());
        }
        double nrm = sqrt(abs(dot(ffv[i], ffv[i])));
        for (int n = 0; n <= nmax; n++) {
            ffv[i][n] /= nrm;
        }
        pi[i] = 1 - norm(dot(ffv[i], fiv[i]));
        pd += pi[i];
    }
    pd /= L;

}

double DynamicsProblem::E(const vector<double>& f, vector<double>& grad) {
    double E = 0;
    Ef.setInput(f.data(), NL_X);
    Ef.setInput(gsparams.data(), NL_P);
    Ef.evaluate();
    Ef.getOutput(E, NL_F);
    if (!grad.empty()) {
        Egradf.setInput(f.data(), NL_X);
        Egradf.setInput(gsparams.data(), NL_P);
        Egradf.evaluate();
        Egradf.output().getArray(grad.data(), grad.size(), DENSE);
    }
    return E;
}

double DynamicsProblem::E(const vector<double>& f, double t) {
    gsparams.back() = t;
    vector<double> g;
    return E(f, g);
}

SX DynamicsProblem::energync() {

    complex<SX> expth = complex<SX>(1, 0);
    complex<SX> expmth = ~expth;

    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<complex<SX>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += f[i][n].real() * f[i][n].real() + f[i][n].imag() * f[i][n].imag();
        }
    }

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2;

    for (int i = 0; i < L; i++) {

        int j1 = mod(i - 1);
        int j2 = mod(i + 1);

        Ei = complex<SX>(0, 0);
        Ej1 = complex<SX>(0, 0);
        Ej2 = complex<SX>(0, 0);

        for (int n = 0; n <= nmax; n++) {

            Ei += (0.5 * U0 * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                for (int m = 1; m <= nmax; m++) {
                    Ej1 += -J[j1] * expth * g(n, m) * ~f[i][n + 1] * ~f[j1][m - 1]
                            * f[i][n] * f[j1][m];
                    Ej2 += -J[i] * expmth * g(n, m) * ~f[i][n + 1] * ~f[j2][m - 1]
                            * f[i][n] * f[j2][m];
                }
            }

        }

        E += Ei / norm2[i];
        E += Ej1 / (norm2[i] * norm2[j1]);
        E += Ej2 / (norm2[i] * norm2[j2]);
    }

    return E.real();
}

SX DynamicsProblem::energya(bool normalize) {
    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<complex<SX>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += f[i][n].real() * f[i][n].real() + f[i][n].imag() * f[i][n].imag();
        }
    }

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

    for (int i = 0; i < L; i++) {
        int k1 = mod(i - 2);
        int j1 = mod(i - 1);
        int j2 = mod(i + 1);
        int k2 = mod(i + 2);

        Ei = complex<SX>(0, 0);
        Ej1 = complex<SX>(0, 0);
        Ej2 = complex<SX>(0, 0);
        Ej1j2 = complex<SX>(0, 0);
        Ej1k1 = complex<SX>(0, 0);
        Ej2k2 = complex<SX>(0, 0);

        for (int n = 0; n <= nmax; n++) {
            Ei += (0.5 * U0 * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                Ej1 += -J[j1] * (n + 1)* ~f[i][n + 1] * f[i][n] * ~f[j1][n] * f[j1][n + 1];
                Ej2 += -J[i] * (n + 1)* ~f[i][n + 1] * f[i][n] * ~f[j2][n] * f[j2][n + 1];

                for (int a = -nmax+1; a <= nmax; a++) {
                    if (a != 0) {
                        if (n + a + 1 >= 0 && n + a + 1 <= nmax) {
                            Ej1 += 0.5 * (n + 1) * (n + a + 1) * J[j1] * J[j1]*~f[i][n] * f[i][n]*~f[j1][n + a + 1] * f[j1][n + a + 1] / eps(U0, i, j1, a);
                            Ej2 += 0.5 * (n + 1) * (n + a + 1)* J[i] * J[i]*~f[i][n] * f[i][n]*~f[j2][n + a + 1] * f[j2][n + a + 1] / eps(U0, i, j2, a);
                        }
                        if (n - a + 1 >= 0 && n - a + 1 <= nmax) {
                            Ej1 += -0.5 * (n + 1) * (n - a + 1) * J[j1] * J[j1]*~f[i][n] * f[i][n]*~f[j1][n - a + 1] * f[j1][n - a + 1] / eps(U0, i, j1, a);
                            Ej2 += -0.5 * (n + 1) * (n - a + 1) * J[i] * J[i]*~f[i][n] * f[i][n]*~f[j2][n - a + 1] * f[j2][n - a + 1] / eps(U0, i, j2, a);
                        }
                    }
                }

                if (n < nmax - 1) {
                    Ej1 += 0.5*(n + 1)*(n + 2) * J[j1] * J[j1]*~f[i][n + 2] * f[i][n]*~f[j1][n] * f[j1][n + 2]*(1 / eps(U0, i, j1, 1) - 1 / eps(U0, i, j1, -1));
                    Ej2 += 0.5*(n + 1)*(n + 2) * J[i] * J[i]*~f[i][n + 2] * f[i][n]*~f[j2][n] * f[j2][n + 2]*(1 / eps(U0, i, j2, 1) - 1 / eps(U0, i, j2, -1));
                
                    for (int a = -nmax+1; a <= nmax; a++) {
                        if (a != 0) {
                            if (n+a >= 0 && n+a <= nmax && n+a+1 >= 0 && n+a+1 <= nmax && n-a+1 >= 0 && n-a+1 <= nmax && n-a+2 >= 0 && n-a+2 <= nmax) {
                                Ej1j2 += 0.5 * J[j1] * J[i] * ~f[i][n+2] * f[i][n] / eps(U0, i, j1, a) * ga(n, -a) * ga(n+1, a) * ~f[j2][n+a] * f[j2][n+a+1] * ~f[j1][n-a+1] * f[j1][n-a+2];
                                Ej1j2 += 0.5 * J[j1] * J[i] * f[i][n+2] * ~f[i][n] / eps(U0, i, j1, a) * ga(n, -a) * ga(n+1, a) * f[j2][n+a] * ~f[j2][n+a+1] * f[j1][n-a+1] * ~f[j1][n-a+2];
                                Ej1j2 += 0.5 * J[j1] * J[i] * ~f[i][n+2] * f[i][n] / eps(U0, i, j2, a) * ga(n, -a) * ga(n+1, a) * ~f[j1][n+a] * f[j1][n+a+1] * ~f[j2][n-a+1] * f[j2][n-a+2];
                                Ej1j2 += 0.5 * J[j1] * J[i] * f[i][n+2] * ~f[i][n] / eps(U0, i, j2, a) * ga(n, -a) * ga(n+1, a) * f[j1][n+a] * ~f[j1][n+a+1] * f[j2][n-a+1] * ~f[j2][n-a+2];
                            }
                            if (n-a >= 0 && n-a <= nmax && n-a+1 >= 0 && n-a+1 <= nmax && n+a+1 >= 0 && n+a+1 <= nmax && n+a+2 >= 0 && n+a+2 <= nmax) {
                                Ej1j2 -= 0.5 * J[j1] * J[i] * ~f[i][n+2] * f[i][n] / eps(U0, i, j1, -a) * ga(n, a) * ga(n+1, -a) * ~f[j2][n-a] * f[j2][n-a+1] * ~f[j1][n+a+1] * f[j1][n+a+2];
                                Ej1j2 -= 0.5 * J[j1] * J[i] * f[i][n+2] * ~f[i][n] / eps(U0, i, j1, -a) * ga(n, a) * ga(n+1, -a) * f[j2][n-a] * ~f[j2][n-a+1] * f[j1][n+a+1] * ~f[j1][n+a+2];
                                Ej1j2 -= 0.5 * J[j1] * J[i] * ~f[i][n+2] * f[i][n] / eps(U0, i, j2, -a) * ga(n, a) * ga(n+1, -a) * ~f[j1][n-a] * f[j1][n-a+1] * ~f[j2][n+a+1] * f[j2][n+a+2];
                                Ej1j2 -= 0.5 * J[j1] * J[i] * f[i][n+2] * ~f[i][n] / eps(U0, i, j2, -a) * ga(n, a) * ga(n+1, -a) * f[j1][n-a] * ~f[j1][n-a+1] * f[j2][n+a+1] * ~f[j2][n+a+2];
                            }
                            
                            if (n+a >= 0 && n+a <= nmax && n+a+1 >= 0 && n+a+1 <= nmax) {
                                Ej1k1 += 0.5 * J[j1] * J[k1] * ~f[j1][n] * f[j1][n] / eps(U0, i, j1, a) * ga(n, -a) * ga(n+a, a) * ~f[k1][n+a] * f[k1][n+a+1] * ~f[i][n+a+1] * f[i][n+a];
                                Ej1k1 += 0.5 * J[j1] * J[k1] * f[j1][n] * ~f[j1][n] / eps(U0, i, j1, a) * ga(n, -a) * ga(n+a, a) * f[k1][n+a] * ~f[k1][n+a+1] * f[i][n+a+1] * ~f[i][n+a];
                                Ej2k2 += 0.5 * J[j2] * J[i] * ~f[j2][n] * f[j2][n] / eps(U0, i, j2, a) * ga(n, -a) * ga(n+a, a) * ~f[k2][n+a] * f[k2][n+a+1] * ~f[i][n+a+1] * f[i][n+a];
                                Ej2k2 += 0.5 * J[j2] * J[i] * f[j2][n] * ~f[j2][n] / eps(U0, i, j2, a) * ga(n, -a) * ga(n+a, a) * f[k2][n+a] * ~f[k2][n+a+1] * f[i][n+a+1] * ~f[i][n+a];
                            }
                            if (n-a >= 0 && n-a <= nmax && n-a+1 >= 0 && n-a+1 <= nmax) {
                                Ej1k1 -= 0.5 * J[j1] * J[k1] * ~f[j1][n] * f[j1][n] / eps(U0, i, j1, -a) * ga(n, a) * ga(n-a, -a) * ~f[k1][n-a] * f[k1][n-a+1] * ~f[i][n-a+1] * f[i][n-a];
                                Ej1k1 -= 0.5 * J[j1] * J[k1] * f[j1][n] * ~f[j1][n] / eps(U0, i, j1, -a) * ga(n, a) * ga(n-a, -a) * f[k1][n-a] * ~f[k1][n-a+1] * f[i][n-a+1] * ~f[i][n-a];
                                Ej2k2 -= 0.5 * J[j2] * J[i] * ~f[j2][n] * f[j2][n] / eps(U0, i, j2, -a) * ga(n, a) * ga(n-a, -a) * ~f[k2][n-a] * f[k2][n-a+1] * ~f[i][n-a+1] * f[i][n-a];
                                Ej2k2 -= 0.5 * J[j2] * J[i] * f[j2][n] * ~f[j2][n] / eps(U0, i, j2, -a) * ga(n, a) * ga(n-a, -a) * f[k2][n-a] * ~f[k2][n-a+1] * f[i][n-a+1] * ~f[i][n-a];
                            }
                        }
                    }
                }
            }
        }

        if (normalize) {
        Ei /= norm2[i];
        Ej1 /= norm2[i] * norm2[j1];
        Ej2 /= norm2[i] * norm2[j2];
        Ej1j2 /= norm2[i] * norm2[j1] * norm2[j2];
        Ej1k1 /= norm2[i] * norm2[j1] * norm2[k1];
        Ej2k2 /= norm2[i] * norm2[j2] * norm2[k2];
        }

        E += Ei;
        E += Ej1;
        E += Ej2;
        E += Ej1j2;
        E += Ej1k1;
        E += Ej2k2;
    }

    return E.real();
}

SX DynamicsProblem::energy() {

    complex<SX> expth = complex<SX>(1, 0);
    complex<SX> expmth = ~expth;
    complex<SX> exp2th = expth*expth;
    complex<SX> expm2th = ~exp2th;

    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<complex<SX>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += f[i][n].real() * f[i][n].real() + f[i][n].imag() * f[i][n].imag();
        }
    }

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

    for (int i = 0; i < L; i++) {

        int k1 = mod(i - 2);
        int j1 = mod(i - 1);
        int j2 = mod(i + 1);
        int k2 = mod(i + 2);

        Ei = complex<SX>(0, 0);
        Ej1 = complex<SX>(0, 0);
        Ej2 = complex<SX>(0, 0);
        Ej1j2 = complex<SX>(0, 0);
        Ej1k1 = complex<SX>(0, 0);
        Ej2k2 = complex<SX>(0, 0);

        for (int n = 0; n <= nmax; n++) {
            Ei += (0.5 * (U0 + dU[i]) * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                Ej1 += -J[j1] * expth * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                        * f[i][n] * f[j1][n + 1];
                Ej2 += -J[i] * expmth * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                        * f[j2][n + 1];

                if (n > 0) {
                    Ej1 += 0.5 * J[j1] * J[j1] * exp2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1];
                    Ej2 += 0.5 * J[i] * J[i] * expm2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1];
                }
                if (n < nmax - 1) {
                    Ej1 -= 0.5 * J[j1] * J[j1] * exp2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j1][n] * f[i][n] * f[j1][n + 2];
                    Ej2 -= 0.5 * J[i] * J[i] * expm2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j2][n] * f[i][n] * f[j2][n + 2];
                }

                if (n > 1) {
                    Ej1 += -J[j1] * J[j1] * exp2th * g(n, n - 1) * g(n - 1, n)
                            * (eps(dU, i, j1, n, n - 1, i, j1, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                            * ~f[i][n + 1] * ~f[j1][n - 2] * f[i][n - 1] * f[j1][n];
                    Ej2 += -J[i] * J[i] * expm2th * g(n, n - 1) * g(n - 1, n)
                            * (eps(dU, i, j2, n, n - 1, i, j2, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                            * ~f[i][n + 1] * ~f[j2][n - 2] * f[i][n - 1] * f[j2][n];
                }
                if (n < nmax - 2) {
                    Ej1 -= -J[j1] * J[j1] * exp2th * g(n, n + 3) * g(n + 1, n + 2)
                            * (eps(dU, i, j1, n, n + 3, i, j1, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                            * ~f[i][n + 2] * ~f[j1][n + 1] * f[i][n] * f[j1][n + 3];
                    Ej2 -= -J[i] * J[i] * expm2th * g(n, n + 3) * g(n + 1, n + 2)
                            * (eps(dU, i, j2, n, n + 3, i, j2, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                            * ~f[i][n + 2] * ~f[j2][n + 1] * f[i][n] * f[j2][n + 3];
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1) {
                        Ej1 += 0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1] -
                                ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
                        Ej2 += 0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1] -
                                ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);

                        Ej1 += J[j1] * expth * g(n, m) * (eps(dU, i, j1, n, m) / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
                        Ej2 += J[i] * expmth * g(n, m) * (eps(dU, i, j2, n, m) / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];

                        if (n != m - 3 && m > 1 && n < nmax - 1) {
                            Ej1 += -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                    * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                            Ej2 += -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                    * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                        }
                        if (n != m + 1 && n > 0 && m < nmax) {
                            Ej1 -= -0.5 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                            Ej2 -= -0.5 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                        }

                        if (n > 0) {
                            Ej1j2 += -J[j1] * J[i] * g(n, m) * g(n - 1, n)
                                    * (eps(dU, i, j1, n, m, i, j2, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][n - 1]
                                    * f[i][n - 1] * f[j1][m] * f[j2][n];
                            Ej1j2 += -J[i] * J[j1] * g(n, m) * g(n - 1, n)
                                    * (eps(dU, i, j2, n, m, i, j1, n - 1, n) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n - 1, n))))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][n - 1]
                                    * f[i][n - 1] * f[j2][m] * f[j1][n];
                        }
                        if (n < nmax - 1) {
                            Ej1j2 -= -J[j1] * J[i] * g(n, m) * g(n + 1, n + 2)
                                    * (eps(dU, i, j1, n, m, i, j2, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                    * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][n + 1]
                                    * f[i][n] * f[j1][m] * f[j2][n + 2];
                            Ej1j2 -= -J[i] * J[j1] * g(n, m) * g(n + 1, n + 2)
                                    * (eps(dU, i, j2, n, m, i, j1, n + 1, n + 2) / (eps(U0, n, m) * (eps(U0, n, m) + eps(U0, n + 1, n + 2))))
                                    * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][n + 1]
                                    * f[i][n] * f[j2][m] * f[j1][n + 2];
                        }

                        Ej1 += -0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1)
                                * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                                * (~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m] -
                                ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1]);
                        Ej2 += -0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1)
                                * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, n + 1)))
                                * (~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m] -
                                ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1]);

                        for (int q = 1; q <= nmax; q++) {
                            if (n < nmax - 1 && n != q - 2) {
                                Ej1j2 += -0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                        * f[i][n] * f[j1][m] * f[j2][q];
                                Ej1j2 += -0.5 * J[i] * J[j1] * g(n, m) * g(n + 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, q)))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                        * f[i][n] * f[j2][m] * f[j1][q];
                            }
                            if (n > 0 && n != q) {
                                Ej1j2 -= -0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][q - 1]
                                        * f[i][n - 1] * f[j1][m] * f[j2][q];
                                Ej1j2 -= -0.5 * J[i] * J[j1] * g(n, m) * g(n - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][q - 1]
                                        * f[i][n - 1] * f[j2][m] * f[j1][q];
                            }

                            if (m != q) {
                                Ej1k1 += -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][q - 1]
                                        * f[i][n] * f[j1][m] * f[k1][q];
                                Ej2k2 += -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][q - 1]
                                        * f[i][n] * f[j2][m] * f[k2][q];
                                Ej1k1 -= -0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][q - 1]
                                        * f[i][n] * f[j1][m - 1] * f[k1][q];
                                Ej2k2 -= -0.5 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][q - 1]
                                        * f[i][n] * f[j2][m - 1] * f[k2][q];
                            }

                        }

                        for (int p = 0; p < nmax; p++) {

                            if (p != n - 1 && 2 * n - m == p && n > 0) {
                                Ej1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                                Ej1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                            }
                            if (p != n + 1 && 2 * n - m == p - 2 && n < nmax - 1) {
                                Ej1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n] * f[j1][m] * f[j2][p + 1];
                                Ej1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n] * f[j2][m] * f[j1][p + 1];
                            }

                            if (p != n - 1 && 2 * n - m != p && n > 0) {
                                Ej1j2 += -0.25 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1)
                                        * (eps(dU, i, j1, n, m, i, j2, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                                Ej1j2 += -0.25 * J[i] * J[j1] * g(n, m) * g(n - 1, p + 1)
                                        * (eps(dU, i, j2, n, m, i, j1, p, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, p + 1))))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                            }
                            if (p != n + 1 && 2 * n - m != p - 2 && n < nmax - 1) {
                                Ej1j2 -= -0.25 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1)
                                        * (eps(dU, i, j1, n, m, i, j2, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n] * f[j1][m] * f[j2][p + 1];
                                Ej1j2 -= -0.25 * J[i] * J[j1] * g(n, m) * g(n + 1, p + 1)
                                        * (eps(dU, i, j2, n, m, i, j1, p, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, p + 1))))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n] * f[j2][m] * f[j1][p + 1];
                            }

                            if (p != m - 1 && n != p) {
                                Ej1k1 += -0.25 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, p + 1)
                                        * (eps(dU, i, j1, n, m, j1, k1, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                        * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][p] * f[i][n] * f[j1][m - 1] * f[k1][p + 1] -
                                        ~f[i][n + 1] * ~f[j1][m] * ~f[k1][p] * f[i][n] * f[j1][m] * f[k1][p + 1]);
                                Ej2k2 += -0.25 * J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, p + 1)
                                        * (eps(dU, i, j2, n, m, j2, k2, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                        * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][p] * f[i][n] * f[j2][m - 1] * f[k2][p + 1] -
                                        ~f[i][n + 1] * ~f[j2][m] * ~f[k2][p] * f[i][n] * f[j2][m] * f[k2][p + 1]);
                            }
                        }

                        Ej1k1 += 0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][n]
                                * f[i][n] * f[j1][m - 1] * f[k1][n + 1] -
                                ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
                                * f[i][n] * f[j1][m] * f[k1][n + 1]);
                        Ej2k2 += 0.5 * J[j2] * J[i] * expm2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][n]
                                * f[i][n] * f[j2][m - 1] * f[k2][n + 1] -
                                ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
                                * f[i][n] * f[j2][m] * f[k2][n + 1]);

                        Ej1k1 += -J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, m)
                                * (eps(dU, i, j1, n, m, j1, k1, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][m - 1] * f[i][n] * f[j1][m - 1] * f[k1][m] -
                                ~f[i][n + 1] * ~f[j1][m] * ~f[k1][m - 1] * f[i][n] * f[j1][m] * f[k1][m]);
                        Ej2k2 += -J[i] * J[j2] * expm2th * g(n, m) * g(m - 1, m)
                                * (eps(dU, i, j2, n, m, j2, k2, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][m - 1] * f[i][n] * f[j2][m - 1] * f[k2][m] -
                                ~f[i][n + 1] * ~f[j2][m] * ~f[k2][m - 1] * f[i][n] * f[j2][m] * f[k2][m]);

                        if (m != n - 1 && n != m && m < nmax && n > 0) {
                            Ej1 += -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j1, n, m, i, j1, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                            Ej2 += -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j2, n, m, i, j2, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                        }
                        if (n != m - 3 && n != m - 2 && n < nmax - 1 && m > 1) {
                            Ej1 -= -0.25 * J[j1] * J[j1] * exp2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j1, n, m, i, j1, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                    * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                            Ej2 -= -0.25 * J[i] * J[i] * expm2th * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j2, n, m, i, j2, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                    * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                        }
                    }
                }
            }
        }

        Ei /= norm2[i];
        Ej1 /= norm2[i] * norm2[j1];
        Ej2 /= norm2[i] * norm2[j2];
        Ej1j2 /= norm2[i] * norm2[j1] * norm2[j2];
        Ej1k1 /= norm2[i] * norm2[j1] * norm2[k1];
        Ej2k2 /= norm2[i] * norm2[j2] * norm2[k2];

        E += Ei;
        E += Ej1;
        E += Ej2;
        E += Ej1j2;
        E += Ej1k1;
        E += Ej2k2;
    }

    return E.real();
}

SX DynamicsProblem::canonicala() {

    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<complex<SX>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += f[i][n].real() * f[i][n].real() + f[i][n].imag() * f[i][n].imag();
        }
    }

    complex<SX> S = complex<SX>(0, 0);

    complex<SX> Sj1, Sj2;

    for (int i = 0; i < L; i++) {

        int j1 = mod(i - 1);
        int j2 = mod(i + 1);

        Sj1 = complex<SX>(0, 0);
        Sj2 = complex<SX>(0, 0);

        for (int a = -nmax; a <= nmax; a++) {
            if (a != 0) {
                for (int n = 0; n < nmax; n++) {
                    if (n-a >= 0 && n-a <= nmax && n-a+1 >= 0 && n-a+1 <= nmax) {
                        Sj1 += -J[j1] / eps(U0, i, j1, a) * ga(n, a) * ~f[i][n+1] * f[i][n] * ~f[j1][n-a] * f[j1][n-a+1];
                        Sj2 += -J[i] / eps(U0, i, j2, a) * ga(n, a) * ~f[i][n+1] * f[i][n] * ~f[j2][n-a] * f[j2][n-a+1];
                    }
                }
            }
        }

        Sj1 /= norm2[i] * norm2[j1];
        Sj2 /= norm2[i] * norm2[j2];
        
        S += Sj1;
        S += Sj2;
    }

    return S.imag();
}

SX DynamicsProblem::canonical() {

    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<complex<SX>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += f[i][n].real() * f[i][n].real() + f[i][n].imag() * f[i][n].imag();
        }
    }

    complex<SX> S = complex<SX>(0, 0);

    complex<SX> Sj1, Sj2;

    for (int i = 0; i < L; i++) {

        int j1 = mod(i - 1);
        int j2 = mod(i + 1);

        Sj1 = complex<SX>(0, 0);
        Sj2 = complex<SX>(0, 0);

        for (int n = 0; n <= nmax; n++) {
            if (n < nmax) {
                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1) {
                        Sj1 += -J[j1] * g(n, m) * (1 / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
                        Sj2 += -J[i] * g(n, m) * (1 / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];

                    }
                }
            }
        }

        Sj1 /= norm2[i] * norm2[j1];
        Sj2 /= norm2[i] * norm2[j2];

        S += Sj1;
        S += Sj2;
    }

    return S.real();
}

SX DynamicsProblem::energy0() {

    complex<SX> expth = complex<SX>(1, 0);
    complex<SX> expmth = ~expth;
    complex<SX> exp2th = expth*expth;
    complex<SX> expm2th = ~exp2th;

    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<complex<SX>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += f[i][n].real() * f[i][n].real() + f[i][n].imag() * f[i][n].imag();
        }
    }

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

    for (int i = 0; i < L; i++) {

        int k1 = mod(i - 2);
        int j1 = mod(i - 1);
        int j2 = mod(i + 1);
        int k2 = mod(i + 2);

        Ei = complex<SX>(0, 0);
        Ej1 = complex<SX>(0, 0);
        Ej2 = complex<SX>(0, 0);
        Ej1j2 = complex<SX>(0, 0);
        Ej1k1 = complex<SX>(0, 0);
        Ej2k2 = complex<SX>(0, 0);


        for (int n = 0; n <= nmax; n++) {
            Ei += (0.5 * U0 * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                Ej1 += -J[j1] * expth * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                        * f[i][n] * f[j1][n + 1];
                Ej2 += -J[i] * expmth * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                        * f[j2][n + 1];

                if (n > 0) {
                    Ej1 += 0.5 * J[j1] * J[j1] * exp2th * g(n, n) * g(n - 1, n + 1)
                            * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1]
                            * (1 / eps(U0, n, n) - 1 / eps(U0, n - 1, n + 1));
                    Ej2 += 0.5 * J[i] * J[i] * expm2th * g(n, n) * g(n - 1, n + 1)
                            * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1]
                            * (1 / eps(U0, n, n) - 1 / eps(U0, n - 1, n + 1));
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1) {
                        Ej1 += 0.5 * (J[j1] * J[j1] / eps(U0, n, m)) * g(n, m)
                                * g(m - 1, n + 1)
                                * (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1]
                                - ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
                        Ej2 += 0.5 * (J[i] * J[i] / eps(U0, n, m)) * g(n, m)
                                * g(m - 1, n + 1)
                                * (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1]
                                - ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);
                    }
                }

                if (n > 0) {
                    Ej1j2 += 0.5 * (J[j1] * J[i] / eps(U0, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[j2][n]
                            * f[i][n - 1] * f[j1][n] * f[j2][n + 1];
                    Ej1j2 += 0.5 * (J[i] * J[j1] / eps(U0, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[j1][n]
                            * f[i][n - 1] * f[j2][n] * f[j1][n + 1];
                    Ej1k1 += 0.5 * (J[j1] * J[k1] / eps(U0, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[k1][n]
                            * f[i][n] * f[j1][n + 1] * f[k1][n - 1];
                    Ej2k2 += 0.5 * (J[i] * J[j2] / eps(U0, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[k2][n]
                            * f[i][n] * f[j2][n + 1] * f[k2][n - 1];
                    Ej1j2 -= 0.5 * (J[j1] * J[i] / eps(U0, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                            * ~f[j2][n - 1] * f[i][n - 1] * f[j1][n + 1] * f[j2][n];
                    Ej1j2 -= 0.5 * (J[i] * J[j1] / eps(U0, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n]
                            * ~f[j1][n - 1] * f[i][n - 1] * f[j2][n + 1] * f[j1][n];
                    Ej1k1 -= 0.5 * (J[j1] * J[k1] / eps(U0, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j1][n - 1]
                            * ~f[k1][n + 1] * f[i][n - 1] * f[j1][n + 1] * f[k1][n];
                    Ej2k2 -= 0.5 * (J[i] * J[j2] / eps(U0, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j2][n - 1]
                            * ~f[k2][n + 1] * f[i][n - 1] * f[j2][n + 1] * f[k2][n];
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1 && n < nmax) {
                        Ej1j2 += 0.5 * J[j1] * J[i] * exp2th / eps(U0, n, m)
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
                                * ~f[j2][m] * f[i][n + 1] * f[j1][m] * f[j2][m - 1];
                        Ej1j2 += 0.5 * J[i] * J[j1] * expm2th / eps(U0, n, m)
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
                                * ~f[j1][m] * f[i][n + 1] * f[j2][m] * f[j1][m - 1];
                        Ej1k1 += 0.5 * J[j1] * J[k1] * exp2th / eps(U0, n, m)
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
                                * ~f[k1][n] * f[i][n] * f[j1][m - 1] * f[k1][n + 1];
                        Ej2k2 += 0.5 * J[i] * J[j2] * expm2th / eps(U0, n, m)
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
                                * ~f[k2][n] * f[i][n] * f[j2][m - 1] * f[k2][n + 1];
                        Ej1j2 -= 0.5 * J[j1] * J[i] * exp2th / eps(U0, n, m)
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j1][m - 1]
                                * ~f[j2][m] * f[i][n] * f[j1][m] * f[j2][m - 1];
                        Ej1j2 -= 0.5 * J[i] * J[j1] * expm2th / eps(U0, n, m)
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j2][m - 1]
                                * ~f[j1][m] * f[i][n] * f[j2][m] * f[j1][m - 1];
                        Ej1k1 -= 0.5 * J[j1] * J[k1] * exp2th / eps(U0, n, m)
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m]
                                * ~f[k1][n] * f[i][n] * f[j1][m] * f[k1][n + 1];
                        Ej2k2 -= 0.5 * J[i] * J[j2] * expm2th / eps(U0, n, m)
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m]
                                * ~f[k2][n] * f[i][n] * f[j2][m] * f[k2][n + 1];
                    }
                }
            }

        }

        E += Ei / norm2[i];

        E += Ej1 / (norm2[i] * norm2[j1]);
        E += Ej2 / (norm2[i] * norm2[j2]);

        E += Ej1j2 / (norm2[i] * norm2[j1] * norm2[j2]);
        E += Ej1k1 / (norm2[i] * norm2[j1] * norm2[k1]);
        E += Ej2k2 / (norm2[i] * norm2[j2] * norm2[k2]);
    }

    return E.real();
}

