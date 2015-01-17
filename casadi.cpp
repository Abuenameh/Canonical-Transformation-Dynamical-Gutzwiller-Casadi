#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

using namespace boost;
using namespace boost::filesystem;

#include <casadi/core/function/parallelizer.hpp>

#include "casadi.hpp"

double energyfunc(const vector<double>& x, vector<double>& grad, void *data) {
    //    DynamicsProblem* prob = static_cast<DynamicsProblem*> (data);
    //    return prob->E(x, grad);
    return DynamicsProblem::E(x, grad);
}

double JW(double W) {
    return alpha * (W * W) / (Ng * Ng + W * W);
}

double JWij(double Wi, double Wj) {
    return alpha * (Wi * Wj) / (sqrt(Ng * Ng + Wi * Wi) * sqrt(Ng * Ng + Wj * Wj));
}

double UW(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
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
        res += ~v[i] * w[i];
    }
    return res;
}

complex<double> b0(vector<vector<complex<double>>>& f, int i) {
    complex<double> bi = 0;
    //    cout << f << endl;
    //    for (int n = 1; n <= nmax; n++) {
    //        cout << i << " " << n << endl;
    //        complex<double> qwe = f[i][n];
    //        cout << qwe << endl;
    //        cout << "~ " << i << " " << n-1 << endl;
    //        complex<double> asd = ~f[i][n-1];
    //        cout << asd << endl;
    //    }
    for (int n = 1; n <= nmax; n++) {
        bi += sqrt(1.0 * n) * ~f[i][n - 1] * f[i][n];
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
                bi += -J[i] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * n + 1) * ~f[j2][m - 1] * f[j2][m] * (~f[i][n + 1] * f[i][n + 1] - ~f[i][n] * f[i][n]);
                bi += -J[j1] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * n + 1) * ~f[j1][m - 1] * f[j1][m] * (~f[i][n + 1] * f[i][n + 1] - ~f[i][n] * f[i][n]);

                if (m < nmax) {
                    bi += -J[i] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * m + 1) * ~f[j2][n + 1] * f[j2][n] * ~f[i][m - 1] * f[i][m + 1];
                    bi += -J[j1] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * m + 1) * ~f[j1][n + 1] * f[j1][n] * ~f[i][m - 1] * f[i][m + 1];
                }
                if (m > 1) {
                    bi += J[i] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * m - 1) * ~f[j2][n + 1] * f[j2][n] * ~f[i][m - 2] * f[i][m];
                    bi += J[j1] * g2(n, m) / eps(U, n, m) * sqrt(1.0 * m - 1) * ~f[j1][n + 1] * f[j1][n] * ~f[i][m - 2] * f[i][m];
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
                bi -= (n + 1) * sqrt(1.0 * m * (m - 1) * (p + 1)) * ~f[i][n] * ~f[a][p + 1] * ~f[k][m - 2] * f[i][n] * f[a][p] * f[k][m];
                bi += (n + 1) * sqrt(1.0 * m * (m - 1) * (p + 1)) * ~f[i][n + 1] * ~f[a][p + 1] * ~f[k][m - 2] * f[i][n + 1] * f[a][p] * f[k][m];
            }
            if (m < nmax) {
                bi += (n + 1) * sqrt(1.0 * m * (m + 1) * (p + 1)) * ~f[i][n] * ~f[a][p + 1] * ~f[k][m - 1] * f[i][n] * f[a][p] * f[k][m + 1];
                bi -= (n + 1) * sqrt(1.0 * m * (m + 1) * (p + 1)) * ~f[i][n + 1] * ~f[a][p + 1] * ~f[k][m - 1] * f[i][n + 1] * f[a][p] * f[k][m + 1];
            }
        }
        else {
            if (p == m - 1) {
                if (m < nmax) {
                    bi += m * (n + 1) * sqrt(1.0 * m + 1) * ~f[i][n] * ~f[k][m] * f[i][n] * f[k][m + 1];
                    bi -= m * (n + 1) * sqrt(1.0 * m + 1) * ~f[i][n + 1] * ~f[k][m] * f[i][n + 1] * f[k][m + 1];
                }
            }
            else if (p == m - 2) {
                bi -= (m - 1) * (n + 1) * sqrt(1.0 * m) * ~f[i][n] * ~f[k][m - 1] * f[i][n] * f[k][m];
                bi += (m - 1) * (n + 1) * sqrt(1.0 * m) * ~f[i][n + 1] * ~f[k][m - 1] * f[i][n + 1] * f[k][m];
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
                bi += sqrt(1.0 * (p + 1) * (n + 1) * m * (m + 1) * q) * ~f[i][n + 1] * ~f[a][p + 1] * ~f[k][m - 2] * f[i][n] * f[a][p] * f[k][m + 1];
            }
            if (q == m + 2) {
                bi -= sqrt(1.0 * (p + 1) * (n + 1) * m * (m + 1) * q) * ~f[i][n + 1] * ~f[a][p + 1] * ~f[k][m - 1] * f[i][n] * f[a][p] * f[k][m + 2];
            }
            if (q == m - 2) {
                bi -= sqrt(1.0 * (p + 1) * (n + 1) * (m - 1) * m * q) * ~f[i][n + 1] * ~f[a][p + 1] * ~f[k][m - 3] * f[i][n] * f[a][p] * f[k][m];
            }
            if (q == m + 1 && m >= 2) {
                bi += sqrt(1.0 * (p + 1) * (n + 1) * (m - 1) * m * q) * ~f[i][n + 1] * ~f[a][p + 1] * ~f[k][m - 2] * f[i][n] * f[a][p] * f[k][m + 1];
            }
        }
        else if (p == n + 1) {
            if (q == m - 1 && n < nmax - 1 && m < nmax) {
                bi += sqrt(1.0 * (n + 2) * (n + 1) * m * (m + 1) * (m - 1)) * ~f[i][n + 2] * ~f[k][m - 2] * f[i][n] * f[k][m + 1];
            }
            if (q == m + 2 && n < nmax - 1) {
                bi -= sqrt(1.0 * (n + 2) * (n + 1) * m * (m + 1) * (m + 2)) * ~f[i][n + 2] * ~f[k][m - 1] * f[i][n] * f[k][m + 2];
            }
            if (q == m - 2 && n < nmax - 1) {
                bi -= sqrt(1.0 * (n + 2) * (n + 1) * (m - 1) * m * (m - 2)) * ~f[i][n + 2] * ~f[k][m - 3] * f[i][n] * f[k][m];
            }
            if (q == m + 1 && n < nmax - 1 && m >= 2) {
                bi += sqrt(1.0 * (n + 2) * (n + 1) * (m - 1) * m * (m + 1)) * ~f[i][n + 2] * ~f[k][m - 2] * f[i][n] * f[k][m + 1];
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
                bi += sqrt(1.0 * q * (n + 1) * (n + 2) * m * (m + 1)) * ~f[i][n + 2] * ~f[b][q - 1] * ~f[k][m - 1] * f[i][n] * f[b][q] * f[k][m + 1];
            }
            if (p == n + 1 && m >= 2) {
                bi -= sqrt(1.0 * q * (n + 1) * (n + 2) * (m - 1) * m) * ~f[i][n + 2] * ~f[b][q - 1] * ~f[k][m - 2] * f[i][n] * f[b][q] * f[k][m];
            }
            if (p == n - 1 && m < nmax) {
                bi -= sqrt(1.0 * q * n * (n + 1) * m * (m + 1)) * ~f[i][n + 1] * ~f[b][q - 1] * ~f[k][m - 1] * f[i][n - 1] * f[b][q] * f[k][m + 1];
            }
            if (p == n - 1 && m >= 2) {
                bi += sqrt(1.0 * q * n * (n + 1) * (m - 1) * m) * ~f[i][n + 1] * ~f[b][q - 1] * ~f[k][m - 2] * f[i][n - 1] * f[b][q] * f[k][m];
            }
        }
        else {
            if (q == m + 2 && p == n + 1) {
                bi += sqrt(1.0 * (n + 1) * (n + 2) * m * (m + 1) * (m + 2)) * ~f[i][n + 2] * ~f[k][m - 1] * f[i][n] * f[k][m + 2];
            }
            if (q == m + 1 && m >= 2 && p == n + 1) {
                bi -= sqrt(1.0 * (n + 1) * (n + 2) * (m - 1) * m * (m + 1)) * ~f[i][n + 2] * ~f[k][m - 2] * f[i][n] * f[k][m + 1];
            }
            if (q == m + 2 && p == n - 1) {
                bi -= sqrt(1.0 * n * (n + 1) * m * (m + 1) * (m + 2)) * ~f[i][n + 1] * ~f[k][m - 1] * f[i][n - 1] * f[k][m + 2];
            }
            if (q == m + 1 && m >= 2 && p == n - 1) {
                bi += sqrt(1.0 * n * (n + 1) * (m - 1) * m * (m + 1)) * ~f[i][n + 1] * ~f[k][m - 2] * f[i][n - 1] * f[k][m + 1];
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
                bi += m * sqrt(1.0 * (n + 1) * q * (m + 1)) * ~f[i][n + 1] * ~f[b][q - 1] * ~f[k][m] * f[i][n] * f[b][q] * f[k][m + 1];
            }
            if (p == m - 2) {
                bi -= (m - 1) * sqrt(1.0 * (n + 1) * q * m) * ~f[i][n + 1] * ~f[b][q - 1] * ~f[k][m - 1] * f[i][n] * f[b][q] * f[k][m];
            }
            if (p == m) {
                bi -= (m + 1) * sqrt(1.0 * (n + 1) * q * m) * ~f[i][n + 1] * ~f[b][q - 1] * ~f[k][m - 1] * f[i][n] * f[b][q] * f[k][m];
            }
            if (p == m - 1 && m >= 2) {
                bi += m * sqrt(1.0 * (n + 1) * q * (m - 1)) * ~f[i][n + 1] * ~f[b][q - 1] * ~f[k][m - 2] * f[i][n] * f[b][q] * f[k][m - 1];
            }
        }
        else if (n == q - 1) {
            if (p == m - 1 && m < nmax) {
                bi += (n + 1) * m * sqrt(1.0 * (m + 1)) * ~f[i][n + 1] * ~f[k][m] * f[i][n + 1] * f[k][m + 1];
            }
            if (p == m - 2) {
                bi -= (n + 1) * (m - 1) * sqrt(1.0 * m) * ~f[i][n + 1] * ~f[k][m - 1] * f[i][n + 1] * f[k][m];
            }
            if (p == m) {
                bi -= (n + 1) * (m + 1) * sqrt(1.0 * m) * ~f[i][n + 1] * ~f[k][m - 1] * f[i][n + 1] * f[k][m];
            }
            if (p == m - 1 && m >= 2) {
                bi += (n + 1) * m * sqrt(1.0 * (m - 1)) * ~f[i][n + 1] * ~f[k][m - 2] * f[i][n + 1] * f[k][m - 1];
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
                bi += 2 * (n + 1) * sqrt(1.0 * m * (p + 1) * (n + 1)) * ~f[j][m - 1] * ~f[a][p + 1] * ~f[k][n] * f[j][m] * f[a][p] * f[k][n + 1];
            }
            if (q == n) {
                bi -= (n + 1) * sqrt(1.0 * m * (p + 1) * n) * ~f[j][m - 1] * ~f[a][p + 1] * ~f[k][n - 1] * f[j][m] * f[a][p] * f[k][n];
            }
            if (q == n + 2) {
                bi -= (n + 1) * sqrt(1.0 * m * (p + 1) * (n + 2)) * ~f[j][m - 1] * ~f[a][p + 1] * ~f[k][n + 1] * f[j][m] * f[a][p] * f[k][n + 2];
            }
        }
        else if (p == m - 1) {
            if (q == n + 1) {
                bi += 2 * (n + 1) * m * sqrt(1.0 * (n + 1)) * ~f[j][p + 1] * ~f[k][n] * f[j][m] * f[k][n + 1];
            }
            if (q == n) {
                bi -= (n + 1) * m * sqrt(1.0 * n) * ~f[j][p + 1] * ~f[k][n - 1] * f[j][m] * f[k][n];
            }
            if (q == n + 2) {
                bi -= (n + 1) * m * sqrt(1.0 * (n + 2)) * ~f[j][p + 1] * ~f[k][n + 1] * f[j][m] * f[k][n + 2];
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
                bi += (n + 1) * sqrt(1.0 * (p + 1) * (m - 1) * m) * ~f[j][m - 2] * ~f[a][p + 1] * f[j][m] * f[a][p] * (~f[k][n + 1] * f[k][n + 1] - ~f[k][n] * f[k][n]);
            }
            if (q == m + 1) {
                bi -= (n + 1) * sqrt(1.0 * (p + 1) * (m + 1) * m) * ~f[j][m - 1] * ~f[a][p + 1] * f[j][m + 1] * f[a][p] * (~f[k][n + 1] * f[k][n + 1] - ~f[k][n] * f[k][n]);
            }
        }
        else {
            if (p == n + 1 && q == m - 1) {
                bi += (n + 1) * sqrt(1.0 * m * (m - 1) * (n + 2)) * ~f[j][m - 2] * ~f[k][n + 2] * f[j][m] * f[k][n + 1];
            }
            if (p == n + 1 && q == m + 1) {
                bi -= (n + 1) * sqrt(1.0 * m * (m + 1) * (n + 2)) * ~f[j][m - 1] * ~f[k][n + 2] * f[j][m + 1] * f[k][n + 1];
            }
            if (p == n && q == m - 1) {
                bi -= (n + 1) * sqrt(1.0 * m * (m - 1) * (n + 1)) * ~f[j][m - 2] * ~f[k][n + 1] * f[j][m] * f[k][n];
            }
            if (p == n && q == m + 1) {
                bi += (n + 1) * sqrt(1.0 * m * (m + 1) * (n + 1)) * ~f[j][m - 1] * ~f[k][n + 1] * f[j][m + 1] * f[k][n];
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
                bi += (n + 1) * sqrt(1.0 * m * q * (p + 1)) * ~f[j][m - 1] * ~f[b][q - 1] * ~f[k][n + 2] * f[j][m] * f[b][q] * f[k][n + 1];
            }
            if (p == n) {
                bi -= 2 * (n + 1) * sqrt(1.0 * m * q * (p + 1)) * ~f[j][m - 1] * ~f[b][q - 1] * ~f[k][n + 1] * f[j][m] * f[b][q] * f[k][n];
            }
            if (p == n - 1) {
                bi += (n + 1) * sqrt(1.0 * m * q * (p + 1)) * ~f[j][m - 1] * ~f[b][q - 1] * ~f[k][n] * f[j][m] * f[b][q] * f[k][n - 1];
            }
        }
        else if (m == q - 1) {
            if (p == n + 1) {
                bi += (n + 1) * sqrt(1.0 * m * (m + 1) * (n + 2)) * ~f[j][m - 1] * ~f[k][n + 2] * f[j][m + 1] * f[k][n + 1];
            }
            if (p == n) {
                bi -= 2 * (n + 1) * sqrt(1.0 * m * (m + 1) * (n + 1)) * ~f[j][m - 1] * ~f[k][n + 1] * f[j][m + 1] * f[k][n];
            }
            if (p == n - 1) {
                bi += (n + 1) * sqrt(1.0 * m * (m + 1) * n) * ~f[j][m - 1] * ~f[k][n] * f[j][m + 1] * f[k][n - 1];
            }
        }
    }
    return bi;
}

complex<double> bf8(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (i == k && m == p + 1 && j == a) {
        if (i != b) {
            bi += (n + 1) * m * sqrt(1.0 * q) * ~f[j][m] * ~f[b][q - 1] * ~f[k][n + 1] * f[j][m] * f[b][q] * f[k][n + 1];
            bi -= (n + 1) * m * sqrt(1.0 * q) * ~f[j][m - 1] * ~f[b][q - 1] * ~f[k][n + 1] * f[j][m - 1] * f[b][q] * f[k][n + 1];
            bi -= (n + 1) * m * sqrt(1.0 * q) * ~f[j][m] * ~f[b][q - 1] * ~f[k][n] * f[j][m] * f[b][q] * f[k][n];
            bi += (n + 1) * m * sqrt(1.0 * q) * ~f[j][m - 1] * ~f[b][q - 1] * ~f[k][n] * f[j][m - 1] * f[b][q] * f[k][n];
        }
        else {
            if (q == n + 2) {
                bi += (n + 1) * m * sqrt(1.0 * (n + 2)) * ~f[k][n + 1] * f[k][n + 2] * (~f[j][m] * f[j][m] - ~f[j][m - 1] * f[j][m - 1]);
            }
            if (q == n + 1) {
                bi -= (n + 1) * m * sqrt(1.0 * (n + 1)) * ~f[k][n] * f[k][n + 1] * (~f[j][m] * f[j][m] - ~f[j][m - 1] * f[j][m - 1]);
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

void compileFunction(Function& f, string name) {
    string sourcename = name + ".c";
    f.generateCode(sourcename);
    string libraryname = name + ".so";
    string cmd;
    cmd = "gcc -shared -o " + libraryname + " " + sourcename;
    ::system(cmd.c_str());
}

void loadFunction(Function& f, string name) {
    string libraryname = name + ".so";
    f = ExternalFunction(libraryname);
    f.init();
}

boost::mutex sum_mutex;

void SumFunction2(CustomFunction& f, void* user_data) {
    //    Parallelizer* par = static_cast<Parallelizer*>(user_data);
    //    int numInputs = f.getNumInputs();
    //    int numOutputs = f.getNumOutputs();
    {
        boost::mutex::scoped_lock lock(sum_mutex);
        f.getInput(0);
    }
    //    for (int i = 0; i < numInputs; i++) {
    //    for (int j = 0; j < L*(nmax+1); j++) {
    //        par->setInput(f.getInput(i), i);
    ////        par->setInput(f.getInput(i), j*numInputs + i);
    //    }
    //    }
    //    DMatrix parinput = repmat(f.input(i), L*(nmax+1));
}

struct sumdata {
    DMatrix input;
    DMatrix output;
    DMatrix outputi;
    vector<Function>* odes;
};

void SumFunction(CustomFunction& f, void* user_data) {
    {
        //        boost::mutex::scoped_lock lock(sum_mutex);
        sumdata* data = static_cast<sumdata*> (user_data);
        //    vector<Function>* fsp = static_cast<vector<Function>*>(user_data);
        //    vector<Function>& fs = *fsp;
        vector<Function>& fs = *(data->odes);
        DMatrix& input = data->input;
        DMatrix& output = data->output;
        DMatrix& outputi = data->outputi;
        int numFuncs = fs.size();
        //    int numInputs = fs[0].getNumInputs();
        //    int numOutputs = fs[0].getNumOutputs();
        //    int numFuncs = fs->size();
        //    int numInputs = (*fs)[0].getNumInputs();
        //    int numOutputs = (*fs)[0].getNumOutputs();
        //    f.getInput(0);
        //    f.input_struct();
        //    DMatrix matrix = 1;
        //    ostringstream qwe;
        //    qwe << fs[0].input(0);
        //    cout << f.getInput(0) << endl;
        //    DMatrix matrix;
        //    DMatrix matrixi;
        for (int i = 0; i < DAE_NUM_IN; i++) {
            //        DMatrix& input = f.input_struct().data.at(i);
            //        DMatrix& input = f.input(i);//f.getInput(i);            
            //        double asd = f.getInput(i).front();
            //        matrix = f.getInput(i);            
            //        f.getInput(i);
            //        DMatrix input(f.getInput(i));
            //        matrix = f.getInput(i);
            /*DMatrix*/ input = f.input(i);
            for (int j = 0; j < numFuncs; j++) {
                //            fs[j].input(i).set(f.input(i));
                //            fs[j].input_struct().data.at(i) = input;
                fs[j].setInput(input, i);
                //            fs[j].setInput(matrix, i);
            }
            //        matrix = 0;
        }
        for (int i = 0; i < numFuncs; i++) {
            fs[i].evaluate();
        }
        //    DMatrix output = 0;
        for (int i = 0; i < DAE_NUM_OUT; i++) {
            //        /*DMatrix*/ output = 0;
            //        matrix = 0;
            output = 0;
            for (int j = 0; j < numFuncs; j++) {
                /*DMatrix*/ outputi = fs[j].output(i);
                //            DMatrix outputi(fs[j].getOutput(i));
                //            matrixi = fs[j].getOutput(i);
                output += outputi;
                //            matrix += matrixi;
            }
            //        f.setOutput(matrix, i);
            f.setOutput(output, i);
        }
    }
}

//boost::mutex problem_mutex;

double DynamicsProblem::scale = 1;
double DynamicsProblem::U00;
vector<double> DynamicsProblem::J0;
vector<double> DynamicsProblem::xi;

vector<double> DynamicsProblem::x0;

vector<Function> DynamicsProblem::Efunc;
vector<Function> DynamicsProblem::Egradf;
//vector<Function> DynamicsProblem::sodes;

SX DynamicsProblem::sode;
SX DynamicsProblem::sx;
SX DynamicsProblem::sp;
SX DynamicsProblem::st;

vector<double> DynamicsProblem::sparams;
vector<double> DynamicsProblem::gsparams;

void DynamicsProblem::setup(double Wi_, double Wf_, double mu_, vector<double>& xi_, vector<double>& f0_) {

    xi = xi_;

    sparams.clear();
    sparams.push_back(Wi_);
    sparams.push_back(Wf_);
    sparams.push_back(mu_);
    for (double xii : xi_) sparams.push_back(xii);
    sparams.push_back(1);

    gsparams.clear();
    gsparams.push_back(Wi_);
    gsparams.push_back(Wf_);
    gsparams.push_back(mu_);
    for (double xii : xi_) gsparams.push_back(xii);
    gsparams.push_back(1);
    gsparams.push_back(0);

    U00 = scale * UW(Wi_);
    J0 = vector<double>(L);
    for (int i = 0; i < L; i++) {
        J0[i] = scale * JWij(Wi_ * xi[i], Wi_ * xi[mod(i + 1)]);
    }

    string compiledir = "L" + to_string(L) + "nmax" + to_string(nmax);
    path compilepath(compiledir);

    //    if (true) {
    if (!exists(compilepath)) {

        create_directory(compilepath);

        vector<SX> fin = SX::sym("f", 1, 1, 2 * L * dim);
        vector<SX> dU = SX::sym("dU", 1, 1, L);
        vector<SX> J = SX::sym("J", 1, 1, L);
        SX U0 = SX::sym("U0");
        SX t = SX::sym("t");
        SX mu = SX::sym("mu");

        SX tau = SX::sym("tau");
        SX Wi = SX::sym("Wi");
        SX Wf = SX::sym("Wf");
        SX Wt = if_else(t < tau, Wi + (Wf - Wi) * t / tau, Wf + (Wi - Wf) * (t - tau) / tau);

        vector<SX> xiv = SX::sym("xi", 1, 1, L);

        U0 = scale * UW(Wt);
        //    U00 = scale*UW(Wi_);
        //    J0 = vector<double>(L);
        for (int i = 0; i < L; i++) {
            //        J0[i] = scale*JWij(Wi_ * xi[i], Wi_ * xi[mod(i + 1)]);
            J[i] = scale * JWij(Wt * xiv[i], Wt * xiv[mod(i + 1)]);
            dU[i] = scale * UW(Wt * xiv[i]) - U0;
        }

        vector<SX> paramsv;
        paramsv.push_back(Wi);
        paramsv.push_back(Wf);
        paramsv.push_back(mu);
        for (SX xii : xiv) paramsv.push_back(xii);
        paramsv.push_back(tau);

        vector<SX> gsparamsv(paramsv.begin(), paramsv.end());
        gsparamsv.push_back(t);

        //    sparams.clear();
        //    sparams.push_back(Wi_);
        //    sparams.push_back(Wf_);
        //    sparams.push_back(mu_);
        //    for (double xii : xi_) sparams.push_back(xii);
        //    sparams.push_back(1);

        //    gsparams.clear();
        //    gsparams.push_back(Wi_);
        //    gsparams.push_back(Wf_);
        //    gsparams.push_back(mu_);
        //    for (double xii : xi_) gsparams.push_back(xii);
        //    gsparams.push_back(1);
        //    gsparams.push_back(0);

        SX x = SX::sym("x", fin.size());
        SX p = SX::sym("p", paramsv.size());
        SX gsp = SX::sym("gsp", gsparamsv.size());
        sx = x;
        sp = p;
        st = t;

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

        vector<SX> GSEs;
        vector<SX> Es;
        vector<SX> Sdts;
        //    vector<Function> odes;

        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                SX E = energy(i, n, fin, J, U0, dU, mu);
                Es.push_back(E);

                SX S = canonical(i, n, fin, J, U0, dU, mu);
                SXFunction Sf(vector<SX>{t}, vector<SX>{S});
                Sf.init();
                Function Sdtf = Sf.gradient(0, 0);
                Sdtf.init();
                SX Sdt = Sdtf.call(vector<SX>{t})[0];
                Sdts.push_back(Sdt);

                SX GSE = E;
                GSE = substitute(vector<SX>{GSE}, fin, xs)[0];
                GSE = substitute(vector<SX>{GSE}, gsparamsv, gsps)[0];
                GSEs.push_back(GSE);

                SXFunction Ef = SXFunction(nlpIn("x", x, "p", gsp), nlpOut("f", GSE));
                Ef.init();
                compileFunction(Ef, compiledir + "/E." + to_string(i) + "." + to_string(n));
                Function Egradf = Ef.gradient(NL_X, NL_F);
                Egradf.init();
                compileFunction(Egradf, compiledir + "/Egrad." + to_string(i) + "." + to_string(n));

                SX HSr = Sdt;
                SX HSi = -E;
                HSr = substitute(vector<SX>{HSr}, fin, xs)[0];
                HSr = substitute(vector<SX>{HSr}, paramsv, ps)[0];
                HSi = substitute(vector<SX>{HSi}, fin, xs)[0];
                HSi = substitute(vector<SX>{HSi}, paramsv, ps)[0];
                simplify(HSr);
                simplify(HSi);

                SXFunction HSf = SXFunction(vector<SX>{x, p}, vector<SX>{HSr, HSi});
                HSf.init();
                Function HSrdff = HSf.gradient(0, 0);
                Function HSidff = HSf.gradient(0, 1);
                HSrdff.init();
                HSidff.init();

                SX HSrdftmp = HSrdff.call(vector<SX>{x, p})[0];
                SX HSidftmp = HSidff.call(vector<SX>{x, p})[0];

                SX ode = SX::sym("ode", 2 * L * dim);
                for (int j = 0; j < L * dim; j++) {
                    ode[2 * j] = 0;
                    ode[2 * j + 1] = 0;
                    try {
                        ode[2 * j] += 0.5 * HSrdftmp[2 * j];
                    }
                    catch (CasadiException& e) {
                    }
                    try {
                        ode[2 * j] -= 0.5 * HSidftmp[2 * j + 1];
                    }
                    catch (CasadiException& e) {
                    }
                    try {
                        ode[2 * j + 1] += 0.5 * HSidftmp[2 * j];
                    }
                    catch (CasadiException& e) {
                    }
                    try {
                        ode[2 * j + 1] += 0.5 * HSrdftmp[2 * j + 1];
                    }
                    catch (CasadiException& e) {
                    }

                    //                ode[2 * j] = 0.5 * (HSrdftmp[2 * j] - HSidftmp[2 * j + 1]);
                    //                ode[2 * j + 1] = 0.5 * (HSidftmp[2 * j] + HSrdftmp[2 * j + 1]);
                    //        ode[2 * j] = 0.5 * - HSidftmp[2 * j + 1];
                    //        ode[2 * j + 1] = 0.5 * HSidftmp[2 * j];
                }
                Function ode_func = SXFunction(daeIn("x", x, "t", t, "p", p), daeOut("ode", ode));
                ode_func.init();
                //    sodes.push_back(ode_func);
                compileFunction(ode_func, compiledir + "/ode." + to_string(i) + "." + to_string(n));

            }
        }
        for (int i = 0; i < 1; i++) {
            for (int n = 0; n <= 0; n++) {
                SX E = energy(fin, J, U0, dU, mu);
//                SX E = energy(i, n, fin, J, U0, dU, mu);
                Es.push_back(E);

                SX S = canonical(fin, J, U0, dU, mu);
//                SX S = canonical(i, n, fin, J, U0, dU, mu);
                SXFunction Sf(vector<SX>{t}, vector<SX>{S});
                Sf.init();
                Function Sdtf = Sf.gradient(0, 0);
                Sdtf.init();
                SX Sdt = Sdtf.call(vector<SX>{t})[0];
                Sdts.push_back(Sdt);

//                SX GSE = energy(i, n, fin, J, U0, dU, mu);//E;
//                GSE = substitute(vector<SX>{GSE}, fin, xs)[0];
//                GSE = substitute(vector<SX>{GSE}, gsparamsv, gsps)[0];
//                GSEs.push_back(GSE);
//
//                SXFunction Ef = SXFunction(nlpIn("x", x, "p", gsp), nlpOut("f", GSE));
//                Ef.init();
//                compileFunction(Ef, compiledir + "/E." + to_string(i) + "." + to_string(n));
//                Function Egradf = Ef.gradient(NL_X, NL_F);
//                Egradf.init();
//                compileFunction(Egradf, compiledir + "/Egrad." + to_string(i) + "." + to_string(n));

                SX HSr = Sdt;
                SX HSi = -E;
                HSr = substitute(vector<SX>{HSr}, fin, xs)[0];
                HSr = substitute(vector<SX>{HSr}, paramsv, ps)[0];
                HSi = substitute(vector<SX>{HSi}, fin, xs)[0];
                HSi = substitute(vector<SX>{HSi}, paramsv, ps)[0];
                simplify(HSr);
                simplify(HSi);

                SXFunction HSf = SXFunction(vector<SX>{x, p}, vector<SX>{HSr, HSi});
                HSf.init();
                Function HSrdff = HSf.gradient(0, 0);
                Function HSidff = HSf.gradient(0, 1);
                HSrdff.init();
                HSidff.init();

                SX HSrdftmp = HSrdff.call(vector<SX>{x, p})[0];
                SX HSidftmp = HSidff.call(vector<SX>{x, p})[0];

                SX ode = SX::sym("ode", 2 * L * dim);
                for (int j = 0; j < L * dim; j++) {
                    ode[2 * j] = 0;
                    ode[2 * j + 1] = 0;
                    try {
                        ode[2 * j] += 0.5 * HSrdftmp[2 * j];
                    }
                    catch (CasadiException& e) {
                    }
                    try {
                        ode[2 * j] -= 0.5 * HSidftmp[2 * j + 1];
                    }
                    catch (CasadiException& e) {
                    }
                    try {
                        ode[2 * j + 1] += 0.5 * HSidftmp[2 * j];
                    }
                    catch (CasadiException& e) {
                    }
                    try {
                        ode[2 * j + 1] += 0.5 * HSrdftmp[2 * j + 1];
                    }
                    catch (CasadiException& e) {
                    }

                    //                ode[2 * j] = 0.5 * (HSrdftmp[2 * j] - HSidftmp[2 * j + 1]);
                    //                ode[2 * j + 1] = 0.5 * (HSidftmp[2 * j] + HSrdftmp[2 * j + 1]);
                    //        ode[2 * j] = 0.5 * - HSidftmp[2 * j + 1];
                    //        ode[2 * j + 1] = 0.5 * HSidftmp[2 * j];
                }
                sode = ode;
//                Function ode_func = SXFunction(daeIn("x", x, "t", t, "p", p), daeOut("ode", ode));
//                ode_func.init();
                //    sodes.push_back(ode_func);
//                compileFunction(ode_func, compiledir + "/ode." + to_string(i) + "." + to_string(n));

            }
        }
    }

//    SX qwe = SX::sym("qwe", 2 * L * dim);
//    SX sdf = SX::sym("sdf", 10);
//    //    SX asd = 2*qwe;
//    SX asd = SX::sym("asd");
//    asd = qwe[0];
//    SXFunction zxc = SXFunction(nlpIn("x", qwe, "p", sdf), nlpOut("f", asd));
//    zxc.init();
    Efunc = vector<Function>(L * (nmax + 1));
    Egradf = vector<Function>(L * (nmax + 1));
    //    sodes = vector<Function>(L*(nmax+1));
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            int j = i * (nmax + 1) + n;
//            Efunc[j] = zxc;
//            Egradf[j] = zxc;
                    loadFunction(Efunc[j], compiledir + "/E." + to_string(i) + "." + to_string(n));
                    loadFunction(Egradf[j], compiledir + "/Egrad." + to_string(i) + "." + to_string(n));
            //        loadFunction(sodes[j], compiledir + "/ode." + to_string(i) + "." + to_string(n));
        }
    }

    x0.clear();
    for (double f0i : f0_) x0.push_back(f0i);
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

    opt lopt(LD_LBFGS, 2 * L * dim);
    lopt.set_lower_bounds(-1);
    lopt.set_upper_bounds(1);
    lopt.set_min_objective(energyfunc, NULL);

    double E0;
//            try {
        result res = lopt.optimize(x0, E0);
//        }
//        catch (std::exception& e) {
//            stop_time = microsec_clock::local_time();
//            enum result res = lopt->last_optimize_result();
//            gsresult = to_string(res) + ": " + e.what();
//            cerr << e.what() << endl;
//        }

    //    vector<complex<double>> fi(dim);
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

    //    SX E = energy();
    //    SX S = canonical();
    //
    //    SXFunction Sf(vector<SX>{t}, vector<SX>{S});
    //    Sf.init();
    //    Function Sdtf = Sf.gradient(0, 0);
    //    Sdtf.init();
    //    SX Sdt = Sdtf.call(vector<SX>{t})[0];
    //
    //    SX GSE = E;
    //
    //    x = SX::sym("x", fin.size());
    //    p = SX::sym("p", params.size());
    //    gsp = SX::sym("gsp", gsparams.size());
    //
    //    vector<SX> xs;
    //    for (int i = 0; i < x.size(); i++) {
    //        xs.push_back(x.at(i));
    //    }
    //    vector<SX> ps;
    //    for (int i = 0; i < p.size(); i++) {
    //        ps.push_back(p.at(i));
    //    }
    //    vector<SX> gsps;
    //    for (int i = 0; i < gsp.size(); i++) {
    //        gsps.push_back(gsp.at(i));
    //    }
    //
    //    GSE = substitute(vector<SX>{GSE}, fin, xs)[0];
    //    GSE = substitute(vector<SX>{GSE}, gsparams, gsps)[0];
    //    simplify(GSE);
    //
    //    Efunc = SXFunction(nlpIn("x", x, "p", gsp), nlpOut("f", GSE));
    //    Efunc.init();
    //    Egradf = Efunc.gradient(NL_X, NL_F);
    //    Egradf.init();

    //    lopt = new opt(LD_LBFGS, 2 * L * dim);
    //    lopt->set_lower_bounds(-1);
    //    lopt->set_upper_bounds(1);
    //    lopt->set_min_objective(energyfunc, this);
    //
    //    setInitial(f0_);
    //    solve();
    //
    //        f0 = vector<vector<complex<double>>>(L, vector<complex<double>>(dim));
    //        for (int i = 0; i < L; i++) {
    //            for (int n = 0; n <= nmax; n++) {
    //                f0[i][n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
    //            }
    //            double nrm = sqrt(abs(dot(f0[i], f0[i])));
    //            for (int n = 0; n <= nmax; n++) {
    //                f0[i][n] /= nrm;
    //            }
    //        }
    //        
    //    SX HSr = Sdt;
    //    SX HSi = -E;
    //    HSr = substitute(vector<SX>{HSr}, fin, xs)[0];
    //    HSr = substitute(vector<SX>{HSr}, params, ps)[0];
    //    HSi = substitute(vector<SX>{HSi}, fin, xs)[0];
    //    HSi = substitute(vector<SX>{HSi}, params, ps)[0];
    //    simplify(HSr);
    //    simplify(HSi);
    //
    //    SXFunction HSf = SXFunction(vector<SX>{x, p}, vector<SX>{HSr, HSi});
    //    HSf.init();
    //    Function HSrdff = HSf.gradient(0, 0);
    //    Function HSidff = HSf.gradient(0, 1);
    //    HSrdff.init();
    //    HSidff.init();
    //
    //    SX HSrdftmp = HSrdff.call(vector<SX>{x, p})[0];
    //    SX HSidftmp = HSidff.call(vector<SX>{x, p})[0];
    //
    //    ode = SX::sym("ode", 2 * L * dim);
    //    for (int i = 0; i < L * dim; i++) {
    //                ode[2 * i] = 0.5 * (HSrdftmp[2 * i] - HSidftmp[2 * i + 1]);
    //                ode[2 * i + 1] = 0.5 * (HSidftmp[2 * i] + HSrdftmp[2 * i + 1]);
    ////        ode[2 * i] = 0.5 * - HSidftmp[2 * i + 1];
    ////        ode[2 * i + 1] = 0.5 * HSidftmp[2 * i];
    //    }
    //    ode_func = SXFunction(daeIn("x", x, "t", t, "p", p), daeOut("ode", ode));
    //
    //    Function g;
    //    integrator = new CvodesInterface(ode_func, g);
    //    integrator->setOption("max_num_steps", 1000000);
    //    integrator->setOption("stop_at_end", false);
    ////    integrator->setOption("linear_multistep_method", "adams");
    ////        integrator->setOption("linear_solver", "csparse");
    ////        integrator->setOption("linear_solver_type", "user_defined");
    ////        integrator = new RkIntegrator(ode_func, g);
    ////        integrator->setOption("number_of_finite_elements", 1000);
    //    integrator->setOption("t0", 0);
    ////    integrator->setOption("tf", 1);
    //    integrator->init();
}

const char* gettype(SharedObjectNode* node) {
    const type_info& t = typeid(DynamicsProblem*);
    return typeid(node).name();
}

//string printNode(SharedObjectNode* node) {
//    ostringstream out;
//    node->print(out);
//    return out.str();
//}

//DynamicsProblem::DynamicsProblem(double Wi, double Wf, double mu_, vector<double>& xi, vector<double>& f0_) : mu(mu_) {

DynamicsProblem::DynamicsProblem(int thread_, double tauf) : thread(thread_) {

    //    fin = SX::sym("f", 1, 1, 2 * L * dim);
    //    dU = SX::sym("dU", 1, 1, L);
    //    J = SX::sym("J", 1, 1, L);
    //    U0 = SX::sym("U0");
    //    t = SX::sym("t");
    //
    //    tau = SX::sym("tau");
    //    Wt = if_else(t < tau, Wi + (Wf - Wi) * t / tau, Wf + (Wi - Wf) * (t - tau) / tau);
    //
    ////    mu = 0.5;
    ////    SX Ut = 1;
    ////    double Ji = 0.2;
    ////    double Jf = 0.01;
    ////    SX Jt = if_else(t < tau, Ji + (Jf - Ji) * t / tau, Jf + (Ji - Jf)*(t - tau) / tau);
    ////    U0 = Ut;
    //    U0 = scale*UW(Wt);
    ////    U0 = scale*UW(Wi);
    ////    U00 = 1;
    //    U00 = scale*UW(Wi);
    //    J0 = vector<double>(L);
    //    for (int i = 0; i < L; i++) {
    ////        J0[i] = Ji;
    //        J0[i] = scale*JWij(Wi * xi[i], Wi * xi[mod(i + 1)]);
    ////        J[i] = Jt;
    //        J[i] = scale*JWij(Wt * xi[i], Wt * xi[mod(i + 1)]);
    //        //        J[i] = scale*JWij(Wt, Wt);
    //        //        Jp[i] = JWij(Wpt * xi[i], Wpt * xi[mod(i + 1)]);
    ////        dU[i] = 0;
    //        dU[i] = scale * UW(Wt * xi[i]) - U0;
    //    }
    //
    //    vector<SX> params;
    //    params.push_back(tau);
    //
    //    vector<SX> gsparams(params.begin(), params.end());
    //    gsparams.push_back(t);
    //    
    //    SX E = energy();
    //    SX S = canonical();
    //
    //    SXFunction Sf(vector<SX>{t}, vector<SX>{S});
    //    Sf.init();
    //    Function Sdtf = Sf.gradient(0, 0);
    //    Sdtf.init();
    //    SX Sdt = Sdtf.call(vector<SX>{t})[0];
    //
    //    SX GSE = E;
    //
    //    x = SX::sym("x", fin.size());
    //    p = SX::sym("p", params.size());
    //    gsp = SX::sym("gsp", gsparams.size());
    //
    //    vector<SX> xs;
    //    for (int i = 0; i < x.size(); i++) {
    //        xs.push_back(x.at(i));
    //    }
    //    vector<SX> ps;
    //    for (int i = 0; i < p.size(); i++) {
    //        ps.push_back(p.at(i));
    //    }
    //    vector<SX> gsps;
    //    for (int i = 0; i < gsp.size(); i++) {
    //        gsps.push_back(gsp.at(i));
    //    }
    //
    //    GSE = substitute(vector<SX>{GSE}, fin, xs)[0];
    //    GSE = substitute(vector<SX>{GSE}, gsparams, gsps)[0];
    //    simplify(GSE);
    //
    //    Efunc = SXFunction(nlpIn("x", x, "p", gsp), nlpOut("f", GSE));
    //    Efunc.init();
    //    Egradf = Efunc.gradient(NL_X, NL_F);
    //    Egradf.init();
    //
    //    lopt = new opt(LD_LBFGS, 2 * L * dim);
    //    lopt->set_lower_bounds(-1);
    //    lopt->set_upper_bounds(1);
    //    lopt->set_min_objective(energyfunc, this);
    //
    //    setInitial(f0_);
    //    solve();
    //
    //        f0 = vector<vector<complex<double>>>(L, vector<complex<double>>(dim));
    //        for (int i = 0; i < L; i++) {
    //            for (int n = 0; n <= nmax; n++) {
    //                f0[i][n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
    //            }
    //            double nrm = sqrt(abs(dot(f0[i], f0[i])));
    //            for (int n = 0; n <= nmax; n++) {
    //                f0[i][n] /= nrm;
    //            }
    //        }
    //        
    //    SX HSr = Sdt;
    //    SX HSi = -E;
    //    HSr = substitute(vector<SX>{HSr}, fin, xs)[0];
    //    HSr = substitute(vector<SX>{HSr}, params, ps)[0];
    //    HSi = substitute(vector<SX>{HSi}, fin, xs)[0];
    //    HSi = substitute(vector<SX>{HSi}, params, ps)[0];
    //    simplify(HSr);
    //    simplify(HSi);
    //
    //    SXFunction HSf = SXFunction(vector<SX>{x, p}, vector<SX>{HSr, HSi});
    //    HSf.init();
    //    Function HSrdff = HSf.gradient(0, 0);
    //    Function HSidff = HSf.gradient(0, 1);
    //    HSrdff.init();
    //    HSidff.init();
    //
    //    SX HSrdftmp = HSrdff.call(vector<SX>{x, p})[0];
    //    SX HSidftmp = HSidff.call(vector<SX>{x, p})[0];
    //
    //    ode = SX::sym("ode", 2 * L * dim);
    //    for (int i = 0; i < L * dim; i++) {
    //                ode[2 * i] = 0.5 * (HSrdftmp[2 * i] - HSidftmp[2 * i + 1]);
    //                ode[2 * i + 1] = 0.5 * (HSidftmp[2 * i] + HSrdftmp[2 * i + 1]);
    ////        ode[2 * i] = 0.5 * - HSidftmp[2 * i + 1];
    ////        ode[2 * i + 1] = 0.5 * HSidftmp[2 * i];
    //    }
    //    ode_func = SXFunction(daeIn("x", x, "t", t, "p", p), daeOut("ode", ode));

    //        return;

    f0 = vector<vector<complex<double>>>(L, vector<complex<double>>(dim));
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            f0[i][n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
        }
    }

    params = sparams;

    string compiledir = "L" + to_string(L) + "nmax" + to_string(nmax);
    odes = vector<Function>(L * (nmax + 1));
//    for (int i = 0; i < L; i++) {
//        for (int n = 0; n <= nmax; n++) {
    for (int i = 0; i < 1; i++) {
        for (int n = 0; n <= 0; n++) {
            int j = i * (nmax + 1) + n;
//            loadFunction(odes[j], compiledir + "/ode." + to_string(i) + "." + to_string(n));
            //        odes[j] = ExternalFunction(compiledir + "/ode." + to_string(i) + "." + to_string(n) + ".so");
            //        odes[j].init();
        }
    }
    odes.resize(1);

    vector<Sparsity> ins;
    ins.push_back(Sparsity::dense(2 * L * dim));
    ins.push_back(Sparsity::dense(0, 0));
    ins.push_back(Sparsity::dense(params.size()));
    ins.push_back(Sparsity::dense(1));
    //    ins.push_back(Sparsity::dense(1));

    vector<Sparsity> outs;
    outs.push_back(Sparsity::dense(2 * L * dim));
    outs.push_back(Sparsity::dense(0, 0));
    outs.push_back(Sparsity::dense(0, 0));
    //    outs.push_back(Sparsity::dense(1));

//        SX x = SX::sym("x", 2*L*dim);
//        SX p = SX::sym("p", params.size());
//        SX t = SX::sym("t");
//        SX out = SX::sym("ode", 2*L*dim);

    //    odes = sodes;
    sumdata* data = new sumdata;
    data->odes = &odes;
    /*Function*/ ode_func = CustomFunction(SumFunction, ins, outs); //, odes[0].getInputScheme(), odes[0].getOutputScheme());
    ode_func.setOption("user_data", &odes);
    ode_func.setOption("user_data", data);
    //    ode_func.setInputScheme(IOScheme(SCHEME_DAEInput));
    //    ode_func.setOutputScheme(IOScheme(SCHEME_DAEOutput));
    ode_func.setNumInputs(DAE_NUM_IN); //odes[0].getNumInputs());
    ode_func.setNumOutputs(DAE_NUM_OUT); //odes[0].getNumOutputs());
    //    ode_func.setInputScheme(daeIn("x", x, "t", t, "p", p).scheme);
    //    ode_func.setOutputScheme(daeOut("ode", out).scheme);
    ode_func.init();

    ode = sode;
               ode_func = SXFunction(daeIn("x", sx, "t", st, "p", sp), daeOut("ode", ode));
    //    ode_func.getInput(0);
    //    SX qwe = SX::sym("qwe", 2*L*dim);
    //    SX sdf = SX::sym("sdf", 9);
    ////    SX asd = 2*qwe;
    //    SX asd = SX::sym("asd", 2*L*dim);
    //    for (int i = 0; i < 2*L*dim; i++) {
    //        asd[i] = 2*qwe[i];
    //    }
    //    SXFunction zxc = SXFunction(daeIn("x", qwe, "p", sdf), daeOut("ode", asd));
    //    odes = vector<Function>{zxc};
    //    qwef = ode_func;
    //    ode_func = zxc;
    //    zxc.getInput(0);
    //    ode_func = SXFunction(daeIn("x", qwe, "p", sdf), daeOut("ode", asd));
    //    ode_func.init();
    //    ode_func.getInput(0);

    //    ExternalFunction* ff = new ExternalFunction(compiledir+"/ode.0.0.so");
    //    delete ff;
    //        loadFunction(ff, compiledir + "/ode.0.0");
    //        ff.getInput(0);

    //    par = Parallelizer(odes);
    //    par.init();
    //    
    //    ode_func = CustomFunction(SumFunction, ins, outs);
    //    ode_func.setOption("user_data", &par);
    //    ode_func.setNumInputs(odes[0].getNumInputs());
    //    ode_func.setNumOutputs(odes[0].getNumOutputs());
    //    ode_func.init();

    //    DMatrix mat = ode_func.getInput(0);
    //    cout << mat << endl;

    //    ode_func = Function();
    //    ode_func.setInputScheme(IOScheme(SCHEME_DAEInput));
    //    ode_func.setOutputScheme(IOScheme(SCHEME_DAEOutput));

    Function g;
//        integrator = new CvodesInterface(ode_func, g);
//        integrator->setOption("max_num_steps", 1000000);
//        integrator->setOption("stop_at_end", false);
    ////    integrator->setOption("linear_multistep_method", "adams");
    ////        integrator->setOption("linear_solver", "csparse");
    ////        integrator->setOption("linear_solver_type", "user_defined");
    //    if(thread == 0) {
    integrator = new RkIntegrator(ode_func, g);
    integrator->setOption("number_of_finite_elements", 1000);//20000); //100000000);
//    integrator = new CollocationIntegrator(ode_func, g);
    integrator->setOption("t0", 0);
    integrator->setOption("tf", 2*tauf);
    //    integrator->setOption("tf", 1);
    integrator->init();
    //    }
}

void DynamicsProblem::setTau(double tau_) {
    //    params.clear();
    //    params.push_back(tau_ / scale);

    params.back() = tau_ / scale;

    //    gsparams = vector<double>(params.begin(), params.end());
    //    gsparams.push_back(0);

    tf = 2 * tau_ / scale;
}

//void DynamicsProblem::setInitial(vector<double>& f0) {
//    x0.clear();
//    for (double f0i : f0) x0.push_back(f0i);
//    vector<complex<double>> fi(dim);
//    for (int i = 0; i < L; i++) {
//        for (int n = 0; n <= nmax; n++) {
//            fi[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
//        }
//        double nrm = sqrt(abs(dot(fi, fi)));
//        for (int n = 0; n <= nmax; n++) {
//            x0[2 * (i * dim + n)] /= nrm;
//            x0[2 * (i * dim + n) + 1] /= nrm;
//        }
//    }
//}

//void DynamicsProblem::solve() {
//    double E0;
//    //    try {
//    start_time = microsec_clock::local_time();
//    enum result res = lopt->optimize(x0, E0);
//    stop_time = microsec_clock::local_time();
//    gsresult = to_string(res);
//    //    }
//    //    catch (std::exception& e) {
//    //        stop_time = microsec_clock::local_time();
//    //        enum result res = lopt->last_optimize_result();
//    //        gsresult = to_string(res) + ": " + e.what();
//    //        cerr << e.what() << endl;
//    //    }
//
//    vector<complex<double>> fi(dim);
//    for (int i = 0; i < L; i++) {
//        for (int n = 0; n <= nmax; n++) {
//            fi[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
//        }
//        double nrm = sqrt(abs(dot(fi, fi)));
//        for (int n = 0; n <= nmax; n++) {
//            x0[2 * (i * dim + n)] /= nrm;
//            x0[2 * (i * dim + n) + 1] /= nrm;
//        }
//    }
////    cout << setprecision(10) << E0 << endl;
////        cout << setprecision(10) << x0 << endl;
////        exit(0);
//
//    time_period period(start_time, stop_time);
//    gsruntime = to_simple_string(period.length());
//}

void DynamicsProblem::evolve() {
    //    if (thread > 0) return;
    //    ode_func.getInput(0);
    start_time = microsec_clock::local_time();
    ptime eval_start_time = microsec_clock::local_time();

    integrator->setInput(x0, INTEGRATOR_X0);
    integrator->setInput(params, INTEGRATOR_P);

    integrator->reset();
    {
        //        boost::mutex::scoped_lock lock(sum_mutex);
        //        ode_func.input(DAE_T).set(double(0));
        //      ode_func.input(DAE_X).set(x0);
        //      F.input(DAE_Z).set(Z_);
        //      ode_func.input(DAE_P).set(params);
        //      ode_func.evaluate();
        //      ode_func.output(DAE_ODE).get(x0);
        //      F.output(DAE_ALG).get(Z_);
        integrator->integrate(tf);
    }

    //    DMatrix xfm3(integrator->output(INTEGRATOR_XF));
    DMatrix xfm = integrator->output(INTEGRATOR_XF);
    //    vector<double> qwe(2*L*dim);
    //    integrator->output(INTEGRATOR_XF).get(qwe);
    {
        //        boost::mutex::scoped_lock lock(sum_mutex);
        //   xfm = integrator->output(INTEGRATOR_XF);
    }
    //    return;
    vector<double> xf;
    for (int i = 0; i < 2 * L * dim; i++) {
        xf.push_back(xfm[i].getValue());
    }

    //        vector<vector<complex<double>>> f0(L, vector<complex<double>>(dim));
    //        for (int i = 0; i < L; i++) {
    //            for (int n = 0; n <= nmax; n++) {
    //                f0[i][n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
    //            }
    //            double nrm = sqrt(abs(dot(f0[i], f0[i])));
    //            for (int n = 0; n <= nmax; n++) {
    //                f0[i][n] /= nrm;
    //            }
    //        }

    //        vector<vector<complex<double>>> ff(L, vector<complex<double>>(dim));
    ff = vector<vector<complex<double>>>(L, vector<complex<double>>(dim));
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            ff[i][n] = complex<double>(xf[2 * (i * dim + n)], xf[2 * (i * dim + n) + 1]);
        }
        double nrm = sqrt(abs(dot(ff[i], ff[i])));
        for (int n = 0; n <= nmax; n++) {
            ff[i][n] /= nrm;
        }
    }

    b0 = vector<complex<double>>(L);
    bf = vector<complex<double>>(L);
    for (int i = 0; i < L; i++) {
        b0[i] = b(f0, i, J0, U00);
        bf[i] = b(ff, i, J0, U00);
    }
    //        vector<complex<double>> bc0(L), bcf(L);
    //        for (int i = 0; i < L; i++) {
    //            bc0[i] = b(f0, i, J0, U00);
    //            bcf[i] = b(ff, i, J0, U00);
    //        }
    //        b0 = vector<double>(L);
    //        bf = vector<double>(L);
    //        for (int i = 0; i < L; i++) {
    //            b0[i] = abs(bc0[i]);
    //            bf[i] = abs(bcf[i]);
    //        }

    vector<double> grad;
    E0 = E(x0, grad);
    Ef = E(xf, grad);
    Q = Ef - E0;


    vector<double> pi(L);
    pd = 0;
    for (int i = 0; i < L; i++) {
        pi[i] = 1 - norm(dot(ff[i], f0[i]));
        pd += pi[i];
    }
    pd /= L;

    ptime eval_stop_time = microsec_clock::local_time();
    time_period eval_period(eval_start_time, eval_stop_time);

    stop_time = microsec_clock::local_time();
    time_period period(start_time, stop_time);
    runtime = to_simple_string(period.length());

}

double DynamicsProblem::E(const vector<double>& f, vector<double>& grad) {
    //    vector<double> params(2);
    //    params[0] = 1;
    //    params[1] = 0;
    //    double E = 0;
    double E = 0;
    for (int i = 0; i < Efunc.size(); i++) {
        double Ei = 0;
        Efunc[i].setInput(f.data(), NL_X);
        Efunc[i].setInput(gsparams.data(), NL_P);
        Efunc[i].evaluate();
        Efunc[i].getOutput(Ei, NL_F);
        E += Ei;
    }
    if (!grad.empty()) {
        fill(grad.begin(), grad.end(), 0);
        for (int i = 0; i < Egradf.size(); i++) {
            vector<double> gradi(2 * L * dim);
            Egradf[i].setInput(f.data(), NL_X);
            Egradf[i].setInput(gsparams.data(), NL_P);
            Egradf[i].evaluate();
            Egradf[i].output().getArray(gradi.data(), gradi.size(), DENSE);
            for (int j = 0; j < 2 * L * dim; j++) {
                grad[j] += gradi[j];
            }
        }
    }
    return E;
    //    Efunc.setInput(f.data(), NL_X);
    //    Efunc.setInput(params.data(), NL_P);
    //    Efunc.evaluate();
    //    Efunc.getOutput(E, NL_F);
    //    if (!grad.empty()) {
    //        Egradf.setInput(f.data(), NL_X);
    //        Egradf.setInput(params.data(), NL_P);
    //        Egradf.evaluate();
    //        Egradf.output().getArray(grad.data(), grad.size(), DENSE);
    //    }
    //    return E;
}

//double DynamicsProblem::E(const vector<double>& f, double t) {
//    gsparams.back() = t;
//    vector<double> g;
//    return E(f, g);
//}

SX DynamicsProblem::energync(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu) {

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

SX DynamicsProblem::energya(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu) {
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

                for (int a = -nmax + 1; a <= nmax; a++) {
                    if (a != 0) {
                        if (n + a + 1 >= 0 && n + a + 1 <= nmax) {
                            Ej1 += 0.5 * (n + 1) * (n + a + 1) * J[j1] * J[j1]*~f[i][n] * f[i][n]*~f[j1][n + a + 1] * f[j1][n + a + 1] / eps(U0, i, j1, a);
                            Ej2 += 0.5 * (n + 1) * (n + a + 1) * J[i] * J[i]*~f[i][n] * f[i][n]*~f[j2][n + a + 1] * f[j2][n + a + 1] / eps(U0, i, j2, a);
                        }
                        if (n - a + 1 >= 0 && n - a + 1 <= nmax) {
                            Ej1 += -0.5 * (n + 1) * (n - a + 1) * J[j1] * J[j1]*~f[i][n] * f[i][n]*~f[j1][n - a + 1] * f[j1][n - a + 1] / eps(U0, i, j1, a);
                            Ej2 += -0.5 * (n + 1) * (n - a + 1) * J[i] * J[i]*~f[i][n] * f[i][n]*~f[j2][n - a + 1] * f[j2][n - a + 1] / eps(U0, i, j2, a);
                        }
                    }
                }

                if (n < nmax - 1) {
                    Ej1 += 0.5 * (n + 1)*(n + 2) * J[j1] * J[j1]*~f[i][n + 2] * f[i][n]*~f[j1][n] * f[j1][n + 2]*(1 / eps(U0, i, j1, 1) - 1 / eps(U0, i, j1, -1));
                    Ej2 += 0.5 * (n + 1)*(n + 2) * J[i] * J[i]*~f[i][n + 2] * f[i][n]*~f[j2][n] * f[j2][n + 2]*(1 / eps(U0, i, j2, 1) - 1 / eps(U0, i, j2, -1));

                    for (int a = -nmax + 1; a <= nmax; a++) {
                        if (a != 0) {
                            if (n + a >= 0 && n + a <= nmax && n + a + 1 >= 0 && n + a + 1 <= nmax && n - a + 1 >= 0 && n - a + 1 <= nmax && n - a + 2 >= 0 && n - a + 2 <= nmax) {
                                Ej1j2 += 0.5 * J[j1] * J[i] * ~f[i][n + 2] * f[i][n] / eps(U0, i, j1, a) * ga(n, -a) * ga(n + 1, a) * ~f[j2][n + a] * f[j2][n + a + 1] * ~f[j1][n - a + 1] * f[j1][n - a + 2];
                                Ej1j2 += 0.5 * J[j1] * J[i] * f[i][n + 2] * ~f[i][n] / eps(U0, i, j1, a) * ga(n, -a) * ga(n + 1, a) * f[j2][n + a] * ~f[j2][n + a + 1] * f[j1][n - a + 1] * ~f[j1][n - a + 2];
                                Ej1j2 += 0.5 * J[j1] * J[i] * ~f[i][n + 2] * f[i][n] / eps(U0, i, j2, a) * ga(n, -a) * ga(n + 1, a) * ~f[j1][n + a] * f[j1][n + a + 1] * ~f[j2][n - a + 1] * f[j2][n - a + 2];
                                Ej1j2 += 0.5 * J[j1] * J[i] * f[i][n + 2] * ~f[i][n] / eps(U0, i, j2, a) * ga(n, -a) * ga(n + 1, a) * f[j1][n + a] * ~f[j1][n + a + 1] * f[j2][n - a + 1] * ~f[j2][n - a + 2];
                            }
                            if (n - a >= 0 && n - a <= nmax && n - a + 1 >= 0 && n - a + 1 <= nmax && n + a + 1 >= 0 && n + a + 1 <= nmax && n + a + 2 >= 0 && n + a + 2 <= nmax) {
                                Ej1j2 -= 0.5 * J[j1] * J[i] * ~f[i][n + 2] * f[i][n] / eps(U0, i, j1, -a) * ga(n, a) * ga(n + 1, -a) * ~f[j2][n - a] * f[j2][n - a + 1] * ~f[j1][n + a + 1] * f[j1][n + a + 2];
                                Ej1j2 -= 0.5 * J[j1] * J[i] * f[i][n + 2] * ~f[i][n] / eps(U0, i, j1, -a) * ga(n, a) * ga(n + 1, -a) * f[j2][n - a] * ~f[j2][n - a + 1] * f[j1][n + a + 1] * ~f[j1][n + a + 2];
                                Ej1j2 -= 0.5 * J[j1] * J[i] * ~f[i][n + 2] * f[i][n] / eps(U0, i, j2, -a) * ga(n, a) * ga(n + 1, -a) * ~f[j1][n - a] * f[j1][n - a + 1] * ~f[j2][n + a + 1] * f[j2][n + a + 2];
                                Ej1j2 -= 0.5 * J[j1] * J[i] * f[i][n + 2] * ~f[i][n] / eps(U0, i, j2, -a) * ga(n, a) * ga(n + 1, -a) * f[j1][n - a] * ~f[j1][n - a + 1] * f[j2][n + a + 1] * ~f[j2][n + a + 2];
                            }

                            if (n + a >= 0 && n + a <= nmax && n + a + 1 >= 0 && n + a + 1 <= nmax) {
                                Ej1k1 += 0.5 * J[j1] * J[k1] * ~f[j1][n] * f[j1][n] / eps(U0, i, j1, a) * ga(n, -a) * ga(n + a, a) * ~f[k1][n + a] * f[k1][n + a + 1] * ~f[i][n + a + 1] * f[i][n + a];
                                Ej1k1 += 0.5 * J[j1] * J[k1] * f[j1][n] * ~f[j1][n] / eps(U0, i, j1, a) * ga(n, -a) * ga(n + a, a) * f[k1][n + a] * ~f[k1][n + a + 1] * f[i][n + a + 1] * ~f[i][n + a];
                                Ej2k2 += 0.5 * J[j2] * J[i] * ~f[j2][n] * f[j2][n] / eps(U0, i, j2, a) * ga(n, -a) * ga(n + a, a) * ~f[k2][n + a] * f[k2][n + a + 1] * ~f[i][n + a + 1] * f[i][n + a];
                                Ej2k2 += 0.5 * J[j2] * J[i] * f[j2][n] * ~f[j2][n] / eps(U0, i, j2, a) * ga(n, -a) * ga(n + a, a) * f[k2][n + a] * ~f[k2][n + a + 1] * f[i][n + a + 1] * ~f[i][n + a];
                            }
                            if (n - a >= 0 && n - a <= nmax && n - a + 1 >= 0 && n - a + 1 <= nmax) {
                                Ej1k1 -= 0.5 * J[j1] * J[k1] * ~f[j1][n] * f[j1][n] / eps(U0, i, j1, -a) * ga(n, a) * ga(n - a, -a) * ~f[k1][n - a] * f[k1][n - a + 1] * ~f[i][n - a + 1] * f[i][n - a];
                                Ej1k1 -= 0.5 * J[j1] * J[k1] * f[j1][n] * ~f[j1][n] / eps(U0, i, j1, -a) * ga(n, a) * ga(n - a, -a) * f[k1][n - a] * ~f[k1][n - a + 1] * f[i][n - a + 1] * ~f[i][n - a];
                                Ej2k2 -= 0.5 * J[j2] * J[i] * ~f[j2][n] * f[j2][n] / eps(U0, i, j2, -a) * ga(n, a) * ga(n - a, -a) * ~f[k2][n - a] * f[k2][n - a + 1] * ~f[i][n - a + 1] * f[i][n - a];
                                Ej2k2 -= 0.5 * J[j2] * J[i] * f[j2][n] * ~f[j2][n] / eps(U0, i, j2, -a) * ga(n, a) * ga(n - a, -a) * f[k2][n - a] * ~f[k2][n - a + 1] * f[i][n - a + 1] * ~f[i][n - a];
                            }
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

SX DynamicsProblem::energy(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu) {

    SX E = 0;
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
        E += energy(i, n, fin, J, U0, dU, mu);
        }
    }
    return E;
}

SX DynamicsProblem::energy(int i, int n, vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu) {

    complex<SX> expth = complex<SX>(1, 0);
    complex<SX> expmth = ~expth;
    complex<SX> exp2th = expth*expth;
    complex<SX> expm2th = ~exp2th;

    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        f[j] = reinterpret_cast<complex<SX>*> (&fin[2 * j * dim]);
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

    //    for (int i = 0; i < L; i++) {

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

    //        for (int n = 0; n <= nmax; n++) {
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
    //        }

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
    //    }

    return E.real();
}

SX DynamicsProblem::canonicala(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu) {

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
                    if (n - a >= 0 && n - a <= nmax && n - a + 1 >= 0 && n - a + 1 <= nmax) {
                        Sj1 += -J[j1] / eps(U0, i, j1, a) * ga(n, a) * ~f[i][n + 1] * f[i][n] * ~f[j1][n - a] * f[j1][n - a + 1];
                        Sj2 += -J[i] / eps(U0, i, j2, a) * ga(n, a) * ~f[i][n + 1] * f[i][n] * ~f[j2][n - a] * f[j2][n - a + 1];
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

SX DynamicsProblem::canonical(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu) {

    SX S = 0;
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
        S += canonical(i, n, fin, J, U0, dU, mu);
        }
    }
    return S;
}

SX DynamicsProblem::canonical(int i, int n, vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu) {

    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        f[j] = reinterpret_cast<complex<SX>*> (&fin[2 * j * dim]);
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    complex<SX> S = complex<SX>(0, 0);

    complex<SX> Sj1, Sj2;

    //    for (int i = 0; i < L; i++) {

    int j1 = mod(i - 1);
    int j2 = mod(i + 1);

    Sj1 = complex<SX>(0, 0);
    Sj2 = complex<SX>(0, 0);

    //        for (int n = 0; n <= nmax; n++) {
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
    //        }

    Sj1 /= norm2[i] * norm2[j1];
    Sj2 /= norm2[i] * norm2[j2];

    S += Sj1;
    S += Sj2;
    //    }

    return S.imag();
}

SX DynamicsProblem::energy0(vector<SX>& fin, vector<SX>& J, SX& U0, vector<SX>& dU, SX& mu) {

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

