#include <boost/thread.hpp>

using namespace boost;

#include "casadi.hpp"

//double energyfunc(const vector<double>& x, vector<double>& grad, void *data) {
//    DynamicsProblem* prob = static_cast<DynamicsProblem*> (data);
//    return prob->E(x, grad);
//}

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
        bi += sqrt(1.0*n) * f[i][n - 1] * f[i][n];
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
                bi += -J[i] * g2(n, m) / eps(U, n, m) * sqrt(1.0*n + 1) * conj(f[j2][m - 1]) * f[j2][m] * (conj(f[i][n + 1]) * f[i][n + 1] - conj(f[i][n]) * f[i][n]);
                bi += -J[j1] * g2(n, m) / eps(U, n, m) * sqrt(1.0*n + 1) * conj(f[j1][m - 1]) * f[j1][m] * (conj(f[i][n + 1]) * f[i][n + 1] - conj(f[i][n]) * f[i][n]);

                if (m < nmax) {
                    bi += -J[i] * g2(n, m) / eps(U, n, m) * sqrt(1.0*m + 1) * conj(f[j2][n + 1]) * f[j2][n] * conj(f[i][m - 1]) * f[i][m + 1];
                    bi += -J[j1] * g2(n, m) / eps(U, n, m) * sqrt(1.0*m + 1) * conj(f[j1][n + 1]) * f[j1][n] * conj(f[i][m - 1]) * f[i][m + 1];
                }
                if (m > 1) {
                    bi += J[i] * g2(n, m) / eps(U, n, m) * sqrt(1.0*m - 1) * conj(f[j2][n + 1]) * f[j2][n] * conj(f[i][m - 2]) * f[i][m];
                    bi += J[j1] * g2(n, m) / eps(U, n, m) * sqrt(1.0*m - 1) * conj(f[j1][n + 1]) * f[j1][n] * conj(f[i][m - 2]) * f[i][m];
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
                bi -= (n + 1) * sqrt(1.0*m * (m - 1) * (p + 1)) * conj(f[i][n]) * conj(f[a][p + 1]) * conj(f[k][m - 2]) * f[i][n] * f[a][p] * f[k][m];
                bi += (n + 1) * sqrt(1.0*m * (m - 1) * (p + 1)) * conj(f[i][n + 1]) * conj(f[a][p + 1]) * conj(f[k][m - 2]) * f[i][n + 1] * f[a][p] * f[k][m];
            }
            if (m < nmax) {
                bi += (n + 1) * sqrt(1.0*m * (m + 1) * (p + 1)) * conj(f[i][n]) * conj(f[a][p + 1]) * conj(f[k][m - 1]) * f[i][n] * f[a][p] * f[k][m + 1];
                bi -= (n + 1) * sqrt(1.0*m * (m + 1) * (p + 1)) * conj(f[i][n + 1]) * conj(f[a][p + 1]) * conj(f[k][m - 1]) * f[i][n + 1] * f[a][p] * f[k][m + 1];
            }
        }
        else {
            if (p == m - 1) {
                if (m < nmax) {
                    bi += m * (n + 1) * sqrt(1.0*m + 1) * conj(f[i][n]) * conj(f[k][m]) * f[i][n] * f[k][m + 1];
                    bi -= m * (n + 1) * sqrt(1.0*m + 1) * conj(f[i][n + 1]) * conj(f[k][m]) * f[i][n + 1] * f[k][m + 1];
                }
            }
            else if (p == m - 2) {
                bi -= (m - 1) * (n + 1) * sqrt(1.0*m) * conj(f[i][n]) * conj(f[k][m - 1]) * f[i][n] * f[k][m];
                bi += (m - 1) * (n + 1) * sqrt(1.0*m) * conj(f[i][n + 1]) * conj(f[k][m - 1]) * f[i][n + 1] * f[k][m];
            }
        }
    }
    return bi;
}

complex<double> bf2(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (b == k && j == k) {
        if (a != i) {
            if (q == m-1 && m >= 2 && m < nmax) {
                bi += sqrt(1.0 * (p+1) * (n+1) * m * (m+1) * q) * conj(f[i][n+1]) * conj(f[a][p+1]) * conj(f[k][m-2]) * f[i][n] * f[a][p] * f[k][m+1];
            }
            if (q == m+2 && m < nmax-1) {
                bi -= sqrt(1.0 * (p+1) * (n+1) * m * (m+1) * q) * conj(f[i][n+1]) * conj(f[a][p+1]) * conj(f[k][m-1]) * f[i][n] * f[a][p] * f[k][m+2];
            }
            if (q == m-2 && m >= 3) {
                bi -= sqrt(1.0 * (p+1) * (n+1) * (m-1) * m * q) * conj(f[i][n+1]) * conj(f[a][p+1]) * conj(f[k][m-3]) * f[i][n] * f[a][p] * f[k][m];
            }
            if (q == m+1 && m >= 2 && m < nmax) {
                bi += sqrt(1.0 * (p+1) * (n+1) * (m-1) * m * q) * conj(f[i][n+1]) * conj(f[a][p+1]) * conj(f[k][m-2]) * f[i][n] * f[a][p] * f[k][m+1];
            }
        }
        else if (p == n+1) {
            if (q == m-1 && n < nmax - 1 && m >= 2 && m < nmax) {
                bi += sqrt(1.0 * (n+2) * (n+1) * m * (m+1) * (m-1)) * conj(f[i][n+2]) * conj(f[k][m-2]) * f[i][n] * f[k][m+1];
            }
            if (q == m+2 && n < nmax - 1 && m < nmax - 1) {
                bi -= sqrt(1.0 * (n+2) * (n+1) * m * (m+1) * (m+2)) * conj(f[i][n+2]) * conj(f[k][m-1]) * f[i][n] * f[k][m+2];
            }
            if (q == m-2 && n < nmax - 1 && m >= 3) {
                bi -= sqrt(1.0 * (n+2) * (n+1) * (m-1) * m * (m-2)) * conj(f[i][n+2]) * conj(f[k][m-3]) * f[i][n] * f[k][m];
            }
            if (q == m+1 && n < nmax - 1 && m >= 2 && m < nmax) {
                bi += sqrt(1.0 * (n+2) * (n+1) * (m-1) * m * (m+1)) * conj(f[i][n+2]) * conj(f[k][m-2]) * f[i][n] * f[k][m+1];
            }
        }
    }
    return bi;
}

complex<double> bf3(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (i == a && j == k) {
        if (b != k) {
            if (p == n+1 && n < nmax - 1 && m < nmax) {
                bi += sqrt(1.0 * q * (n+1) * (n+2) * m * (m+1)) * conj(f[i][n+2]) * conj(f[b][q-1]) * conj(f[k][m-1]) * f[i][n] * f[b][q] * f[k][m+1];
            }
            if (p == n+1 && n < nmax - 1 && m >= 2) {
                bi -= sqrt(1.0 * q * (n+1) * (n+2) * (m-1) * m) * conj(f[i][n+2]) * conj(f[b][q-1]) * conj(f[k][m-2]) * f[i][n] * f[b][q] * f[k][m];
            }
            if (p == n-1 && n >= 1 && m < nmax) {
                bi -= sqrt(1.0 * q * n * (n+1) * m * (m+1)) * conj(f[i][n+1]) * conj(f[b][q-1]) * conj(f[k][m-1]) * f[i][n-1] * f[b][q] * f[k][m+1];
            }
            if (p == n-1 && m >= 2 && n >= 1) {
                bi += sqrt(1.0 * q * n * (n+1) * (m-1) * m) * conj(f[i][n+1]) * conj(f[b][q-1]) * conj(f[k][m-2]) * f[i][n-1] * f[b][q] * f[k][m];
            }
        }
        else {
            if (q == m+2 && p == n+1 && n < nmax - 1 && m < nmax - 1) {
                bi += sqrt(1.0 * (n+1) * (n+2) * m * (m+1) * (m+2)) * conj(f[i][n+2]) * conj(f[k][m-1]) * f[i][n] * f[k][m+2];
            }
            if (q == m+1 && p == n+1 && n < nmax - 1 && m >= 2 && m < nmax) {
                bi -= sqrt(1.0 * (n+1) * (n+2) * (m-1) * m * (m+1)) * conj(f[i][n+2]) * conj(f[k][m-2]) * f[i][n] * f[k][m+1];
            }
            if (q == m+2 && p == n-1 && n >= 1 && m < nmax - 1) {
                bi -= sqrt(1.0 * n * (n+1) * m * (m+1) * (m+2)) * conj(f[i][n+1]) * conj(f[k][m-1]) * f[i][n-1] * f[k][m+2];
            }
            if (q == m+1 && p == n-1 && m >= 2 && m < nmax) {
                bi += sqrt(1.0 * n * (n+1) * (m-1) * m * (m+1)) * conj(f[i][n+1]) * conj(f[k][m-2]) * f[i][n-1] * f[k][m+1];
            }
        }
    }
    return bi;
}

complex<double> bf4(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
    if (a == k && j == k) {
        if (b != i) {
            if (p == m-1 && m < nmax) {
                bi += m * sqrt(1.0 * (n+1) * q * (m+1)) * conj(f[i][n+1]) * conj(f[b][q-1]) * conj(f[k][m]) * f[i][n] * f[b][q] * f[k][m+1];
            }
            if (p == m-2) {
                bi -= (m-1) * sqrt(1.0 * (n+1) * q * m) * conj(f[i][n+1]) * conj(f[b][q-1]) * conj(f[k][m-1]) * f[i][n] * f[b][q] * f[k][m];
            }
            if (p == m) {
                bi -= (m+1) * sqrt(1.0 * (n+1) * q * m) * conj(f[i][n+1]) * conj(f[b][q-1]) * conj(f[k][m-1]) * f[i][n] * f[b][q] * f[k][m];
            }
            if (p == m-1 && m >= 2) {
                bi += m * sqrt(1.0 * (n+1) * q * (m-1)) * conj(f[i][n+1]) * conj(f[b][q-1]) * conj(f[k][m-2]) * f[i][n] * f[b][q] * f[k][m-1];
            }
        }
        else if (n == q-1) {
            if (p == m-1 && m < nmax) {
                bi += (n+1) * m * sqrt(1.0 * (m+1)) * conj(f[i][n+1]) * conj(f[k][m]) * f[i][n+1] * f[k][m+1];
            }
            if (p == m-2) {
                bi -= (n+1) * (m-1) * sqrt(1.0 * m) * conj(f[i][n+1]) * conj(f[k][m-1]) * f[i][n+1] * f[k][m];
            }
            if (p == m) {
                bi -= (n+1) * (m+1) * sqrt(1.0 * m) * conj(f[i][n+1]) * conj(f[k][m-1]) * f[i][n+1] * f[k][m];
            }
            if (p == m-1 && m >= 2) {
                bi += (n+1) * m * sqrt(1.0 * (m-1)) * conj(f[i][n+1]) * conj(f[k][m-2]) * f[i][n+1] * f[k][m-1];
            }
        }
    }
    return bi;
}



complex<double> bf(vector<vector<complex<double>>>& f, int k, int i, int j, int a, int b, int n, int m, int p, int q) {
    complex<double> bi = 0;
//    bi += bf1(f, k, i, j, a, b, n, m, p, q);
//    bi += bf2(f, k, i, j, a, b, n, m, p, q);
//    bi += bf3(f, k, i, j, a, b, n, m, p, q);
    bi += bf4(f, k, i, j, a, b, n, m, p, q);
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
                            if (n != m-1 && p != q-1) {
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
    
    vector<vector<complex<double>>> ff({vector<complex<double>>{0.4704433287137315,0.31278265828701396,0.020491099557798563,0.5120264351039052,
    0.4418501306816557,0.47225796949755755},vector<complex<double>>{0.5055884439289716,0.30266074435637946,0.09512521370830883,0.24918922817582706,
    0.5783175475335864,0.4971735468465169},vector<complex<double>>{0.23298975595641222,0.5093931448039285,0.3201022274576789,0.32897215154641374,
    0.4453746331978017,0.5264862023202649},vector<complex<double>>{0.030551894237901167,0.6347337871321124,0.013578078921715465,0.6730435257425468,
    0.20898496392524407,0.31517127774180836},vector<complex<double>>{0.5035952546202369,0.5821938736982794,0.27559620353757475,0.29400743122032663,
    0.3952426430295389,0.2980465216473689}});
    vector<double> JJ(5, 8.608695652173912e6);
    double UU = 1.0633270321361059e8;
    complex<double> qwe = b2(ff, 0, JJ, UU);
//    complex<double> qwe = bf1(ff, 0, 1, 0, 0, 1, 4, 4, 3, 5);
    cout << qwe << endl;
    exit(0);
    
    fin = SX::sym("f", 1, 1, 2 * L * dim);
    dU = SX::sym("dU", 1, 1, L);
    J = SX::sym("J", 1, 1, L);
    Jp = SX::sym("Jp", 1, 1, L);
    U0 = SX::sym("U0");
    mu = SX::sym("mu");
    t = SX::sym("t");
    xi = SX::sym("xi", 1, 1, L);

    Wi = SX::sym("Wi");
    Wf = SX::sym("Wf");
    tau = SX::sym("tau");
    Wt = if_else(t < tau, Wi + (Wf - Wi) * t / tau, Wf + (Wi - Wf) * (t - tau) / tau);
    SXFunction Wtf(vector<SX>{t}, vector<SX>{Wt});
    Wtf.init();
    Function Wtdt = Wtf.gradient(0, 0);
    Wtdt.init();
    SX Wpt = Wtdt.call(vector<SX>{t})[0];
    U0 = UW(Wt);
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wt * xi[i], Wt * xi[mod(i + 1)]);
        Jp[i] = JWij(Wpt * xi[i], Wpt * xi[mod(i + 1)]);
        dU[i] = UW(Wt * xi[i]) - U0;
    }

    vector<SX> params;
    params.push_back(Wi);
    params.push_back(Wf);
    params.push_back(tau);
    for (SX sx : xi) params.push_back(sx);
    params.push_back(mu);

    complex<SX> HSc = HS();

    x = SX::sym("x", fin.size());
    p = SX::sym("p", params.size());

    vector<SX> xs;
    for (int i = 0; i < x.size(); i++) {
        xs.push_back(x.at(i));
    }
    vector<SX> ps;
    for (int i = 0; i < p.size(); i++) {
        ps.push_back(p.at(i));
    }

    SX HSr = HSc.real();
    SX HSi = HSc.imag();
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

    SX HSrdftmp = HSrdff.call(vector<SX>{x, p})[0];
    SX HSidftmp = HSidff.call(vector<SX>{x, p})[0];

    ode = SX::sym("ode", 2 * L * dim);
    for (int i = 0; i < L * dim; i++) {
        ode[2 * i] = 0.5 * (HSrdftmp[2 * i] - HSidftmp[2 * i + 1]);
        ode[2 * i + 1] = 0.5 * (HSidftmp[2 * i] + HSrdftmp[2 * i + 1]);
    }
    ode_func = SXFunction(daeIn("x", x, "t", t, "p", p), daeOut("ode", ode));

    Function g;
    integrator = new CvodesInterface(ode_func, g);
    integrator->setOption("max_num_steps", 100000);
    integrator->init();
}

string DynamicsProblem::getRuntime() {
    time_period period(start_time, stop_time);
    return to_simple_string(period.length());
}

void DynamicsProblem::setParameters(double Wi, double Wf, double tau, vector<double>& xi, double mu) {
    params.clear();
    params.push_back(Wi);
    params.push_back(Wf);
    params.push_back(tau);
    for (double xii : xi) params.push_back(xii);
    params.push_back(mu);
    integrator->setOption("t0", 0);
    integrator->setOption("tf", 2 * tau);
    integrator->init();
}

void DynamicsProblem::setInitial(vector<double>& f0) {
    x0.clear();
    for (double f0i : f0) x0.push_back(f0i);
}

void DynamicsProblem::evolve() {
    integrator->setInput(x0, INTEGRATOR_X0);
    integrator->setInput(params, INTEGRATOR_P);
    integrator->evaluate();
    DMatrix xf = integrator->output(INTEGRATOR_XF);
    //      cout << xf << endl;

    //      vector<complex<double> > ff(L*dim);
    //      for (int i = 0; i < L*dim; i++) {
    //          ff[i] = complex<double>(xf[2*i].getValue(), xf[2*i+1].getValue());
    //      }
    vector<vector<complex<double>>> f0(L, vector<complex<double>>(dim));
    vector<vector<complex<double>>> ff(L, vector<complex<double>>(dim));
    vector<double> pi(L);
    double p = 0;
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            f0[i][n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
            ff[i][n] = complex<double>(xf[2 * (i * dim + n)].getValue(), xf[2 * (i * dim + n) + 1].getValue());
        }
        pi[i] = 1 - norm(dot(ff[i], f0[i]));
        p += pi[i];
    }
    p /= L;
    cout << p << endl;
}

//double DynamicsProblem::solve(vector<double>& f) {
//    nlp.setInput(params, "p");
//
//    nlp.evaluate();
//
//    Dictionary& stats = const_cast<Dictionary&> (nlp.getStats());
//    status = stats["return_status"].toString();
//    runtime = stats["t_mainloop"].toDouble();
//
//    DMatrix xout = nlp.output("x");
//    f.resize(xout.size());
//    for (int i = 0; i < xout.size(); i++) {
//        f[i] = xout.at(i);
//    }
//    return nlp.output("f").getValue();
//}

complex<SX> DynamicsProblem::HS() {

    vector<complex<SX>* > f(L);
    vector<SX> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<complex<SX>*> (&fin[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            //            norm2[i] += f[i][n].real() * f[i][n].real() + f[i][n].imag() * f[i][n].imag();
        }
    }

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

    //    complex<SX> S0 = complex<SX>(0, 0);
    complex<SX> S = complex<SX>(0, 0);

    complex<SX> Sj10, Sj20; //, Sj1, Sj2, Sj1j2, Sj1k1, Sj2k2;

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

        Sj10 = complex<SX>(0, 0);
        Sj20 = complex<SX>(0, 0);

        for (int n = 0; n <= nmax; n++) {
            Ei += (0.5 * (U0 + dU[i]) * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                Ej1 += -J[j1] * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                        * f[i][n] * f[j1][n + 1];
                Ej2 += -J[i] * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                        * f[j2][n + 1];

                if (n > 0) {
                    Ej1 += 0.5 * J[j1] * J[j1] * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1];
                    Ej2 += 0.5 * J[i] * J[i] * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1];
                }
                if (n < nmax - 1) {
                    Ej1 -= 0.5 * J[j1] * J[j1] * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j1][n] * f[i][n] * f[j1][n + 2];
                    Ej2 -= 0.5 * J[i] * J[i] * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j2][n] * f[i][n] * f[j2][n + 2];
                }

                if (n > 1) {
                    Ej1 += -J[j1] * J[j1] * g(n, n - 1) * g(n - 1, n)
                            * (eps(dU, i, j1, n, n - 1, i, j1, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                            * ~f[i][n + 1] * ~f[j1][n - 2] * f[i][n - 1] * f[j1][n];
                    Ej2 += -J[i] * J[i] * g(n, n - 1) * g(n - 1, n)
                            * (eps(dU, i, j2, n, n - 1, i, j2, n - 1, n) / (eps(U0, n, n - 1)*(eps(U0, n, n - 1) + eps(U0, n - 1, n))))
                            * ~f[i][n + 1] * ~f[j2][n - 2] * f[i][n - 1] * f[j2][n];
                }
                if (n < nmax - 2) {
                    Ej1 -= -J[j1] * J[j1] * g(n, n + 3) * g(n + 1, n + 2)
                            * (eps(dU, i, j1, n, n + 3, i, j1, n + 1, n + 2) / (eps(U0, n, n + 3)*(eps(U0, n, n + 3) + eps(U0, n + 1, n + 2))))
                            * ~f[i][n + 2] * ~f[j1][n + 1] * f[i][n] * f[j1][n + 3];
                    Ej2 -= -J[i] * J[i] * g(n, n + 3) * g(n + 1, n + 2)
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

                        Ej1 += J[j1] * g(n, m) * (eps(dU, i, j1, n, m) / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
                        Ej2 += J[i] * g(n, m) * (eps(dU, i, j2, n, m) / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];

                        Sj10 += -Jp[j1] * g(n, m) * (1 / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
                        Sj20 += -Jp[i] * g(n, m) * (1 / eps(U0, n, m))
                                * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];

                        if (n != m - 3 && m > 1 && n < nmax - 1) {
                            Ej1 += -0.5 * J[j1] * J[j1] * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                    * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                            Ej2 += -0.5 * J[i] * J[i] * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, n + 1, m - 1)))
                                    * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                        }
                        if (n != m + 1 && n > 0 && m < nmax) {
                            Ej1 -= -0.5 * J[j1] * J[j1] * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, n - 1, m + 1)))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                            Ej2 -= -0.5 * J[i] * J[i] * g(n, m) * g(n - 1, m + 1)
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
                                Ej1k1 += -0.5 * J[j1] * J[k1] * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m] * ~f[k1][q - 1]
                                        * f[i][n] * f[j1][m] * f[k1][q];
                                Ej2k2 += -0.5 * J[i] * J[j2] * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j2, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j2][m] * ~f[k2][q - 1]
                                        * f[i][n] * f[j2][m] * f[k2][q];
                                Ej1k1 -= -0.5 * J[j1] * J[k1] * g(n, m) * g(m - 1, q)
                                        * (eps(dU, i, j1, n, m) / (eps(U0, n, m) * eps(U0, m - 1, q)))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][q - 1]
                                        * f[i][n] * f[j1][m - 1] * f[k1][q];
                                Ej2k2 -= -0.5 * J[i] * J[j2] * g(n, m) * g(m - 1, q)
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
                                Ej1k1 += -0.25 * J[j1] * J[k1] * g(n, m) * g(m - 1, p + 1)
                                        * (eps(dU, i, j1, n, m, j1, k1, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                        * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][p] * f[i][n] * f[j1][m - 1] * f[k1][p + 1] -
                                        ~f[i][n + 1] * ~f[j1][m] * ~f[k1][p] * f[i][n] * f[j1][m] * f[k1][p + 1]);
                                Ej2k2 += -0.25 * J[i] * J[j2] * g(n, m) * g(m - 1, p + 1)
                                        * (eps(dU, i, j2, n, m, j2, k2, p, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, p + 1))))
                                        * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][p] * f[i][n] * f[j2][m - 1] * f[k2][p + 1] -
                                        ~f[i][n + 1] * ~f[j2][m] * ~f[k2][p] * f[i][n] * f[j2][m] * f[k2][p + 1]);
                            }
                        }

                        Ej1k1 += 0.5 * J[j1] * J[k1] * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][n]
                                * f[i][n] * f[j1][m - 1] * f[k1][n + 1] -
                                ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
                                * f[i][n] * f[j1][m] * f[k1][n + 1]);
                        Ej2k2 += 0.5 * J[j2] * J[i] * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][n]
                                * f[i][n] * f[j2][m - 1] * f[k2][n + 1] -
                                ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
                                * f[i][n] * f[j2][m] * f[k2][n + 1]);

                        Ej1k1 += -J[j1] * J[k1] * g(n, m) * g(m - 1, m)
                                * (eps(dU, i, j1, n, m, j1, k1, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][m - 1] * f[i][n] * f[j1][m - 1] * f[k1][m] -
                                ~f[i][n + 1] * ~f[j1][m] * ~f[k1][m - 1] * f[i][n] * f[j1][m] * f[k1][m]);
                        Ej2k2 += -J[i] * J[j2] * g(n, m) * g(m - 1, m)
                                * (eps(dU, i, j2, n, m, j2, k2, m - 1, m) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, m - 1, m))))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][m - 1] * f[i][n] * f[j2][m - 1] * f[k2][m] -
                                ~f[i][n + 1] * ~f[j2][m] * ~f[k2][m - 1] * f[i][n] * f[j2][m] * f[k2][m]);

                        if (m != n - 1 && n != m && m < nmax && n > 0) {
                            Ej1 += -0.25 * J[j1] * J[j1] * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j1, n, m, i, j1, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                    * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n - 1] * f[j1][m + 1];
                            Ej2 += -0.25 * J[i] * J[i] * g(n, m) * g(n - 1, m + 1)
                                    * (eps(dU, i, j2, n, m, i, j2, m, n) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n - 1, m + 1))))
                                    * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n - 1] * f[j2][m + 1];
                        }
                        if (n != m - 3 && n != m - 2 && n < nmax - 1 && m > 1) {
                            Ej1 -= -0.25 * J[j1] * J[j1] * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j1, n, m, i, j1, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                    * ~f[i][n + 2] * ~f[j1][m - 2] * f[i][n] * f[j1][m];
                            Ej2 -= -0.25 * J[i] * J[i] * g(n, m) * g(n + 1, m - 1)
                                    * (eps(dU, i, j2, n, m, i, j2, m - 2, n + 2) / (eps(U0, n, m)*(eps(U0, n, m) + eps(U0, n + 1, m - 1))))
                                    * ~f[i][n + 2] * ~f[j2][m - 2] * f[i][n] * f[j2][m];
                        }
                    }
                }
            }
        }

        //        Ei /= norm2[i];
        //        Ej1 /= norm2[i] * norm2[j1];
        //        Ej2 /= norm2[i] * norm2[j2];
        //        Ej1j2 /= norm2[i] * norm2[j1] * norm2[j2];
        //        Ej1k1 /= norm2[i] * norm2[j1] * norm2[k1];
        //        Ej2k2 /= norm2[i] * norm2[j2] * norm2[k2];

        E += Ei;
        E += Ej1;
        E += Ej2;
        E += Ej1j2;
        E += Ej1k1;
        E += Ej2k2;

        S += Sj10;
        S += Sj20;

        //        S += Sj10;
        //        S += Sj20;
        //        S += Sj1;
        //        S += Sj2;
        //        S += Sj1j2;
        //        S += Sj1k1;
        //        S += Sj2k2;
    }

    return complex<SX>(S.real(), -E.real());

    //    return -complex<SX>(0, 1)*E + S; 

    //    return E.real();// - (complex<SX>(0, 1) * S).real();
}
