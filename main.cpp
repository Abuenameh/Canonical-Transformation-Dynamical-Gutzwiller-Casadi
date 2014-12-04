/* 
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on 17 November 2014, 22:05
 */

#include <cstdlib>

using namespace std;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"
#include "casadi.hpp"

#include <casadi/interfaces/sundials/cvodes_interface.hpp>

/*
 * 
 */
int main(int argc, char** argv) {
    
    cout << setprecision(10);

    double Wi = 2e11;
    double Wf = 1e11;
    double tau = 1e-6;
    double mu = 0.5;
    vector<double> xi(L, 1);

    int ndim = 2 * L*dim;

    GroundStateProblem gsprob;
    gsprob.setParameters(Wi, xi, mu);
    gsprob.setTheta(0);

    opt lopt(LD_LBFGS, ndim);
    lopt.set_lower_bounds(-1);
    lopt.set_upper_bounds(1);
    lopt.set_min_objective(energyfunc, &gsprob);

    vector<double> f0(ndim, 1);

    double E0;
    string result0;
    try {
        gsprob.start();
        result res = lopt.optimize(f0, E0);
        gsprob.stop();
        result0 = to_string(res);
    }
    catch (std::exception& e) {
        gsprob.stop();
        result res = lopt.last_optimize_result();
        result0 = to_string(res) + ": " + e.what();
        cout << e.what() << endl;
        E0 = numeric_limits<double>::quiet_NaN();
    }

    DynamicsProblem prob;
    prob.setParameters(Wi, Wf, tau, xi, mu);
    //    vector<double> f0(2*L*dim, 1/sqrt(2.*dim));
    prob.setInitial(f0);
    prob.evolve();

    return 0;

    SX q = SX::sym("q", 2);
    SX w = SX::sym("w");
    SX q2 = q[0] * q[1] * q[1] + 3 * w * w*w;
    SX w2 = w * w + 2 * q[0] * q[0] * q[1];
    SXFunction qw = SXFunction(vector<SX>{q, w}, vector<SX>{q2, w2});
    qw.init();
    Function qwdiff = qw.gradient(0, 1);
    qwdiff.init();
    vector<SX> out = qwdiff.call(vector<SX>{q, w});
    cout << out << endl;

    return 0;

    SX t = SX::sym("t");
    SX u = SX::sym("u");
    //SX s = SX::sym("s");
    //SX v = SX::sym("v");
    //SX m = SX::sym("m");
    //      vector<SX> ode(3);
    //      ode[0] = v;
    //      ode[1] = (u - 0.02 * v * v) / m;
    //      ode[2] = -0.01 * u * u;
    //      vector<SX> quad(2);
    //      quad[0] = s;
    //      quad[1] = 0.45 * m * v * v;
    //      vector<vector<SX> > ode_vars(DAE_NUM_IN);
    //      ode_vars[DAE_T].push_back(t);
    //      ode_vars[DAE_X].push_back(s); ode_vars[DAE_X].push_back(v); ode_vars[DAE_X].push_back(m);
    //      ode_vars[DAE_P].push_back(u);
    SX x = SX::sym("x", 3);
    //      vector<SX> ode(3);
    SX ode = SX::sym("ode", 3);
    ode[0] = x[1];
    ode[1] = (u - 0.02 * x[1] * x[1]) / x[2];
    ode[2] = -0.01 * u * u;
    //      SX qwe = SX::sym("qwe", 3);
    //      qwe = substitute(vector<SX>{E}, , )
    //      SXFunction asd(x,x);
    SXFunction ode_func(daeIn("x", x, "t", t, "p", u), daeOut("ode", ode));
    //      SXFunction ode_func(ode_vars, ode);
    Function g;
    //      SXFunction quad_func(ode_vars, quad);
    CvodesInterface integrator(ode_func, g);
    integrator.setOption("t0", 0.0);
    integrator.setOption("tf", 1.0);
    integrator.init();
    vector<double> x0(3);
    fill(x0.begin(), x0.end(), 0.0);
    x0[0] = 0;
    x0[1] = 1;
    x0[2] = 10;
    integrator.setInput(x0, INTEGRATOR_X0);
    vector<double> p(1);
    p[0] = 0.15;
    integrator.setInput(p, INTEGRATOR_P);
    integrator.evaluate();
    //      vector<double> xf = integrator.output(INTEGRATOR_XF);
    DMatrix xf = integrator.output(INTEGRATOR_XF);
    cout << "Solution of ODE: (";
    for (int j = 0; j < 3; ++j) {
        cout << xf[j] << ",";
    }
    cout << "\b)\n";

    //    SX x = SX::sym("x", 5);
    //    SX y = x[0]*x[1]+2*x[1]+3*x[2]+x[3]+10*x[4];
    //    SXFunction f = SXFunction(vector<SX>{x}, vector<SX>{y});
    //    f.init();
    //    vector<vector<double> > fin;
    //    vector<double> xin = vector<double>{1,2,3,4,5};
    //    fin.push_back(xin);
    //    f.setInput(vector<double>{1,2,3,4,5});
    //    f.evaluate();
    //    vector<SX> qwe = f.call(vector<SX>{x});
    //    cout << f << endl;
    //    cout << qwe << endl;
    ////    vector<double> fout;
    //    SX fout = f.getOutput();
    //    cout << fout << endl;
    //    Function df = f.gradient(0,0);
    //    df.init();
    ////    df.setInput(vector<double>{1,2,3,4,5});
    ////    df.evaluate();
    //    vector<SX> asd = df.call(vector<SX>{x});
    //    SX zxc = df.call(vector<SX>{x})[0];
    //    cout << asd << endl;
    //    cout << zxc << endl;
    ////    SX dfout = df.getOutput();
    ////    cout << dfout << endl;

    return 0;
}

