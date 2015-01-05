/* 
 * File:   casadi.hpp
 * Author: Abuenameh
 *
 * Created on 06 November 2014, 17:45
 */

#ifndef CASADI_HPP
#define	CASADI_HPP

#include <casadi/casadi.hpp>
#include <casadi/solvers/rk_integrator.hpp>
#include <casadi/interfaces/sundials/cvodes_interface.hpp>

using namespace casadi;

#include <boost/date_time.hpp>

using namespace boost::posix_time;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"

class DynamicsProblem {
public:
    DynamicsProblem(double Wi, double Wf, double mu_, vector<double>& xi, vector<double>& f0);
        ~DynamicsProblem() { delete lopt; delete integrator; }

        void setTau(double tau_);
//    void setParameters(double Wi, double Wf, double tau, vector<double>& xi, double mu);

    double E(const vector<double>& f, vector<double>& grad);
    double E(const vector<double>& f, double t);

    void evolve(int nsteps);

    vector<double> getGS() { return x0; };
    string& getGSRuntime() { return gsruntime; }
    string& getRuntime() { return runtime; }

    string& getGSResult() {
        return gsresult;
    }
    string& getResult() {
        return result;
    }
    
    double getQ() { return Q; }
    double getRho() { return pd; }
//    vector<vector<double>> getBs() { return bv; }
    vector<complex<double>> getB0() { return b0; }
    vector<complex<double>> getBf() { return bf; }
    double getEi() { return E0; }
    double getEf() { return Ef; }
    double getU0() { return U00; }
    vector<double> getJ0() { return J0; }
    vector<vector<complex<double>>> getF0() { return f0; }
    vector<vector<complex<double>>> getFf() { return ff; }
    
    void start() 
    {
        start_time = microsec_clock::local_time();
    }

    void stop() {
        stop_time = microsec_clock::local_time();
    }

private:

    ptime start_time;
    ptime stop_time;
    
    double scale = 1;

    void setInitial(vector<double>& f0);
    void solve();

    complex<SX> HS();
    SX W();
    SX energy();
    SX energya();
    SX energy0();
    SX energync();
    SX canonical();
    SX canonicala();

    vector<SX> fin;
    SX U0;
    vector<SX> dU;
    vector<SX> J;
    double mu;
    SX tau;

    SX Wt;

    SX t;
    SX x;
    SX p;
    SX gsp;

    double tf;

    double U00;
    vector<double> J0;
    
    opt* lopt;

    SX ode;
    SXFunction ode_func;
    CvodesInterface* integrator;
//    RkIntegrator* integrator;

    vector<double> params;
    vector<double> gsparams;
    vector<double> x0;
    
    SXFunction Efunc;
    Function Egradf;

    string gsruntime;
    string gsresult;
    
    string runtime;
    string result;
    
    double E0;
    double Ef;
    double Q;
    double pd;
//    vector<vector<double>> bv;
    vector<complex<double>> b0;
    vector<complex<double>> bf;
    
    vector<vector<complex<double>>> f0;
    vector<vector<complex<double>>> ff;
    
};

double energyfunc(const vector<double>& x, vector<double>& grad, void *data);

#endif	/* CASADI_HPP */

