/* 
 * File:   casadi.hpp
 * Author: Abuenameh
 *
 * Created on 06 November 2014, 17:45
 */

#ifndef CASADI_HPP
#define	CASADI_HPP

#include <casadi/casadi.hpp>
#include <casadi/interfaces/sundials/cvodes_interface.hpp>

using namespace casadi;

#include <boost/date_time.hpp>

using namespace boost::posix_time;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"

class DynamicsProblem {
public:
    DynamicsProblem();
        ~DynamicsProblem() { delete lopt; delete integrator; }

    void setParameters(double Wi, double Wf, double tau, vector<double>& xi, double mu);
    void setInitial(vector<double>& f0);

    double E(const vector<double>& f, vector<double>& grad);
    double E(const vector<double>& f, double t);

    void solve();
    void evolve();

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
    vector<double> getBs() { return bv; }
    double getEi() { return E0; }
    double getEf() { return E1; }
    
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

    complex<SX> HS();
    SX W();
    SX energy();
    SX canonical();

    vector<SX> fin;
    SX U0;
    vector<SX> dU;
    vector<SX> J;
    vector<SX> Jp;
    SX mu;
    vector<SX> xi;
    SX Wi;
    SX Wf;
    SX tau;

    SX Wt;

    SX t;
    SX x;
    SX p;
    SX gsp;

    double tf;

    SXFunction Ufunc;
    vector<SXFunction> Jfunc;

    double U0d;
    vector<double> Jd;
    
    opt* lopt;

    SX ode;
    SXFunction ode_func;
    CvodesInterface* integrator;

    vector<double> params;
    vector<double> gsparams;
    vector<double> x0;
    
    SXFunction Ef;
    Function Egradf;

    string gsruntime;
    string gsresult;
    
    string runtime;
    string result;
    
    double E0;
    double E1;
    double Q;
    double pd;
    vector<double> bv;
};

double energyfunc(const vector<double>& x, vector<double>& grad, void *data);

#endif	/* CASADI_HPP */

