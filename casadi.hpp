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

//#include <pagmo/src/pagmo.h>
//
//using namespace pagmo;
//using namespace pagmo::algorithm;
//using namespace pagmo::problem;

#include "gutzwiller.hpp"

class DynamicsProblem {
public:
    DynamicsProblem();
//    ~DynamicsProblem() { delete integrator; }
    
//    void setParameters(double U0, vector<double>& dU, vector<double>& J, double mu);
void setParameters(double Wi, double Wf, double tau, vector<double>& xi, double mu);
void setInitial(vector<double>& f0);

void evolve();
    
//    double solve(vector<double>& f);
//    
//    double E(const vector<double>& f, vector<double>& grad);
    
    string& getStatus() { return status; }
    string getRuntime();
    
    void start() { start_time = microsec_clock::local_time(); }
    void stop() { stop_time = microsec_clock::local_time(); }
    
private:
    
    ptime start_time;
    ptime stop_time;
    
    complex<SX> HS();
    SX W();
    
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
    
    double tf;
    
    SXFunction Ufunc;
    vector<SXFunction> Jfunc;
    
    double U0d;
    vector<double> Jd;
    
    SX ode;
    SXFunction ode_func;
    CvodesInterface* integrator;
    
    vector<double> params;
    vector<double> x0;
    
    SXFunction Ef;
    Function Egradf;
    NlpSolver nlp;
    
    string status;
    double runtime;
};

class GroundStateProblem {
public:
    GroundStateProblem();
    
    void setParameters(double U0, vector<double>& dU, vector<double>& J, double mu);
    void setTheta(double theta);
    
    double solve(vector<double>& f);
    
    double E(const vector<double>& f, vector<double>& grad);
    
    string& getStatus() { return status; }
    string getRuntime();
    
    void start() { start_time = microsec_clock::local_time(); }
    void stop() { stop_time = microsec_clock::local_time(); }
    
private:
    
    ptime start_time;
    ptime stop_time;
    
    SX energy();
    
    vector<SX> fin;
    SX U0;
    vector<SX> dU;
    vector<SX> J;
    SX mu;
    SX theta;
    
    SX x;
    SX p;
    
    vector<double> params;
    
    SXFunction Ef;
    Function Egradf;
    NlpSolver nlp;
    
    string status;
    double runtime;
};

double energyfunc(const vector<double>& x, vector<double>& grad, void *data);

#endif	/* CASADI_HPP */

