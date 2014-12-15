/* 
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on 17 November 2014, 22:05
 */

#include <cstdlib>
#include <queue>

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/progress.hpp>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time.hpp>

using namespace boost;
using namespace boost::random;
using namespace boost::filesystem;
using namespace boost::posix_time;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"
#include "mathematica.hpp"
#include "casadi.hpp"

#include <casadi/interfaces/sundials/cvodes_interface.hpp>

double UW(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

struct input {
    double tau;
};

struct results {
    double tau;
    double Ei;
    double Ef;
    double Q;
    double p;
    vector<vector<double>> bs;
    string runtime;
};

boost::mutex progress_mutex;
boost::mutex inputs_mutex;
boost::mutex problem_mutex;

void threadfunc(double Wi, double Wf, double mu, vector<double> xi, int nsteps, queue<input>& inputs, vector<results>& res, progress_display& progress) {
    DynamicsProblem* prob;

    {
        boost::mutex::scoped_lock lock(problem_mutex);
        prob = new DynamicsProblem();
    }

    for (;;) {
        input in;
        {
            boost::mutex::scoped_lock lock(inputs_mutex);
            if (inputs.empty()) {
                break;
            }
            in = inputs.front();
            inputs.pop();
        }
        double tau = in.tau;

        results pointRes;
        pointRes.tau = tau;

        //    int ndim = 2 * L*dim;

        vector<double> f0(2 * L*dim, 1);

        {
            boost::mutex::scoped_lock lock(problem_mutex);
            prob->setParameters(Wi, Wf, tau, xi, mu);
        }
        prob->setInitial(f0);
        try {
        prob->solve();
            prob->evolve(nsteps);
            pointRes.Ei = prob->getEi();
            pointRes.Ef = prob->getEf();
            pointRes.Q = prob->getQ();
            pointRes.p = prob->getRho();
            pointRes.bs = prob->getBs();
            pointRes.runtime = prob->getRuntime();
        }
        catch (std::exception& e) {
            cerr << "Failed at tau = " << tau << endl;
            cerr << e.what() << endl;
            pointRes.Ei = numeric_limits<double>::quiet_NaN();
            pointRes.Ef = numeric_limits<double>::quiet_NaN();
            pointRes.Q = numeric_limits<double>::quiet_NaN();
            pointRes.p = numeric_limits<double>::quiet_NaN();
            pointRes.bs = vector<vector<double>>();
            pointRes.runtime = "Failed";
        }

        //        pointRes.Ei = prob->getEi();
        //        pointRes.Ef = prob->getEf();
        //        pointRes.Q = prob->getQ();
        //        pointRes.p = prob->getRho();
        //        pointRes.bs = prob->getBs();
        //        pointRes.runtime = prob->getRuntime();

        {
            boost::mutex::scoped_lock lock(inputs_mutex);
            res.push_back(pointRes);
        }

        {
            boost::mutex::scoped_lock lock(progress_mutex);
            ++progress;
        }
    }

    {
        boost::mutex::scoped_lock lock(problem_mutex);
        delete prob;
    }


}

/*
 * 
 */
int main(int argc, char** argv) {

    ptime begin = microsec_clock::local_time();

    mt19937 rng;
    uniform_real_distribution<> uni(-1, 1);

    int seed = lexical_cast<int>(argv[1]);

    double Wi = lexical_cast<double>(argv[2]);
    double Wf = lexical_cast<double>(argv[3]);

    double mu = lexical_cast<double>(argv[4]);

    double Ui = UW(Wi);

    double D = lexical_cast<double>(argv[5]);

    double taui = lexical_cast<double>(argv[6]);
    double tauf = lexical_cast<double>(argv[7]);
    int ntaus = lexical_cast<int>(argv[8]);
    //    double tf = 2*tau;
    int nsteps = lexical_cast<int>(argv[9]);

    //    double dt = lexical_cast<double>(argv[7]);
    //    int dnsav = lexical_cast<int>(argv[8]);

    //    int nsteps = (int) ceil(2 * tau / dt);
    //    double h = 2 * tau / nsteps;


    int numthreads = lexical_cast<int>(argv[10]);

    int resi = lexical_cast<int>(argv[11]);
    //    int resi = 0;
    //    if (argc > 9) {
    //        resi = lexical_cast<int>(argv[9]);
    //    }

#ifdef AMAZON
    path resdir("/home/ubuntu/Results/Canonical Transformation Dynamical Gutzwiller");
#else
    path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Dynamical Gutzwiller");
    //        path resdir("/Users/Abuenameh/Documents/Simulation Results/Dynamical Gutzwiller Hartmann Comparison");
#endif
    if (!exists(resdir)) {
        cerr << "Results directory " << resdir << " does not exist!" << endl;
        exit(1);
    }
    ostringstream oss;
    oss << "res." << resi << ".txt";
    path resfile = resdir / oss.str();
    //#ifndef AMAZON
    while (exists(resfile)) {
        resi++;
        oss.str("");
        oss << "res." << resi << ".txt";
        resfile = resdir / oss.str();
    }
    //#endif
    //        if (seed == -1) {
    //            resi = -1;
    if (seed < 0) {
        resi = seed;
        oss.str("");
        oss << "res." << resi << ".txt";
        resfile = resdir / oss.str();
    }
    vector<double> xi(L, 1);
    rng.seed(seed);
    if (seed > -1) {
        for (int j = 0; j < L; j++) {
            xi[j] = (1 + D * uni(rng));
        }
    }

    //        double Ui = UW(Wi);
    double mui = mu * 2 * Ui;

    filesystem::ofstream os(resfile);
    printMath(os, "seed", resi, seed);
    printMath(os, "Delta", resi, D);
    printMath(os, "Wres", resi, Wi);
    printMath(os, "mures", resi, mui);
    printMath(os, "Ures", resi, Ui);
    printMath(os, "xires", resi, xi);
    os << flush;

    printMath(os, "tauires", resi, taui);
    printMath(os, "taufres", resi, tauf);
    printMath(os, "ntausres", resi, ntaus);

    printMath(os, "nsteps", resi, nsteps);
    os << flush;

    printMath(os, "Wires", resi, Wi);
    printMath(os, "Wfres", resi, Wf);
    os << flush;

    cout << "Res: " << resi << endl;

    queue<input> inputs;
    for (int i = 0; i < ntaus; i++) {
        input in;
        double tau = taui + i * (tauf - taui) / ntaus;
        in.tau = tau;
        inputs.push(in);
    }

    progress_display progress(inputs.size());

    vector<results> res;

    thread_group threads;
    for (int i = 0; i < numthreads; i++) {
        threads.create_thread(bind(&threadfunc, Wi, Wf, mu, xi, nsteps, boost::ref(inputs), boost::ref(res), boost::ref(progress)));
    }
    threads.join_all();

    vector<double> taures;
    vector<double> Eires;
    vector<double> Efres;
    vector<double> Qres;
    vector<double> pres;
    vector<vector<vector<double>>> bsres;
    vector<string> runtimeres;

    for (results ires : res) {
        taures.push_back(ires.tau);
        Eires.push_back(ires.Ei);
        Efres.push_back(ires.Ef);
        Qres.push_back(ires.Q);
        pres.push_back(ires.p);
        bsres.push_back(ires.bs);
        runtimeres.push_back(ires.runtime);
    }

    printMath(os, "taures", resi, taures);
    printMath(os, "Eires", resi, Eires);
    printMath(os, "Efres", resi, Efres);
    printMath(os, "Qres", resi, Qres);
    printMath(os, "pres", resi, pres);
    printMath(os, "bsres", resi, bsres);
    printMath(os, "runtime", resi, runtimeres);

    ptime end = microsec_clock::local_time();
    time_period period(begin, end);
    cout << endl << period.length() << endl << endl;

    os << "totalruntime[" << resi << "]=\"" << period.length() << "\";" << endl;

    return 0;

    //        cout << setprecision(10);
    //
    //    double Wi = 2e11;
    //    double Wf = 1e11;
    //    double tau = 1e-6;
    //    double mu = 0.5;
    //    vector<double> xi(L, 1);
    //
    //    int ndim = 2 * L*dim;
    //
    //    vector<double> f0(ndim, 1);
    //
    //    vector<double> Eiv, Efv, Qv, pv;
    //    vector<vector<double>> bv;
    //
    //    DynamicsProblem prob;
    //
    //    for (int i = 0; i < 10; i++) {
    //        prob.setParameters(Wi, Wf, tau + i * 0.1e-6, xi, mu);
    //        prob.setInitial(f0);
    //        prob.solve();
    //        prob.evolve();
    //
    //        Eiv.push_back(prob.getEi());
    //        Efv.push_back(prob.getEf());
    //        Qv.push_back(prob.getQ());
    //        pv.push_back(prob.getRho());
    //        bv.push_back(prob.getBs());
    //    }
    //    
    //    cout << math(bv) << endl;
    //    cout << math(Eiv) << endl;
    //    cout << math(Efv) << endl;
    //    cout << math(Qv) << endl;
    //    cout << math(pv) << endl;

    //    cout << prob.getGSRuntime() << endl;
    //    cout << prob.getRuntime() << endl;

    return 0;


    return 0;
}

