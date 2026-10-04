extern "C" {
#include <stddef.h>
}
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag=0,func_flag=0,*SS=nullptr;
#include "cec22_test_func.cpp"
extern "C" double cec22_eval(const double* x, int dim, int func) {
    double f=0.0;
    cec22_test_func(const_cast<double*>(x), &f, dim, 1, func);
    return f;
}
