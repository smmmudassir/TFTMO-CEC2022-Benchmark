#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include "../algorithms/MOSAIC_R.hpp"
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag=0,func_flag=0,*SS=nullptr;
#include "cec22_test_func.cpp"
static const double OPT[13]={0,300,400,600,800,900,1800,2000,2200,2300,2400,2600,2700};
int main(int argc,char**argv){
 if(argc!=6){std::cerr<<"usage: mosaicr_cec22 DIM FUNC RUNS SEED BUDGET\n";return 2;}
 int D=std::atoi(argv[1]),func=std::atoi(argv[2]),runs=std::atoi(argv[3]);uint64_t base=std::strtoull(argv[4],nullptr,10);long long budget=std::strtoll(argv[5],nullptr,10);
 for(int r=0;r<runs;++r){uint64_t seed=base+r;auto objective=[&](const mosaicr::Vec&xx){std::vector<double>x=xx;double f=0;cec22_test_func(x.data(),&f,D,1,func);return f;};auto res=mosaicr::optimize(objective,D,-100,100,budget,seed);double err=res.f-OPT[func];if(err<1e-8)err=0;if(!std::isfinite(err)||res.fes!=budget)return 3;std::cout<<"RESULT,"<<D<<","<<func<<","<<(r+1)<<","<<seed<<","<<std::scientific<<std::setprecision(17)<<err<<","<<res.fes<<"\n";}
}