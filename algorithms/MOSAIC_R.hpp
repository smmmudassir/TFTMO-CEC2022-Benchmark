#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
namespace mosaicr {
using Vec=std::vector<double>;
struct Module{Vec q;double scale=0,utility=0;int age=0;};
struct Arm{std::vector<Vec> U;double radius=0,reward=0;int pulls=0;};
struct Result{Vec x;double f=std::numeric_limits<double>::infinity();long long fes=0;};
inline double dot(const Vec&a,const Vec&b){double s=0;for(size_t i=0;i<a.size();++i)s+=a[i]*b[i];return s;}
inline double norm(const Vec&a){return std::sqrt(std::max(0.0,dot(a,a)));}
inline Vec add(const Vec&a,const Vec&b){Vec r(a.size());for(size_t i=0;i<a.size();++i)r[i]=a[i]+b[i];return r;}
inline Vec sub(const Vec&a,const Vec&b){Vec r(a.size());for(size_t i=0;i<a.size();++i)r[i]=a[i]-b[i];return r;}
inline void axpy(Vec&y,double a,const Vec&x){for(size_t i=0;i<y.size();++i)y[i]+=a*x[i];}
inline Vec proj(const std::vector<Vec>&U,const Vec&v){Vec r(v.size(),0);for(const auto&q:U)axpy(r,dot(q,v),q);return r;}
inline std::vector<Vec> orthonormalize(const std::vector<Vec>&cols,int D,int rank,std::mt19937_64&rng){
 std::normal_distribution<double>N(0,1);std::vector<Vec>Q;
 auto push=[&](Vec u){for(const auto&q:Q)axpy(u,-dot(q,u),q);double n=norm(u);if(n>1e-12){for(double&x:u)x/=n;Q.push_back(std::move(u));return true;}return false;};
 for(const auto&v:cols){if((int)Q.size()>=rank)break;push(v);}while((int)Q.size()<rank){Vec u(D);for(double&x:u)x=N(rng);push(std::move(u));}return Q;
}
inline Vec repair(Vec y,const Vec&lb,const Vec&ub,const Vec&p){for(size_t j=0;j<y.size();++j){if(y[j]<lb[j])y[j]=.5*(lb[j]+p[j]);if(y[j]>ub[j])y[j]=.5*(ub[j]+p[j]);y[j]=std::min(ub[j],std::max(lb[j],y[j]));}return y;}
template<class Objective>
Result optimize(Objective&&obj,int D,double lo,double hi,long long max_fes,uint64_t seed,int NP=0,int A=4,int rank=0,int K=4){
 if(D<=0||max_fes<=0)throw std::runtime_error("bad args");std::mt19937_64 rng(seed);std::uniform_real_distribution<double>U01(0,1),UX(lo,hi);std::normal_distribution<double>N01(0,1);std::cauchy_distribution<double>C01(0,1);
 if(NP<=0)NP=std::max(18,std::min(40,(int)std::lround(10+3*std::sqrt((double)D))));if(rank<=0)rank=std::min(4,std::max(2,D/10));rank=std::max(1,std::min(rank,D));
 Vec lb(D,lo),ub(D,hi),span(D,hi-lo);double spanN=norm(span);std::vector<Vec>X(NP,Vec(D));std::vector<double>f(NP);long long fes=0;
 for(int i=0;i<NP;++i){for(double&x:X[i])x=UX(rng);f[i]=obj(X[i]);++fes;if(fes>=max_fes)break;}int bi=(int)(std::min_element(f.begin(),f.end())-f.begin());Result best{X[bi],f[bi],fes};if(fes>=max_fes)return best;
 std::vector<std::vector<Module>>banks(NP);std::vector<std::vector<Arm>>arms(NP,std::vector<Arm>(A));double base=.08*spanN/std::sqrt((double)D);
 for(int i=0;i<NP;++i)for(int m=0;m<A;++m){arms[i][m].U=orthonormalize({},D,rank,rng);arms[i][m].radius=base;}
 std::vector<int>stagn(NP,0);int maxAge=std::max(5,(int)(.12*max_fes/NP)),stagLim=std::max(5,(int)(.04*max_fes/NP));std::vector<std::pair<double,Vec>>archive;
 auto refresh=[&](int i,int m){std::vector<Vec>c;auto t=banks[i];std::sort(t.begin(),t.end(),[](auto&a,auto&b){return a.utility>b.utility;});for(auto&z:t)c.push_back(z.q);arms[i][m].U=orthonormalize(c,D,rank,rng);};
 auto addmod=[&](int i,const Vec&d,double u){double nd=norm(d);if(nd<=1e-15)return;Vec q=d;for(double&x:q)x/=nd;if(!banks[i].empty()){double mx=-1;int kk=0;for(int k=0;k<(int)banks[i].size();++k){double c=std::abs(dot(q,banks[i][k].q));if(c>mx){mx=c;kk=k;}}if(1-mx<.08){auto&z=banks[i][kk];z.utility=.8*z.utility+.2*u;z.scale=.8*z.scale+.2*nd;z.age=0;return;}}banks[i].push_back({q,nd,u,0});std::sort(banks[i].begin(),banks[i].end(),[](auto&a,auto&b){return a.utility>b.utility;});if((int)banks[i].size()>K)banks[i].resize(K);int mm=0;for(int m=1;m<A;++m)if(arms[i][m].pulls<arms[i][mm].pulls)mm=m;refresh(i,mm);};
 while(fes<max_fes){
  Vec mean(D,0);for(auto&x:X)axpy(mean,1.0/NP,x);std::vector<double>ds(NP);for(int i=0;i<NP;++i)ds[i]=norm(sub(X[i],mean));std::nth_element(ds.begin(),ds.begin()+NP/2,ds.end());double delta=ds[NP/2]/(spanN+1e-15),dn=delta/(delta+.08),alpha=.10+.75*(dn/(dn+.15)),beta=.35*(1+1.25*(1-dn));
  std::vector<int>ord(NP);std::iota(ord.begin(),ord.end(),0);std::sort(ord.begin(),ord.end(),[&](int a,int b){return f[a]<f[b];});int pc=std::max(2,(int)std::ceil(.20*NP));std::vector<int>elite(ord.begin(),ord.begin()+pc);
  for(int j:elite)archive.push_back({f[j],X[j]});std::sort(archive.begin(),archive.end(),[](auto&a,auto&b){return a.first<b.first;});if((int)archive.size()>std::max(8,pc))archive.resize(std::max(8,pc));
  for(int i=0;i<NP&&fes<max_fes;++i){
   std::vector<Module>kept;for(auto z:banks[i]){z.age++;z.utility*=.985;if(z.age<=maxAge&&z.utility>1e-12)kept.push_back(z);}banks[i]=std::move(kept);
   if(U01(rng)<.20){int dnr=elite[rng()%elite.size()];if(dnr!=i&&!banks[dnr].empty()){auto c=banks[dnr];std::sort(c.begin(),c.end(),[](auto&a,auto&b){return a.utility>b.utility;});for(auto&z:c){double nov=1;if(!banks[i].empty()){double mx=0;for(auto&q:banks[i])mx=std::max(mx,std::abs(dot(z.q,q.q)));nov=1-mx;}if(nov>=.08){auto zz=z;zz.utility*=.75;zz.age=0;banks[i].push_back(zz);std::sort(banks[i].begin(),banks[i].end(),[](auto&a,auto&b){return a.utility>b.utility;});if((int)banks[i].size()>K)banks[i].resize(K);break;}}}}
   int total=1;for(auto&a:arms[i])total+=a.pulls;int m=0;double bs=-1e300;for(int mm=0;mm<A;++mm){auto&a=arms[i][mm];double ex=.35*std::sqrt(std::log((double)total+1)/(a.pulls+1.0)),nov=0;if(!banks[i].empty()){for(auto&z:banks[i])nov+=1-norm(proj(a.U,z.q));nov/=banks[i].size();}double s=a.reward+ex+.10*nov;if(s>bs){bs=s;m=mm;}}
   auto&arm=arms[i][m];int pb=elite[rng()%elite.size()];Vec g=sub(X[pb],X[i]);int r1,r2;do{r1=rng()%NP;}while(r1==i);do{r2=rng()%NP;}while(r2==i||r2==r1);Vec diff=sub(X[r1],X[r2]),z(rank);for(double&v:z)v=N01(rng);Vec local(D,0);for(int k=0;k<rank;++k)axpy(local,z[k],arm.U[k]);Vec pg=proj(arm.U,g),pd=proj(arm.U,diff),orth=sub(diff,pd);double F=std::max(.10,std::min(1.0,.45+.15*C01(rng)));Vec d(D,0);axpy(d,F,g);axpy(d,.55*F,diff);axpy(d,alpha,pg);axpy(d,arm.radius,local);axpy(d,.35*beta,orth);Vec yy=repair(add(X[i],d),lb,ub,X[i]);double fy=obj(yy);++fes;arm.pulls++;
   if(fy<f[i]){double old=f[i],rw=std::min(1e6,std::max(0.0,(old-fy)/(std::abs(old)+1e-12)));arm.reward=.85*arm.reward+.15*rw;arm.radius=std::min(.30*spanN/std::sqrt((double)D),1.05*arm.radius);X[i]=yy;f[i]=fy;stagn[i]=0;addmod(i,d,rw+1e-12);if(fy<best.f){best.f=fy;best.x=yy;}}else{arm.reward*=.85;arm.radius=std::max(1e-8,.92*arm.radius);stagn[i]++;}
   if(fes<max_fes&&stagn[i]>=stagLim){auto prod=banks[i];std::sort(prod.begin(),prod.end(),[](auto&a,auto&b){return a.utility>b.utility;});if((int)prod.size()>rank)prod.resize(rank);std::vector<Vec>cols;for(auto&z:prod)cols.push_back(z.q);auto Q=orthonormalize(cols,D,std::max(1,std::min(rank,(int)std::max<size_t>(1,prod.size()))),rng);Vec e=archive.empty()?best.x:archive[rng()%std::min<size_t>(3,archive.size())].second,ret=proj(Q,sub(X[i],e)),noise(D);for(double&v:noise)v=N01(rng);Vec comp=sub(noise,proj(Q,noise));double nc=norm(comp),sig=.18*(1-.7*(double)fes/max_fes);Vec yr=e;axpy(yr,1,ret);if(nc>1e-15)for(int j=0;j<D;++j)yr[j]+=sig*span[j]*comp[j]/nc;yr=repair(yr,lb,ub,e);double fr=obj(yr);++fes;if(fr<f[i]){Vec dr=sub(yr,X[i]);double rw=std::max(0.0,(f[i]-fr)/(std::abs(f[i])+1e-12));X[i]=yr;f[i]=fr;addmod(i,dr,rw+1e-12);if(fr<best.f){best.f=fr;best.x=yr;}}banks[i]=prod;if((int)banks[i].size()>std::max(1,K/2))banks[i].resize(std::max(1,K/2));for(int mm=0;mm<A;++mm)if(mm!=m&&U01(rng)<.5){refresh(i,mm);arms[i][mm].reward*=.5;}stagn[i]=0;}
  }
 }
 best.fes=fes;return best;
}
}
