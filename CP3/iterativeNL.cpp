#include <iostream>
#include <cmath>
using namespace std;

#include "iterativeNL.h"
#include "gaussElim.h"

#define MONITOR 


inl_state fixedpt(void g(const Vector&,Vector&), Vector& x,
	      double tol, int& iter) {
  if(iter < 1) iter = 1;
  if(tol <= 0) return INL_BAD_DATA;

  int maxIter = iter;
  Vector y(x);
  double error;
  for(iter = 1; iter <= maxIter; iter++) {
    g(y,x);      // x = g(y) is the new guess
    y -= x;      // y is now the change to x
    error = maxNorm(y);
#ifdef MONITOR
    cout << "Iter " << iter << ": x= " << x << ", err = " << error << endl;
#endif    
    if (error <= tol) return INL_SUCCESS;
    y = x;
  }
  return INL_WONT_STOP;
}

inl_state newton(void f(const Vector&,Vector&), void df(const Vector&,Matrix&), Vector& x, 
        double tol, int& iter) {
  if (iter < 1) iter = 1;
  if (tol <= 0) return INL_BAD_DATA;

  int n = x.n();
  Vector y(n);
  Matrix jac(n, n);
  double error;
  int maxIter = iter;

  for (iter = 1; iter <= maxIter; iter++) {
    df(x, jac);
    f(x, y);

    Permutation P(n);
    ge_state elim = solve(jac, P, y);
    if (elim != GE_SUCCESS) return INL_BAD_ITERATE;

    x -= y;
    error = maxNorm(y);
#ifdef MONITOR
    cout << "Iter " << iter << ": x= " << x << ", err = " << error << endl;
#endif
  if (error <= tol) return INL_SUCCESS;
  }
  return INL_WONT_STOP;
}