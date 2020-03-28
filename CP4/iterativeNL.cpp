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

inl_state broyden2(void f(const Vector&, Vector&), void df(const Vector&, Matrix&),
        Vector& x, double tol, int& iter, int init) {
  if (iter < 1) iter = 1;
  if (tol <= 0 || (init < 1 && init > 2)) return INL_BAD_DATA;

  int n = x.n();
  Vector y(n);
  Matrix b(n,n);

  if (init == 2){
    Matrix a(n,n);
    df(x, a);
    ge_state s = getInverse(a, b);
    if (s != GE_SUCCESS) return INL_BAD_DATA;
  }
  else {
    b.identity();
  }

  double error;
  int maxIter = iter;
  Vector F(n);
  Vector delF(n);
  Vector g(n);
  Matrix outMat(n,n);
  Matrix m(n,n);
  double c;

  for (iter = 0; iter <= maxIter; iter++) {
    f(x, F);
    matVecMult(b, F, y);
    y *= -1;
    x += y;
    error = maxNorm(y);

#ifdef MONITOR
    cout << "Iter " << iter << ": x= " << x << ", err = " << error << endl;
#endif
    if (error <= tol) return INL_SUCCESS;

    f(x, delF);
    delF -= F;
    matVecMult(b, delF, g);
    c = scDot(y, g);
    g *= -1;
    g += y;
    outMat.outerProduct(g, y);
    matMatMult(outMat, b, m);
    m /= c;
    b += m;
  }

  return INL_WONT_STOP;
}