#include <iostream>
#include <cmath>
#include <string>
#include <map>
#include <fstream>
using namespace std;

#include "iterativeNL.h"

static int prec = 12;

void test_fn(const Vector& x, Vector& y){
  const double u = x(0);
  const double v = x(1);
  y(0) = 6*pow(u, 3) + u*v - 3*pow(v, 3) - 4;
  y(1) = pow(u, 2) - 18*u*pow(v,2) + 16*pow(v, 3) + 1;
}

void test_jac_fn(const Vector& x, Matrix& df) {
  const double u = x(0);
  const double v = x(1);
  df(0,0) = 18*pow(u,2) + v;
  df(0,1) = u - 9*pow(v,2);
  df(1,0) = 2*u - 18*pow(v,2);
  df(1,1) = -36*u*v + 48*pow(v,2);
}

void fn1(const Vector& x, Vector& y) {
  const double u = x(0);
  const double v = x(1);
  y(0) = pow(u,2) + pow(v,2) - 1;
  y(1) = pow((u-1), 2) + pow(v,2) - 1;
}

void dfn1(const Vector& x, Matrix& df){
  const double u = x(0);
  const double v = x(1);
  df(0,0) = 2*u;
  df(0,1) = 2*v;
  df(1,0) = 2*(u-1);
  df(1,1) = 2*v;
}

void fn2(const Vector& x, Vector& y) {
  const double u = x(0);
  const double v = x(1);
  y(0) = pow(u,2) + 4*pow(v,2) - 4;
  y(1) = 4*pow(u, 2) + pow(v,2) - 4;
}

void dfn2(const Vector& x, Matrix& df) {
  const double u = x(0);
  const double v = x(1);
  df(0,0) = 2*u;
  df(0,1) = 8*v;
  df(1,0) = 8*u;
  df(1,1) = 2*v;
}

void fn3(const Vector& x, Vector& y) {
  const double u = x(0);
  const double v = x(1);
  y(0) = pow(u,2) - 4*pow(v,2) - 4;
  y(1) = pow((u-1), 2) + pow(v,2) - 4;
}

void dfn3(const Vector&x, Matrix& df) {
  const double u = x(0);
  const double v = x(1);
  df(0,0) = 2*u;
  df(0,1) = -8*v;
  df(1,0) = 2*(u-1);
  df(1,1) = 2*v;
}

int main() {
  enum method {FIXED_PT = 1, NEWTON, BROYDEN2};
  int userChoice;
  int maxIter;
  double tolerance;

  Vector x(2);
  Matrix df(2, 2);

  cout << "Test case:" << endl;
  cout << "Choose non-linear iterative method (Enter 1 for fixed-point, 2 for Newton, "
          "3 for Broyden2): "
    << flush;
  cin >> userChoice;

  cout << "Enter maxIter and tolerance: " << flush;
  cin >> maxIter >> tolerance;

  cout << "Enter initial guess: " << flush;
  cin >> x;
  cout << endl;

  inl_state s;
  cout.precision(prec);

  typedef void(*fnPointers)(const Vector&, Vector&);
  typedef void(*dfnPointers)(const Vector&, Matrix&);
  fnPointers fns[] = {fn1, fn2, fn3};
  dfnPointers dfns[] = {dfn1, dfn2, dfn3};

  for (int i = 0; i < 3; i++) {
    int iter = maxIter;
    cout << "Function " << i + 1 << endl;
    switch(userChoice) {      
      case FIXED_PT:
        cout << "Fixed-point chosen." << endl;
        s = fixedpt(fns[i], x, tolerance, iter);
        break;
      case NEWTON:
        cout << "Newton chosen." << endl;
        s = newton(fns[i], dfns[i], x, tolerance, iter);
        break;
      case BROYDEN2:
        cout << "Broyden2 chosen." << endl;
        cout << "Choose B0 (1 for I, 2 for (DF(x))^-1): " << flush;
        int init;
        cin >> init;
        s = broyden2(fns[i], dfns[i], x, tolerance, iter, init);
    }

    switch (s){
      case INL_WONT_STOP:
        cout << "ERROR: Exceeded maximum number of iterations." << endl;
        break;
      default:
        cout << "ERROR: Unspecified." << endl;
        break;
      case INL_SUCCESS:
        cout << "The solution is:" << endl;
        cout << x << endl;
        cout << "The number of iterations is: " << iter << endl;
    }
    cout << endl;
  }
  return 0;
}