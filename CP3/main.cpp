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

void test_jac_fn(const Vector&x, Matrix& df) {
  const double u = x(0);
  const double v = x(1);
  df(0,0) = 18*pow(u,2) + v;
  df(0,1) = u - 9*pow(v,2);
  df(1,0) = 2*u - 18*pow(v,2);
  df(1,1) = -36*u*v + 48*pow(v,2);
}

int main() {
  enum method {FIXED_PT = 1, NEWTON};
  int userChoice;
  int maxIter;
  double tolerance;

  Vector x(2);
  Matrix df(2, 2);

  cout << "Test case:" << endl;
  cout << "Choose non-linear iterative method (Enter 1 for fixed-point, 2 for Newton): "
    << flush;
  cin >> userChoice;

  cout << "Enter maxIter and tolerance: " << flush;
  cin >> maxIter >> tolerance;

  cout << "Enter initial guess: " << flush;
  cin >> x;
  cout << endl;

  inl_state s;
  cout.precision(prec);

  switch(userChoice) {
    case FIXED_PT:
      cout << "Fixed-point chosen." << endl;
      s = fixedpt(test_fn, x, tolerance, maxIter);
    case NEWTON:
      cout << "Newton chosen." << endl;
      s = newton(test_fn, test_jac_fn, x, tolerance, maxIter);
  }

  switch (s){
    case INL_WONT_STOP:
      cout << "ERROR: Exceeded maximum number of iterations." << endl;
      return 1;
    default:
      cout << "ERROR: Unspecified." << endl;
      return 1;
    case INL_SUCCESS:
      cout << "The solution is:" << endl;
      cout << x << endl;
      cout << "The number of iterations is: " << maxIter << endl;
      return 0;
  }
}