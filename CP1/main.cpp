#include <iostream>
#include <cmath>
#include <string>
#include <map>
using namespace std;

#include "iterativeLA.h"

int main() {
  int n;
  cout << "Enter size: " << flush;
  cin >> n;
  
  Matrix A(n,n);
  Vector x(n);
  Vector b(n);

  x = 0.;

  cout << "Enter A by rows: " << endl;
  cin >> A;

  cout << "Enter b: " << flush;
  cin >> b;

  enum method {JACOBI = 1, GAUSS, SOR};
  int userChoice;

  cout << "Choose iterative method (Enter 1 for Jacobi, 2 for Gauss-Seidel, 3 for SOR): " << flush;
  cin >> userChoice;

  int maxIter;
  double tolerance;
  cout << "Enter maxIter and tolerance: " << flush;
  cin >> maxIter >> tolerance;

  ila_state s;

  switch (userChoice) {
	case JACOBI:
		cout << "\nJacobi chosen" << endl;
		s = jacobi(A, b, x, maxIter, tolerance);
		break;
	case GAUSS:
		cout << "\nGauss-Seidel chosen" << endl;
		s = gaussSeidel(A, b, x, maxIter, tolerance);
		break;
	case SOR:
		cout << "\nSOR chosen" << endl;
		double omega;
		cout << "Enter omega value (0 < omega < 2): " << flush;
		cin >> omega;
		s = sor(A, b, x, maxIter, tolerance, omega);
  }

  map<int, string> booleanMap = { {0, "False"}, {1, "True"} };

  cout << "Matrix strictly diagonally dominant: " << booleanMap[isStrictDiagDominant(A)] << endl;

  int prec = (int) (log10(1.0/tolerance));
  cout.precision(prec);

  switch(s) {
  case ILA_WONT_STOP:
    cout << "ERROR: Exceeded maximum number of iterations." << endl;
    return 1;
  case ILA_BAD_DIAGONAL:
    cout << "ERROR: A diagonal entry of A was 0." << endl;
    return 1;
  default:
    cout << "ERROR: Unspecified." << endl;
    return 1;
  case ILA_SUCCESS:
    cout << "The solution is:" << endl;
    cout << x << endl;

    Vector y(n);
    matVecMult(A,x,y);
    y -= b;
    cout << "The number of iterations is: " << maxIter << endl;
    cout << "The max-norm of residual is: " << maxNorm(y) << endl;
    cout << "The residual is: " << endl;
    cout << y << endl;
    return 0;
  }
}

