#include <iostream>
#include <cmath>
#include <string>
#include <map>
#include <fstream>
using namespace std;

#include "iterativeLA.h"

int main() {
  int n = 0;
  string useFile;
  cout << "Use file (Y/n): " << flush;
  cin >> useFile;
  string getFileName;
  string bFileName;


  if (useFile == "Y") {	  
	  cout << "Enter matrix file name: " << flush;
	  cin.ignore();
	  getline(cin, getFileName);
	  cout << "Enter b vector file name: " << flush;
	  getline(cin, bFileName);

	  ifstream inFile(getFileName);
	  string line;

	  while (getline(inFile, line))
		  n += 1;
	  cout << "n = " << n << endl;
	  inFile.close();	  
  }
  else {
	  cout << "Enter size: " << flush;
	  cin >> n; 
  }

  Matrix A(n, n);
  Vector x(n);
  Vector b(n);

  if (useFile == "Y") {
	  ifstream inFile(getFileName);	  
	  while (inFile >> A);
	  inFile.close();

	  ifstream bFile(bFileName);
	  while (bFile >> b);
	  bFile.close();

  }
  else {
	  cout << "Enter A by rows: " << endl;
	  cin >> A;

	  cout << "Enter b: " << flush;
	  cin >> b;
  }

  x = 0.;

  enum method {JACOBI = 1, GAUSS, SOR, CG, PREC_JAC_CG, PREC_SSOR_CG};
  int userChoice;

  cout << "Choose iterative method (Enter 1 for Jacobi, 2 for Gauss-Seidel, 3 for SOR, "
           "4 for Conjugate Gradient, 5 for Jacobi PCG, 6 for SSOR PCG): " << flush;
  cin >> userChoice;

  int maxIter;
  double tolerance;
  double omega;
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
		cout << "Enter omega value (0 < omega < 2): " << flush;
		cin >> omega;
		s = sor(A, b, x, maxIter, tolerance, omega);
		break;
	case CG:
		cout << "\nConjugate Gradient chosen" << endl;
		s = conj_grad(A, b, x, maxIter, tolerance);
	case PREC_JAC_CG:
		cout << "\nJacobi PCG chosen" << endl;
		s = prec_conj_grad(jac_prec, A, b, x, 1, maxIter, tolerance);
		break;
	case PREC_SSOR_CG:
		cout << "\nSSOR PCG chosen" << endl;
		cout << "Enter omega value (0 < omega < 2): " << flush;
		cin >> omega;
		s = prec_conj_grad(ssor_prec, A, b, x, omega, maxIter, tolerance);
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

