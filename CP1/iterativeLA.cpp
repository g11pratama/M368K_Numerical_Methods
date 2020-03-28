#include <iostream>
using namespace std;

#include "iterativeLA.h"
#include "matrix.h"
//#include <cmath>

//#define MONITOR  // output history of l2-error
#define MONITOR2 // output history of solution and l2-error

ila_state jacobi(const Matrix& A, const Vector& b, Vector& x,
	     int& maxIter, double tol) {

  // CHECK DATA

  int n = A.n(0);
  if(A.n(1) != n || b.n() != n || x.n() != n) return ILA_BAD_DATA;
  if(tol <= 0) return ILA_BAD_DATA;
  if(maxIter <= 0) maxIter = 1;

  for(int i=0; i<n; i++) {
    if(A(i,i) == 0) return ILA_BAD_DIAGONAL;
  }

  // APPLY JACOBI

  Vector xOld(x);

  for(int iter=0; iter<maxIter; iter++) {

    // Get new x
    for(int i=0; i<n; i++) {
      double sum = 0;
      for(int j=0; j<n; j++) {
		if(j==i) continue;
		sum += A(i,j)*xOld(j);
      }
      x(i) = ( -sum + b(i) ) / A(i,i);
    }

    // Check error tolerance
    xOld -= x;
    double l2error = l2norm(xOld) / (l2norm(x)+1);
#ifdef MONITOR
    cout << "Iter " << iter+1 << ", l2-error " << l2error << endl;
#endif    
#ifdef MONITOR2
    cout << "Iter " << iter+1 << ", approx. solution: " 
         << x << ", l2-error " << l2error << endl;
#endif    
    if( l2error <= tol) {
      maxIter = iter+1;
      return ILA_SUCCESS;
    }
    xOld = x;
  }

  return ILA_WONT_STOP;
}


ila_state gaussSeidel(const Matrix& A, const Vector& b, Vector& x,
	int& maxIter, double tol) {

	// CHECK DATA

	int n = A.n(0);
	if (A.n(1) != n || b.n() != n || x.n() != n) return ILA_BAD_DATA;
	if (tol <= 0) return ILA_BAD_DATA;
	if (maxIter <= 0) maxIter = 1;

	for (int i = 0; i < n; i++) {
		if (A(i, i) == 0) return ILA_BAD_DIAGONAL;
	}

	// APPLY JACOBI

	Vector xOld(x);

	for (int iter = 0; iter < maxIter; iter++) {

		// Get new x
		for (int i = 0; i < n; i++) {
			double sumUpper = 0;
			double sumLower = 0;
			for (int j = i + 1; j < n; j++)				
				sumUpper += A(i, j)*xOld(j);
			for (int j = 0; j < i; j++)
				sumLower += A(i, j)*x(j);
			x(i) = (-sumLower -sumUpper + b(i)) / A(i, i);
		}

		// Check error tolerance
		xOld -= x;
		double l2error = l2norm(xOld) / (l2norm(x) + 1);
#ifdef MONITOR
		cout << "Iter " << iter + 1 << ", l2-error " << l2error << endl;
#endif    
#ifdef MONITOR2
		cout << "Iter " << iter + 1 << ", approx. solution: "
			<< x << ", l2-error " << l2error << endl;
#endif    
		if (l2error <= tol) {
			maxIter = iter + 1;
			return ILA_SUCCESS;
		}
		xOld = x;
	}

	return ILA_WONT_STOP;
}

ila_state sor(const Matrix& A, const Vector& b, Vector& x,
	int& maxIter, double tol, double omega) {

	// CHECK DATA
	if (!(0 < omega < 2)) return ILA_BAD_DATA;

	int n = A.n(0);
	if (A.n(1) != n || b.n() != n || x.n() != n) return ILA_BAD_DATA;
	if (tol <= 0) return ILA_BAD_DATA;
	if (maxIter <= 0) maxIter = 1;

	for (int i = 0; i < n; i++) {
		if (A(i, i) == 0) return ILA_BAD_DIAGONAL;
	}

	// APPLY JACOBI

	Vector xOld(x);

	for (int iter = 0; iter < maxIter; iter++) {

		// Get new x
		for (int i = 0; i < n; i++) {
			double sumUpper = 0;
			double sumLower = 0;
			for (int j = i + 1; j < n; j++)
				sumUpper += A(i, j)*xOld(j);
			for (int j = 0; j < i; j++)
				sumLower += A(i, j)*x(j);
			x(i) = (1 - omega)*xOld(i) + omega*(-sumLower - sumUpper + b(i)) / A(i, i);
		}

		// Check error tolerance
		xOld -= x;
		double l2error = l2norm(xOld) / (l2norm(x) + 1);
#ifdef MONITOR
		cout << "Iter " << iter + 1 << ", l2-error " << l2error << endl;
#endif    
#ifdef MONITOR2
		cout << "Iter " << iter + 1 << ", approx. solution: "
			<< x << ", l2-error " << l2error << endl;
#endif    
		if (l2error <= tol) {
			maxIter = iter + 1;
			return ILA_SUCCESS;
		}
		xOld = x;
	}

	return ILA_WONT_STOP;
}












