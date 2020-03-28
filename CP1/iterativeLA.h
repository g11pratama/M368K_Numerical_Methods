#ifndef IterativeLA_Included
#define IterativeLA_Included

#include "matrix.h"

enum ila_state {ILA_SUCCESS, ILA_WONT_STOP, ILA_BAD_DIAGONAL, ILA_NOT_SYMMETRIC, ILA_BAD_DATA};

ila_state jacobi(const Matrix& A, const Vector& b, Vector& x,
	     int& maxIter, double tol);
ila_state gaussSeidel(const Matrix& A, const Vector& b, Vector& x,
	int& maxIter, double tol);
ila_state sor(const Matrix& A, const Vector& b, Vector& x,
	int& maxIter, double tol, double omega);

#endif
