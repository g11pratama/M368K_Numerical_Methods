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
ila_state conj_grad(const Matrix& A, const Vector& b, Vector& x,
	int& maxIter, double tol);
ila_state prec_conj_grad(ila_state (*prec)(const Matrix&, const Vector&, Vector&, double),
	const Matrix&A, const Vector& b, Vector& x, double omega, int& maxIter,
	double tol);

ila_state jac_prec(const Matrix& A, const Vector& r, Vector& z, double omega);

ila_state ssor_prec(const Matrix& A, const Vector& r, Vector& z, double omega);
#endif
