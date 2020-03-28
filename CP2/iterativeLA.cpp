#include <iostream>
using namespace std;

#include "iterativeLA.h"
#include "matrix.h"
#include <cmath>

//#define MONITOR  // output history of l2-error
//#define MONITOR2 // output history of solution and l2-error

//#define PLOT     // output history of l2-error of residual in Matlab format
#ifdef PLOT
#include "matlabPlot.h"
#endif

ila_state jacobi(const Matrix& A, const Vector& b, Vector& x,
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
#ifdef PLOT
	Vector res(maxIter);
	Vector u(n);
	ofstream fout("Jacobi_plot.m");
#endif

	for (int iter = 0; iter < maxIter; iter++) {

		// Get new x
		for (int i = 0; i < n; i++) {
			double sum = 0;
			for (int j = 0; j < n; j++) {
				if (j == i) continue;
				sum += A(i, j)*xOld(j);
			}
			x(i) = (-sum + b(i)) / A(i, i);
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
#ifdef PLOT
		matVecMult(A, x, u);
		u -= b;
		res(iter) = sqrt(scDot(u, u));

#endif

		if (l2error <= tol) {
			maxIter = iter + 1;
#ifdef PLOT
			res.resize(maxIter);
			matlabPlot(fout, res, "Convergence of Jacobi method",
				"iter", "L2-norm of residual");
			fout.close();
#endif
			return ILA_SUCCESS;
		}
		xOld = x;
	}
#ifdef PLOT
	matlabPlot(fout, res, "Convergence of Jacobi method",
		"iter", "L2-norm of residual");
	fout.close();
#endif 
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
#ifdef PLOT
	Vector res(maxIter);
	Vector u(n);
	ofstream fout("Gauss_Seidel_plot.m");
#endif

	for (int iter = 0; iter < maxIter; iter++) {

		// Get new x
		for (int i = 0; i < n; i++) {
			double sumUpper = 0;
			double sumLower = 0;
			for (int j = i + 1; j < n; j++)
				sumUpper += A(i, j)*xOld(j);
			for (int j = 0; j < i; j++)
				sumLower += A(i, j)*x(j);
			x(i) = (-sumLower - sumUpper + b(i)) / A(i, i);
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
#ifdef PLOT
		matVecMult(A, x, u);
		u -= b;
		res(iter) = sqrt(scDot(u, u));
#endif
		if (l2error <= tol) {
			maxIter = iter + 1;
#ifdef PLOT
			res.resize(maxIter);
			matlabPlot(fout, res, "Convergence of Jacobi method",
				"iter", "L2-norm of residual");
			fout.close();
#endif
			return ILA_SUCCESS;
		}
		xOld = x;
	}
#ifdef PLOT
	matlabPlot(fout, res, "Convergence of Jacobi method",
		"iter", "L2-norm of residual");
	fout.close();
#endif 

	return ILA_WONT_STOP;
}

ila_state sor(const Matrix& A, const Vector& b, Vector& x,
	int& maxIter, double tol, double omega) {

	// CHECK DATA
	if (!(0 < omega && omega < 2)) return ILA_BAD_DATA;

	int n = A.n(0);
	if (A.n(1) != n || b.n() != n || x.n() != n) return ILA_BAD_DATA;
	if (tol <= 0) return ILA_BAD_DATA;
	if (maxIter <= 0) maxIter = 1;

	for (int i = 0; i < n; i++) {
		if (A(i, i) == 0) return ILA_BAD_DIAGONAL;
	}

	// APPLY JACOBI

	Vector xOld(x);
#ifdef PLOT
	Vector res(maxIter);
	Vector u(n);
	ofstream fout("SOR_plot.m");
#endif

	for (int iter = 0; iter < maxIter; iter++) {

		// Get new x
		for (int i = 0; i < n; i++) {
			double sumUpper = 0;
			double sumLower = 0;
			for (int j = i + 1; j < n; j++)
				sumUpper += A(i, j)*xOld(j);
			for (int j = 0; j < i; j++)
				sumLower += A(i, j)*x(j);
			x(i) = (1 - omega)*xOld(i) + omega * (-sumLower - sumUpper + b(i)) / A(i, i);
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
#ifdef PLOT
		matVecMult(A, x, u);
		u -= b;
		res(iter) = sqrt(scDot(u, u));
#endif
		if (l2error <= tol) {
			maxIter = iter + 1;
#ifdef PLOT
			res.resize(maxIter);
			matlabPlot(fout, res, "Convergence of Jacobi method",
				"iter", "L2-norm of residual");
			fout.close();
#endif
			return ILA_SUCCESS;
		}
		xOld = x;
	}
#ifdef PLOT
	matlabPlot(fout, res, "Convergence of Jacobi method",
		"iter", "L2-norm of residual");
	fout.close();
#endif 

	return ILA_WONT_STOP;
}


ila_state prec_conj_grad(ila_state(*prec)(const Matrix&, const Vector&, Vector&, double),
	const Matrix&A, const Vector& b, Vector& x, double omega, int& maxIter,
	double tol) {

	// CHECK DATA

	int n = A.n(0);

	if (A.n(1) != n || b.n() != n || x.n() != n) return ILA_BAD_DATA;
	if (tol <= 0) return ILA_BAD_DATA;
	if (maxIter <= 0) maxIter = 1;

	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			if (A(i, j) != A(j, i)) return ILA_NOT_SYMMETRIC;
		}
	}

	// INITIALIZE CONJUGATE GRADIENTS

	// Set initial residual r = b - Ax
	Vector r(n);

	matVecMult(A, x, r);
	r -= b; r *= (-1);

	Vector z(n);
	prec(A, r, z, omega);

	double alpha = scDot(r, z);

#ifdef MONITOR
	cout << endl << "Iter 0, res l2-error " << sqrt(alpha) << endl;
#endif    
#ifdef MONITOR2
	cout << endl << "Iter 0, guess: " << x
		<< ", res l2-error " << sqrt(alpha) << endl;
#endif    
#ifdef PLOT
	Vector res(maxIter);
	ofstream fout("PCG_plot.m");
	res(0) = sqrt(alpha);
#endif


	// Set initial search direction d = r
	Vector d(z); // Creates d and sets d = r

	double tolSq = tol * tol;

	// CONJUGATE GRADIENT LOOP

	for (int iter = 0; iter < maxIter; iter++) {

		if (scDot(d, d) <= tolSq) {
			maxIter = iter;
#ifdef PLOT
			res.resize(maxIter + 1);
			matlabPlot(fout, res, "Convergence of Conjugate Gradient method",
				"iter", "L2-norm of residual");
			fout.close();
#endif
			return ILA_SUCCESS;
		}

		// Set u = Ad
		Vector u(n);

		matVecMult(A, d, u);

		// Update x = x + td and r = r - tu
		double t = alpha / scDot(d, u);

		for (int i = 0; i < n; i++) {
			x(i) += t * d(i);
			r(i) -= t * u(i);
		}
		prec(A, r, z, omega);

		// Get new search direction d = r + s*d;
		double beta = scDot(r, z);

#ifdef MONITOR
		cout << "Iter " << iter + 1 << ", res l2-error " << sqrt(beta) << endl;
#endif    
#ifdef MONITOR2
		cout << "Iter " << iter + 1 << ", approx. solution: "
			<< x << ", res l2-error " << sqrt(beta) << endl;
#endif    
#ifdef PLOT
		res(iter + 1) = sqrt(beta);
#endif

		if (beta <= tolSq) {
			maxIter = iter + 1;
#ifdef PLOT
			res.resize(maxIter + 1);
			matlabPlot(fout, res, "Convergence of Conjugate Gradient method",
				"iter", "L2-norm of residual");
			fout.close();
#endif
			return ILA_SUCCESS;
		}

		double s = beta / alpha;

		for (int i = 0; i < n; i++) {
			d(i) = z(i) + s * d(i);
		}

		alpha = beta;
	}

	return ILA_WONT_STOP;
}



ila_state jac_prec(const Matrix& A, const Vector& r, Vector& z, double omega) {
	for (int i = 0; i < z.n(); i++) {
		z(i) = 1 / A(i, i) * r(i);
	}
	return ILA_SUCCESS;
};


ila_state ssor_prec(const Matrix& A, const Vector& r, Vector& z, double omega) {
	Vector temp_vect(z.n());
	for (int i = 0; i < z.n(); i++) {
		double run_sum = r(i);
		for (int j = 0; j < i; j++) {
			run_sum -= omega * A(i, j)*temp_vect(j) / A(j, j);
		}
		temp_vect(i) = run_sum;
	}

	for (int i = z.n() - 1; i >= 0; i--) {
		double run_sum = temp_vect(i);
		for (int j = i + 1; j < z.n(); j++) {			
			run_sum -= omega * z(j)*A(i, j);
		}
		z(i) = run_sum / A(i, i);
	}
	return ILA_SUCCESS;

};


ila_state conj_grad(const Matrix& A, const Vector& b, Vector& x,
	int& maxIter, double tol) {

	// CHECK DATA

	int n = A.n(0);

	if (A.n(1) != n || b.n() != n || x.n() != n) return ILA_BAD_DATA;
	if (tol <= 0) return ILA_BAD_DATA;
	if (maxIter <= 0) maxIter = 1;

	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			if (A(i, j) != A(j, i)) return ILA_NOT_SYMMETRIC;
		}
	}

	// INITIALIZE CONJUGATE GRADIENTS

	// Set initial residual r = b - Ax
	Vector r(n);

	matVecMult(A, x, r);
	r -= b; r *= (-1);

	double alpha = scDot(r, r);

#ifdef MONITOR
	cout << endl << "Iter 0, res l2-error " << sqrt(alpha) << endl;
#endif    
#ifdef MONITOR2
	cout << endl << "Iter 0, guess: " << x
		<< ", res l2-error " << sqrt(alpha) << endl;
#endif    
#ifdef PLOT
	Vector res(maxIter);
	ofstream fout("CG_plot.m");
	res(0) = sqrt(alpha);
#endif

	// Set initial search direction d = r
	Vector d(r); // Creates d and sets d = r

	double tolSq = tol * tol;

	// CONJUGATE GRADIENT LOOP

	for (int iter = 0; iter < maxIter; iter++) {

		if (scDot(d, d) <= tolSq) {
			maxIter = iter;
#ifdef PLOT
			res.resize(maxIter + 1);
			matlabPlot(fout, res, "Convergence of Conjugate Gradient method",
				"iter", "L2-norm of residual");
			fout.close();
#endif
			return ILA_SUCCESS;
		}

		// Set u = Ad
		Vector u(n);

		matVecMult(A, d, u);

		// Update x = x + td and r = r - tu
		double t = alpha / scDot(d, u);

		for (int i = 0; i < n; i++) {
			x(i) += t * d(i);
			r(i) -= t * u(i);
		}

		// Get new search direction d = r + s*d;
		double beta = scDot(r, r);

#ifdef MONITOR
		cout << "Iter " << iter + 1 << ", res l2-error " << sqrt(beta) << endl;
#endif    
#ifdef MONITOR2
		cout << "Iter " << iter + 1 << ", approx. solution: "
			<< x << ", res l2-error " << sqrt(beta) << endl;
#endif    
#ifdef PLOT
		res(iter + 1) = sqrt(beta);
#endif

		if (beta <= tolSq) {
			maxIter = iter + 1;
#ifdef PLOT
			res.resize(maxIter + 1);
			matlabPlot(fout, res, "Convergence of Conjugate Gradient method",
				"iter", "L2-norm of residual");
			fout.close();
#endif
			return ILA_SUCCESS;
		}

		double s = beta / alpha;

		for (int i = 0; i < n; i++) {
			d(i) = r(i) + s * d(i);
		}

		alpha = beta;
	}

	return ILA_WONT_STOP;
}