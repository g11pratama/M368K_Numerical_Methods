#include <math.h>
#include "matrix.h"

// Vector /////////////////////////////////////////////////////////////////////

Vector::Vector(const Vector& v) {
  size = v.size;
  array = new double[size];
  for(int i=0; i<size; i++) array[i] = v.array[i];
}

Vector& Vector::operator=(const Vector& v) {
  if(size != v.size)
    std::cerr << "Sizes differ in Vector::operator=(const Vector&)";
  for(int i=0; i<size; i++) array[i] = v.array[i];
  return *this;
}

Vector& Vector::operator=(double x) {
  for(int i=0; i<size; i++) array[i] = x;
  return *this;
}

// NOTE: this is an unsafe assignment, as the size of the array that will be copied cannot be checked.
Vector& Vector::operator=(double* a){
  for(int i=0; i<size; i++) array[i] = a[i];
  return *this;
}

// NOTE: this is safe
Vector& Vector::operator=(std::vector<double>& a){
  if(size != a.size())
    std::cerr << "Sizes differ in Vector::operator=(std::vector<double>&)";

  for(int i=0; i<size; i++) array[i] = a[i];
  return *this;
}

Vector& Vector::operator+=(const Vector& v) {
  if(size != v.size)
    std::cerr << "Sizes differ in Vector::operator+=(const Vector&)";
  for(int i=0; i<size; i++) array[i] += v.array[i];
  return *this;
}

Vector& Vector::operator-=(const Vector& v) {
  if(size != v.size)
    std::cerr << "Sizes differ in Vector::operator-=(const Vector&)";
  for(int i=0; i<size; i++) array[i] -= v.array[i];
  return *this;
}

Vector& Vector::operator*=(double x) {
  for(int i=0; i<size; i++) array[i] *= x;
  return *this;

}

Vector& Vector::operator/=(double x) {
  for(int i=0; i<size; i++) array[i] /= x;
  return *this;
}

double Vector::operator*(const Vector& v) const {
  int n = size;
  if(size > v.size) n = v.size;

  double dot = 0;
  for(int i=0; i<n; i++) dot += array[i]*v.array[i];
  return dot;
}

void Vector::resize(const int n) {
  if(n<1)
    std::cerr << "Size too small in Vector::resize(const int)";
  if (n!=size) {
    int m = (n<size) ? n : size;
    Vector b(m);
    for(int i=0;i<m;i++) b(i) = array[i];
    delete[] array;
    size = n;
    array = new double[size];
    for(int i=0;i<m;i++) array[i] = b(i);
    for(int i=m;i<size;i++) array[i] = 0; 
  }
}


// Matrix /////////////////////////////////////////////////////////////////////

Matrix::Matrix(const Matrix& m) {
  size[0] = m.size[0];
  size[1] = m.size[1];
  array = new double[size[0]*size[1]];
  for(int i=0; i<size[0]*size[1]; i++) array[i] = m.array[i];
}

bool Matrix::diag_dominant() {
  if (size[0] != size[1]) {std::cout<< "Size" << std::endl; return false;}
  for (int i=0; i<size[0]; i++) {
    for (int j=0; j <size[0]; j++){
      if ((j*size[0] + j) == (i*size[0])){continue;}
      if (array[size[0]*i + i] < array[j*size[0] + i]) { return false;}
    }
  }
  return true;
}

Matrix& Matrix::operator=(const Matrix& m) {
  if(size[0] != m.size[0] || size[1] != m.size[1])
    std::cerr << "Sizes differ in Matrix::operator=(const Matrix&)";
  for(int i=0; i<size[0]*size[1]; i++) array[i] = m.array[i];
  return *this;
}

Matrix& Matrix::operator=(double x) {
  for(int i=0; i<size[0]*size[1]; i++) array[i] = x;
  return *this;
}

Matrix& Matrix::operator+=(const Matrix& m) {
  if(size[0] != m.size[0] || size[1] != m.size[1])
    std::cerr << "Sizes differ in Matrix::operator+=(const Matrix&)";
  for(int i=0; i<size[0]*size[1]; i++) array[i] += m.array[i];
  return *this;
}

Matrix& Matrix::operator-=(const Matrix& m) {
  if(size[0] != m.size[0] || size[1] != m.size[1])
    std::cerr << "Sizes differ in Matrix::operator-=(const Matrix&)";
  for(int i=0; i<size[0]*size[1]; i++) array[i] -= m.array[i];
  return *this;
}

Matrix& Matrix::operator*=(double x) {
  for(int i=0; i<size[0]*size[1]; i++) array[i] *= x;
  return *this;
}

Matrix& Matrix::operator/=(double x) {
  for(int i=0; i<size[0]*size[1]; i++) array[i] /= x;
  return *this;
}

Vector Matrix::getCol(int j) const {
  Vector x(n(0));
  for (int i = 0; i < n(0); i++) x(i) = this->operator()(i, j);
  return x;
}

void Matrix::setCol(const Vector& x, int j) {
  if (x.n() != size[0])
    std::cerr << "Sizes differ in Matrix::setCol(const Vector& x, int j)";
  for (int i = 0; i < size[0]; i++) this->operator()(i,j) = x(i);
}

void Matrix::identity() {
  if (size[0] != size[1])
    std::cerr << "Matrix not n x n in Matrix::identity()";
  for (int i = 0; i < size[0]; i++)
    for (int j = 0; j < size[0]; j++) this->operator()(i,j) = (i==j)? 1: 0;    
}

void Matrix::outerProduct(const Vector& x, const Vector& y) {
  if (x.n() != size[0] || y.n() != size[1])
    std::cerr << "Matrix not the proper size in Matrix::outerProduct(const Vector& x, const Vector& y)";
  for (int i = 0; i < x.n(); i++)
    for (int j = 0; j < y.n(); j++) this->operator()(i,j) = x(i)*y(j);
}

// Permutation ////////////////////////////////////////////////////////////////

void Permutation::identity() {
  for(int i=0; i<size; i++) array[i] = i;
  my_parity = 1;
}

void Permutation::swap(int i, int j) {
  int k = array[i];
  array[i] = array[j];
  array[j] = k;
  my_parity *= -1;
}

void Permutation::permute(Vector& b) const {
  if(size > b.n()) return;

  Vector c(b.n());

  for(int i=0; i<size; i++) c(i) = b(array[i]);
  for(int i=0; i<size; i++) b(i) = c(i);
}

// Miscellaneous //////////////////////////////////////////////////////////////

std::ostream& operator<< (std::ostream& os, const Vector& v) {
  for(int i=0; i<v.n(); i++) {
    os << v(i);
    if((i+1)%5) {
      os << "  ";
    } else {
      os << "\n";
    }
  }
  return os;
}

std::ostream& operator<< (std::ostream& os, const Matrix& m) {
  for(int i=0; i<m.n(0); i++) {
    for(int j=0; j<m.n(1); j++) {
      os << m(i,j);
      if((j+1)%5) {
	os << "  ";
      } else {
	os << "\n";
      }
    }
    os << "\n";
  }
  return os;
}

std::istream& operator>> (std::istream& is, Vector& v) {
  for(int i=0; i<v.n(); i++) is >> v(i);
  return is;
}

std::istream& operator>> (std::istream& is, Matrix& m) {
  for(int i=0; i<m.n(0); i++)
  for(int j=0; j<m.n(1); j++) {
    is >> m(i,j);
  }
  return is;
}

double l2norm(const Vector& v) {
  double norm = 0;
  for(int i=0; i<v.n(); i++) norm += v(i)*v(i);
  return sqrt(norm);
}
  
double maxNorm(const Vector& v) {
  double norm = 0;
  for(int i=0; i<v.n(); i++) {
    double a = fabs(v(i));
    if(norm < a) norm = a;
  }
  return norm;
}

double maxNorm(const Matrix& m) {
  double norm = 0;
  for(int i=0; i<m.n(0); i++) {
    double sum=0;
    for(int j=0; j<m.n(1); j++) sum += fabs(m(i,j));
    if(norm < sum) norm = sum;
  }
  return norm;
}

double scDot(const Vector& v1, const Vector& v2) { return v1*v2; }

int matVecMult(const Matrix& M, const Vector& x, Vector& y) {
  if(M.n(0) != y.n() || M.n(1) != x.n()) return 1;

  for(int i=0; i<y.n(); i++) {
    double sum = 0;
    for(int j=0; j<x.n(); j++) {
      sum += M(i,j)*x(j);
    }
    y(i) = sum;
  }
  return 0;
}

int matMatMult(const Matrix& A, const Matrix& B, Matrix& C) {
  if(A.n(0) != C.n(0) || B.n(1) != C.n(1) || A.n(1) != B.n(0)) return 1;

  for(int j=0; j<C.n(1); j++)
  for(int i=0; i<C.n(0); i++) {
    double sum = 0;
    for(int k=0; k<A.n(1); k++) {
      sum += A(i,k)*B(k,j);
    }
    C(i,j) = sum;
  }
  return 0;
}


