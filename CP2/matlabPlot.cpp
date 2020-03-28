///////////////////////////////////////////////////////////////////////////////
// OUTPUT FOR MATLAB PLOT FUNCTION
//
// Provide a simple output routine for producing plots using the
// matlab plot function.
//
// Inputs:
//   fout      The output stream
//   n         The number of points to plot
//   x,y       The x and y coordinates (x may be omitted if x = {1,2,...,n}
//   title     The plot title
//   x-label   The x-axis label
//   y-label   The y-axis label
// Outputs:    The plot file.
// Return:     none (void)
///////////////////////////////////////////////////////////////////////////////
#include "matlabPlot.h"
#include <iostream>
using namespace std;

//#define TEST
#define ROW  // output in rows

static void constructPlot1(ofstream& fout, const string& title,
			  const string& xLabel, const string& yLabel) {
  fout << "figure\n";
  fout << "plot(y);\n";
  fout << "title('" << title << "');\n";
  fout << "xlabel('" << xLabel << "');\n";
  fout << "ylabel('" << yLabel << "');\n";
}

static void constructPlot2(ofstream& fout, const string& title,
			  const string& xLabel, const string& yLabel) {
  fout << "figure\n";
  fout << "plot(x,y);\n";
  fout << "title('" << title << "');\n";
  fout << "xlabel('" << xLabel << "');\n";
  fout << "ylabel('" << yLabel << "');\n";
}

static void constructPlot2_leg(ofstream& fout, const string& title,
			  const string& xLabel, const string& yLabel,
                          const string& legend1, const string& legend2) {
  fout << "figure\n";
  fout << "plot(x,y1,'LineWidth',2,'b');\n";
  fout << "hold on\n";
  fout << "plot(x,y2,'LineWidth',2,'r');\n";
  fout << "title('" << title << "');\n";
  fout << "xlabel('" << xLabel << "');\n";
  fout << "ylabel('" << yLabel << "');\n";
  fout << "legend('" << legend1 << "','" << legend2 << "');\n";
}

static void constructPlot3(ofstream& fout, const string& title,
			  const string& xLabel, const string& yLabel, 
                          const string& zLabel) {
  fout << "figure\n";
  fout << "mesh(x,y,z);\n";
  fout << "title('" << title << "');\n";
  fout << "xlabel('" << xLabel << "');\n";
  fout << "ylabel('" << yLabel << "');\n";
  fout << "zlabel('" << zLabel << "');\n";
}

void matlabPlot(ofstream& fout, int n, double* y,
		const string& title, const string& xLabel,
		const string& yLabel) {
#ifdef ROW
  // Write y
  fout << "y = [";
  for(int i=0; i<n; i++) {
    if (i != 0) fout << ", ";
    fout << y[i];
  }
  fout << "];\n";
#else
  // Write y
  fout << "y = [\n";
  for(int i=0; i<n; i++) fout << y[i] << "\n";
  fout << "];\n";
#endif

  constructPlot1(fout, title, xLabel, yLabel);
}

void matlabPlot(ofstream& fout, int n, double* x, double* y,
		const string& title, const string& xLabel,
		const string& yLabel) {
#ifdef ROW
  // Write x
  fout << "x = [";
  for(int i=0; i<n; i++) {
    if (i != 0) fout << ", ";
    fout << x[i];
  }
  fout << "];\n";

  // Write y
  fout << "y = [";
  for(int i=0; i<n; i++) {
    if (i != 0) fout << ", ";
    fout << y[i];
  }
  fout << "];\n";
#else
  // Write x
  fout << "x = [\n";
  for(int i=0; i<n; i++) fout << x[i] << "\n";
  fout << "];\n";

  // Write y
  fout << "y = [\n";
  for(int i=0; i<n; i++) fout << y[i] << "\n";
  fout << "];\n";
#endif

  constructPlot2(fout, title, xLabel, yLabel);
}

void matlabPlot(ofstream& fout, vector<double>& y,
		const string& title, const string& xLabel,
		const string& yLabel) {
#ifdef ROW
  // Write y
  fout << "y = [";
  for(int i=0; i<y.size(); i++) {
    if (i != 0) fout << ", ";
    fout << y[i];
  }
  fout << "];\n";
#else
  // Write y
  fout << "y = [\n";
  for(int i=0; i<y.size(); i++) fout << y[i] << "\n";
  fout << "];\n";
#endif

  constructPlot1(fout, title, xLabel, yLabel);
}

void matlabPlot(ofstream& fout, vector<double>& x, vector<double>& y,
		const string& title, const string& xLabel,
		const string& yLabel) {
  if (x.size() != y.size()) 
    cerr << "Sizes differ in matlabPlot (vector<double>&, vector<double>&)";
#ifdef ROW
  // Write x
  fout << "x = [";
  for(int i=0; i<x.size(); i++) {
    if (i != 0) fout << ", ";
    fout << x[i];
  } 
  fout << "];\n";

  // Write y
  fout << "y = [";
  for(int i=0; i<y.size(); i++) {
    if (i != 0) fout << ", ";
    fout << y[i];
  } 
  fout << "];\n";
#else
  // Write x
  fout << "x = [\n";
  for(int i=0; i<x.size(); i++) fout << x[i] << "\n";
  fout << "];\n";

  // Write y
  fout << "y = [\n";
  for(int i=0; i<y.size(); i++) fout << y[i] << "\n";
  fout << "];\n";
#endif

  constructPlot2(fout, title, xLabel, yLabel);
}

void matlabPlot(ofstream& fout, vector<double>& x, vector<double> y[2],
		const string& title, const string& xLabel, const string& yLabel, 
                const string& legend1, const string& legend2) {
  if ((x.size() != y[0].size()) || (x.size() != y[1].size())) 
    cerr << "Sizes differ in matlabPlot (vector<double>&, vector<double>y[2])";
#ifdef ROW
  // Write x
  fout << "x = [";
  for(int i=0; i<x.size(); i++) {
    if (i != 0) fout << ", ";
    fout << x[i];
  } 
  fout << "];\n";

  // Write y
  fout << "y1 = [";
  for(int i=0; i<y[0].size(); i++) {
    if (i != 0) fout << ", ";
    fout << y[0][i];
  } 
  fout << "];\n";
  fout << "y2 = [";
  for(int i=0; i<y[1].size(); i++) {
    if (i != 0) fout << ", ";
    fout << y[1][i];
  } 
  fout << "];\n";
#else
  // Write x
  fout << "x = [\n";
  for(int i=0; i<x.size(); i++) fout << x[i] << "\n";
  fout << "];\n";

  // Write y
  fout << "y1 = [\n";
  for(int i=0; i<y[0].size(); i++) fout << y[0][i] << "\n";
  fout << "];\n";
  fout << "y2 = [\n";
  for(int i=0; i<y[1].size(); i++) fout << y[1][i] << "\n";
  fout << "];\n";
#endif

  constructPlot2_leg(fout, title, xLabel, yLabel, legend1, legend2);
}

void matlabPlot(ofstream& fout, Vector& y,
		const string& title, const string& xLabel,
		const string& yLabel) {
#ifdef ROW
  // Write y
  fout << "y = [";
  for(int i=0; i<y.n(); i++) {
    if (i != 0) fout << ", ";
    fout << y(i);
  }
  fout << "];\n";
#else
  // Write y
  fout << "y = [\n";
  for(int i=0; i<y.n(); i++) fout << y(i) << "\n";
  fout << "];\n";
#endif

  constructPlot1(fout, title, xLabel, yLabel);
}

void matlabPlot(ofstream& fout, Vector& x, Vector& y,
		const string& title, const string& xLabel,
		const string& yLabel) {
  if (x.n() != y.n()) 
    cerr << "Sizes differ in matlabPlot (Vector&, Vector&)";
#ifdef ROW
  // Write x
  fout << "x = [";
  for(int i=0; i<x.n(); i++) {
    if (i != 0) fout << ", ";
    fout << x(i);
  }
  fout << "];\n";

  // Write y
  fout << "y = [";
  for(int i=0; i<y.n(); i++) {
    if (i != 0) fout << ", ";
    fout << y(i);
  }
  fout << "];\n";
#else
  // Write x
  fout << "x = [\n";
  for(int i=0; i<x.n(); i++) fout << x(i) << "\n";
  fout << "];\n";

  // Write y
  fout << "y = [\n";
  for(int i=0; i<y.n(); i++) fout << y(i) << "\n";
  fout << "];\n";
#endif

  constructPlot2(fout, title, xLabel, yLabel);
}

void matlabPlot(ofstream& fout, Vector& x, Vector& y, Matrix& z,
                const string& title, const string& xLabel,
                const string& yLabel, const string& zLabel) {
  if ((x.n() != z.n(0)) || (y.n() != z.n(1))) 
    cerr << "Sizes differ in matlabPlot (Vector&, Vector&, Matrix&)"; 
#ifdef ROW
  // Write x
  fout << "x = [";
  for(int i=0; i<x.n(); i++) {
    if (i != 0) fout << ", ";
    fout << x(i);
  }
  fout << "];\n";
  
  // Write y
  fout << "y = [";
  for(int i=0; i<y.n(); i++) {
    if (i != 0) fout << ", ";
    fout << y(i);
  }
  fout << "];\n";
  
  // Write z
  fout << "z = [";
  for (int i=0; i<z.n(1); i++) {
    if (i != 0) fout << ";\n";
    for(int j=0; j<z.n(0); j++) {
      if (j!=0) fout << ", ";
      fout << z(j, i);
    }
  }
  fout << "];\n";
 #else
  // Write x
  fout << "x = [\n";
  for(int i=0; i<x.n(); i++) fout << x(i) << "\n";
  fout << "];\n";

  // Write y
  fout << "y = [\n";
  for(int i=0; i<y.n(); i++) fout << y(i) << "\n";
  fout << "];\n";
  
  // Write z
  fout << "z = [\n";
  for (int i=0; i<z.n(1); i++) {
    for(int j=0; j<z.n(0); j++) fout << z(j, i) << "\n";
  }
  fout << "];\n";
#endif

 
  constructPlot3(fout, title, xLabel, yLabel, zLabel);
}

#ifdef TEST
void main() {
  const int n=6;
  double x[n] = {1,2,4,5,7,8};
  double y[n] = {0,-1,5,3,9,8};

  string title("Test Plot");

  ofstream fout("testPlot1.m");

  matlabPlot(fout, n, x, y, title, "x values", "y values");
  fout.close();

  fout.open("testPlot2.m");
  matlabPlot(fout, n, y, title, "x values", "y values");
}
#endif
