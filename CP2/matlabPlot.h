#ifndef MATLABPLOT_H
#define MATLABPLOT_H
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
//   title     The plot title ("Matlab Plot" if omitted)
//   x-label   The x-axis label ("x" if omitted)
//   y-label   The y-axis label ("y" if omitted)
// Outputs:    The plot file.
// Return:     none (void)
///////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <vector>
#include <string>
#include "matrix.h"

void matlabPlot(std::ofstream& fout, int n, double* y,
		const std::string& title = "Matlab Plot",
		const std::string& xLabel = "x",
		const std::string& yLabel = "y");

void matlabPlot(std::ofstream& fout, int n, double* x, double* y,
		const std::string& title = "Matlab Plot",
		const std::string& xLabel = "x",
		const std::string& yLabel = "y");

void matlabPlot(std::ofstream& fout, std::vector<double>& y,
		const std::string& title = "Matlab Plot",
		const std::string& xLabel = "x",
		const std::string& yLabel = "y");

void matlabPlot(std::ofstream& fout, std::vector<double>& x, std::vector<double>& y,
		const std::string& title = "Matlab Plot",
		const std::string& xLabel = "x",
		const std::string& yLabel = "y");

void matlabPlot(std::ofstream& fout, Vector& y,
		const std::string& title = "Matlab Plot",
		const std::string& xLabel = "x",
		const std::string& yLabel = "y");

void matlabPlot(std::ofstream& fout, Vector& x, Vector& y,
		const std::string& title = "Matlab Plot",
		const std::string& xLabel = "x",
		const std::string& yLabel = "y");

void matlabPlot(std::ofstream& fout, Vector& x, Vector& y, Matrix& z,
		const std::string& title = "Matlab Mesh Plot",
		const std::string& xLabel = "x",
		const std::string& yLabel = "y",
		const std::string& zLabel = "z");

#endif
