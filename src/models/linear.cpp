#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>
#include <sstream>
#include "linear.h"
#include "common.h"
#include "params.h"
#include "serialize.h"
#include "mat.h"
#include "platform_specific.h"  // for nextafter

using namespace std;
using namespace Eigen;

// Residual Sum of Squares
double RSS(const_mat_view X, const_mat_view y, const cvec &coefs, double intercept) {
	return ((y - X * coefs).array() - intercept).matrix().squaredNorm();
}


/* Assume that X is already centered */
void pca(const_mat_view X, mat &comps) {
	JacobiSVD<mat> svd = X.jacobiSvd(Eigen::ComputeFullV);
	comps = svd.matrixV();
}

bool solve(const_mat_view X, const_mat_view Y, mat &C) {
	C = X.colPivHouseholderQr().solve(Y);
	//C = X.jacobiSvd(ComputeThinU | ComputeThinV).solve(Y);
	return normal(C);
}

bool solve(const_mat_view X, const_mat_view y, cvec &coefs) {
	assert(y.cols() == 1);
	mat C;
	if (!solve(X, y, C)) {
		return false;
	}
	coefs = C.col(0);
	return true;
}

bool ridge(const_mat_view X, const_mat_view Y, mat &coefs) {
	mat A = X.transpose() * X;
	
	/*
	 Due to floating point precision limits, sometimes the preset
	 RIDGE_LAMBDA will not change the value of a coefficient when
	 added to it. Therefore, I find the smallest value of lambda
	 that will result in a representable increase for all
	 coefficients.
	*/
	double lambda = RIDGE_LAMBDA;
	if (lambda > 0) {
		for (int i = 0; i < A.cols(); ++i) {
			double inc = nextafter(A(i, i), INF) - A(i, i);
			if (inc > lambda) {
				lambda = inc;
			}
		}
	}
	A.diagonal().array() += lambda;
	mat B = X.transpose() * Y;
	coefs = A.ldlt().solve(B);
	return true;
}

/*
 Implements the so-called "shooting" algorithm for lasso
 regularization. Based on the tutorial and MATLAB code taken from
 
 http://www.gautampendse.com/software/lasso/webpage/lasso_shooting.html
*/
bool lasso_core(const_mat_view X, const rvec &xi_norm2, const_mat_view y, double lambda, cvec &beta) {
	int p = X.cols();
	
	for (int k = 0; k < LASSO_MAX_ITERS; ++k) {
		bool improved = false;
		for (int i = 0; i < p; ++i) {
			double b = beta(i);
			cvec yi = y - X * beta + beta(i) * X.col(i);
			double deltai = yi.dot(X.col(i));
			if (deltai < -lambda) {
				beta(i) = (deltai + lambda) / xi_norm2(i);
			} else if (deltai > LASSO_LAMBDA) {
				beta(i) = (deltai - lambda) / xi_norm2(i);
			} else {
				beta(i) = 0;
			}
			if (abs(b - beta(i)) > LASSO_TOL) {
				improved = true;
			}
		}
		if (!improved) {
			break;
		}
	}
	return true;
}

bool lasso(const_mat_view X, const_mat_view Y, mat &coefs) {
	if (!ridge(X, Y, coefs)) {
		return false;
	}
	rvec xi_norm2 = X.colwise().squaredNorm();
	for (int i = 0; i < coefs.cols(); ++i) {
		cvec beta = coefs.col(i);
		lasso_core(X, xi_norm2, Y.col(i), LASSO_LAMBDA, beta);
		coefs.col(i) = beta;
	}
	return true;
}

/*
 Support functions for forward stepwise regression
*/

// Copy upper triangle into lower triangle
void complete_symmetric_mat(mat &A) {
	int n = A.rows();
	assert(n == A.cols());

	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			A(j, i) = A(i, j);
		}
	}
}

void sweep(const mat &A, int k, rvec &work, mat &B) {
	int n = A.rows();
	work = A.row(k).array() / A(k,k);
	for (int i = 0; i < n; ++i) {
		double Aik = A(i,k);
		for (int j = i; j < n; ++j) {
			B(i,j) = A(i,j) - Aik * work(j);
		}
	}
	B.row(k) = A.row(k).array() / A(k,k);
	B.col(k) = A.col(k).array() / A(k,k);
	B(k,k) = -1.0 / A(k,k);
	complete_symmetric_mat(B);
}

/*
 Like sweep, but only fills out the last column. As long as the resulting
 matrix B is not used for further sweeping, all the information required is in
 the last column: the OLS coefficients and the RSS.
*/
void sweep_last_col(const mat &A, int k, mat &B) {
	int n = A.rows(), last = n-1;
	double x = A(k,last) / A(k,k);
	for (int i = 0; i < n; ++i) {
		B(i,last) = A(i,last) - A(i,k) * x;
	}
	B(k,last) = x;
}

/*
 Fits a linear model to the data using as few nonzero coefficients as possible.
 Basically, start with just the intercept, then keep adding additional
 predictors (nonzero coefficients) to the model that lowers Mallows' Cp
 statistic, one at a time. Stop when Cp can't be lowered further with more
 predictors.

 Uses the sweep operator for efficient multiple regression. See

   Kenneth Lange, "Numerical Analysis for Statisticians 2nd Ed", Chapter 7

 for details.
*/

bool fstep(const_mat_view X, const_mat_view Y, double var, mat &coefs_out) {
	assert(Y.cols() == 1);

	int n = X.rows(), m = X.cols(), p = 0;
	vector<bool> used(m, false);
	cvec y = Y.col(0);
	cvec coefs;
	rvec work(m);
	mat A, Anext;
	double curr_Cp;

	/*
	 Init A to be the symmetric matrix:

	 | X'X  X'y |
	 | y'X  y'y |
	*/
	A.resize(m + 1, m + 1);
	Anext.resize(m + 1, m + 1);
	A.topLeftCorner(m, m) = X.transpose() * X;
	A.topRightCorner(m, 1) = X.transpose() * y;
	A(m, m) = y.dot(y);
	complete_symmetric_mat(A);

	coefs.resize(m);
	coefs.setConstant(0.0);
	curr_Cp = A(m,m) / var - n + 2; // Mallow's Cp statistic
	                                // Estimates model out-of-sample performance
									// Note A(m,m) = RSS with no regressors

	while (p < m) {
		++p;
		double best_Cp = 0;
		int best_pred = -1;
		cvec best_c;
		for (int i = 0; i < m; ++i) {
			if (used[i]) { continue; }

			sweep_last_col(A, i, Anext);
			cvec c = Anext.topRightCorner(m, 1);
			for (int j = 0; j < m; ++j) {
				if (!used[j] && j != i) {
					c(j) = 0.0;
				}
			}
			double rss = Anext(m,m);
			double Cp = rss / var - n + 2 * (p+1);
			if (best_pred < 0 || Cp < best_Cp || 
			    (Cp == best_Cp && c.squaredNorm() < best_c.squaredNorm()))
			{
				best_Cp = Cp;
				best_c = c;
				best_pred = i;
			}
		}
		if (best_Cp < curr_Cp ||
		    (best_Cp == curr_Cp && best_c.squaredNorm() < coefs.squaredNorm()))
		{
			used[best_pred] = true;
			curr_Cp = best_Cp;
			coefs = best_c;
			sweep(A, best_pred, work, Anext);
			A = Anext;
		} else {
			break;
		}
	}
	coefs_out.resize(m, 1);
	coefs_out.col(0) = coefs;
	return true;
}

/*
 Cleaning consists of:
 
 1. collapsing all columns whose elements are identical into a single constant column.
 2. Setting elements whose absolute values are smaller than SAME_THRESH to 0.
 3. Centering X and Y data so that intercepts don't need to be considered.
*/
void clean_lr_data(mat &X, vector<int> &used_cols) {
	int xdims = X.cols(), ndata = X.rows(), newdims = 0;
	
	for (int i = 0; i < xdims; ++i) {
		if (!uniform(X.col(i))) {
			if (newdims < i) {
				X.col(newdims) = X.col(i);
			}
			used_cols.push_back(i);
			++newdims;
		}
	}
	
	X.conservativeResize(ndata, newdims);
	for (int i = 0; i < ndata; ++i) {
		for (int j = 0; j < newdims; ++j) {
			if (fabs(X(i, j)) < SAME_THRESH) {
				X(i, j) = 0.0;
			}
		}
	}
}

void center_data(mat &X, rvec &Xm) {
	Xm = X.colwise().mean();
	X.rowwise() -= Xm;
}

void augment_ones(mat &X) {
	X.conservativeResize(X.rows(), X.cols() + 1);
	X.col(X.cols() - 1).setConstant(1.0);
}

void fix_for_wols(mat &X, mat &Y, const cvec &w) {
	assert(w.size() == X.rows());
	X.conservativeResize(X.rows(), X.cols() + 1);
	X.col(X.cols() - 1).setConstant(1.0);
	
	for (int i = 0, iend = X.cols(); i < iend; ++i) {
		X.col(i).array() *= w.array();
	}
	for (int i = 0, iend = Y.cols(); i < iend; ++i) {
		Y.col(i).array() *= w.array();
	}
}

/*
 Perform linear regression. Assumes that input X and Y are already
 cleaned and centered.
*/
bool linreg_clean(regression_type t, const_mat_view X, const_mat_view Y, double var, mat &coefs) {
	switch (t) {
		case OLS:
			return solve(X, Y, coefs);
		case RIDGE:
			return ridge(X, Y, coefs);
		case LASSO:
			return lasso(X, Y, coefs);
		case FORWARD:
			return fstep(X, Y, var, coefs);
		default:
			FATAL("unknown regression type");
	}
	return false;
}


/*
 Clean up input data to avoid instability, then perform some form of
 regression.
 
 Note that this function modifies inputs X and Y to avoid redundant
 copies.
*/
bool linreg_d(regression_type t, mat &X, mat &Y, const cvec &w, double var, mat &coefs, rvec &inter) {
	int ndata = X.rows();
	int xdim = X.cols();
	int ydim = Y.cols();
	mat coefs1;
	rvec Xm, Ym, inter1;
	vector<int> used;
	
	clean_lr_data(X, used);
	/*
	 The two ways to deal with intercepts:
	 
	 1. Center data before regression, then calculate intercept afterwards using
	    means.
	 2. Append a column of 1's to X. The coefficient for this column will be the
	    intercept.
	 
	 Unfortunately from what I can gather, method 1 doesn't work with weighted
	 regression, and method 2 doesn't work with ridge regression.
	*/
	
	bool augment = (w.size() > 0);
	if (augment) {
		assert(t != RIDGE);
		fix_for_wols(X, Y, w);
	} else {
		center_data(X, Xm);
		center_data(Y, Ym);
	}
	
	if (!linreg_clean(t, X, Y, var, coefs1)) {
		return false;
	}
	
	if (augment) {
		assert(coefs1.rows() == used.size() + 1);
		inter = coefs1.row(coefs1.rows() - 1);
	} else {
		inter = Ym - (Xm * coefs1);
	}
	
	coefs.resize(xdim, ydim);
	coefs.setConstant(0.0);
	for (int i = 0; i < used.size(); ++i) {
		coefs.row(used[i]) = coefs1.row(i);
	}
	return true;
}

bool linreg (
	regression_type t,
	const_mat_view X,
	const_mat_view Y,
	const cvec &w,
	double var,
	mat &coefs,
	rvec &intercept ) 
{
	mat Xc(X), Yc(Y);
	return linreg_d(t, Xc, Yc, w, var, coefs, intercept);
}

bool nfoldcv(const_mat_view X, const_mat_view Y, double var, int n, regression_type t, rvec &avg_error) {
	assert(X.rows() >= n);
	
	int chunk, extra, total_rows, train_rows, test_rows, i, start, end;
	int xcols, ycols;
	mat Xrand, Yrand, Xtrain, Xtest, Ytrain, Ytest, pred, error;
	mat coefs;
	rvec intercept;
	vector<int> r(X.rows());
	
	xcols = X.cols();
	ycols = Y.cols();
	total_rows = X.rows();
	chunk = total_rows / n;
	extra = X.rows() - n * chunk;
	avg_error.resize(ycols);
	avg_error.setConstant(0.0);
	
	// shuffle X and Y
	for (int i = 0, iend = r.size(); i < iend; ++i) {
		r[i] = i;
	}
	random_shuffle(r.begin(), r.end());
	Xrand.resize(total_rows, xcols);
	Yrand.resize(total_rows, ycols);
	for (int i = 0, iend = r.size(); i < iend; ++i) {
		Xrand.row(i) = X.row(r[i]);
		Yrand.row(i) = Y.row(r[i]);
	}
	
	for (i = 0, start = 0; i < n; ++i) {
		if (i < extra) {
			test_rows = chunk + 1;
		} else {
			test_rows = chunk;
		}
		train_rows = total_rows - test_rows;
		end = start + test_rows;
		
		Xtest = Xrand.block(start, 0, test_rows, xcols);
		Ytest = Yrand.block(start, 0, test_rows, ycols);
		
		Xtrain.resize(train_rows, xcols);
		Ytrain.resize(train_rows, ycols);
		
		Xtrain.block(0, 0, start, xcols) = Xrand.block(0, 0, start, xcols);
		Xtrain.block(start, 0, train_rows - start, xcols) = Xrand.block(end, 0, train_rows - start, xcols);
		Ytrain.block(0, 0, start, ycols) = Yrand.block(0, 0, start, ycols);
		Ytrain.block(start, 0, train_rows - start, ycols) = Yrand.block(end, 0, train_rows - start, ycols);
		
		if (!linreg_d(t, Xtrain, Ytrain, cvec(), var, coefs, intercept))
			return false;
		
		pred = Xtest * coefs;
		for (int j = 0, jend = pred.rows(); j < jend; ++j) {
			pred.row(j) += intercept;
		}
		error = (Ytest - pred).array().abs().matrix();
		avg_error += error.colwise().sum();
		start += test_rows;
	}
	avg_error.array() /= total_rows;
	return true;
}
