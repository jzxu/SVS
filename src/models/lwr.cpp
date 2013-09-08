#include <iostream>
#include <cassert>
#include "lwr.h"
#include "nn.h"
#include "linear.h"
#include "serialize.h"

using namespace std;

void norm_vec(const rvec &v, const rvec &min, const rvec &range, rvec &n) {
	n = ((v.array() - min.array()) / range.array()).matrix();
}

double gpr_kernel(const rvec &x1, const rvec &x2, double sigma, double clen) {
	return (sigma * sigma) * exp((x1 - x2).squaredNorm() / (-2 * clen * clen));
}

/*
 Vanilla GPR
*/
void gpr(const mat &Xtrain, const cvec &ytrain, const rvec &xtest, double sigma, double clen, double &mean, double &variance) {
	int n = Xtrain.rows();
	mat K(n, n), Kinv;
	
	for (int i = 0; i < n; ++i) {
		for (int j = i; j < n; ++j) {
			K(i, j) = gpr_kernel(Xtrain.row(i), Xtrain.row(j), sigma, clen);
			K(j, i) = K(i, j);
		}
	}
	Kinv = K.inverse();

	rvec Kstar(n);
	for (int i = 0; i < n; ++i) {
		Kstar(i) = gpr_kernel(xtest, Xtrain.row(i), sigma, clen);
	}

	double Kstar2 = gpr_kernel(xtest, xtest, sigma, clen);
	mean = (Kstar * Kinv * ytrain)(0, 0);
	variance = Kstar2 - (Kstar * Kinv * Kstar.transpose())(0, 0);
}

void LWR::example::serialize(ostream &os) const {
	serializer(os) << x << y;
}

void LWR::example::unserialize(istream &is) {
	unserializer(is) >> x >> y;
}

LWR::LWR(int nnbrs, double noise_var, bool alloc)
: nnbrs(nnbrs), alloc(alloc), xdim(-1), ydim(-1), normalized(false), noise_var(noise_var)
{}

LWR::~LWR() {
	if (alloc) {
		for (int i = 0, iend = data.size(); i < iend; ++i) {
			delete data[i].x;
			delete data[i].y;
		}
	}
}

void LWR::learn(const rvec &x, const rvec &y) {
	example e;
	
	if (xdim < 0) {
		xdim = x.size();
		ydim = y.size();
		xmin.resize(xdim);
		xmax.resize(xdim);
		xrange.resize(xdim);
	}
	
	if (alloc) {
		e.x = new rvec(x);
		e.y = new rvec(y);
	} else {
		e.x = &x;
		e.y = &y;
	}
	data.push_back(e);
	
	if (data.size() == 1) {
		xmin = x;
		xmax = x;
		xrange.setZero();
	} else {
		for (int i = 0; i < xdim; ++i) {
			if (x[i] < xmin[i]) {
				xmin[i] = x[i];
				normalized = false;
			} else if (x[i] > xmax[i]) {
				xmax[i] = x[i];
				normalized = false;
			}
		}
	}
	
	if (normalized) {
		// otherwise just wait for renormalization
		rvec xn;
		norm_vec(x, xmin, xrange, xn);
		Xnorm.append_row(xn);
	}
}

bool LWR::predict(const rvec &x, rvec &y, rvec &neighbors, rvec &dists, rvec &lin_coefs, rvec &intercept) {
	int k = data.size() > nnbrs ? nnbrs : data.size();
	if (k < 2) {
		return false;
	}
	
	normalize();
	
	rvec xn;
	norm_vec(x, xmin, xrange, xn);
	
	vector<int> inds;
	brute_nearest_neighbor(Xnorm.get(), xn, k, inds, dists);
	
	mat X(k, xdim);
	mat Y(k, ydim);
	neighbors.resize(k);
	dists.resize(k);
	for(int i = 0; i < k; ++i) {
		X.row(i) = *data[inds[i]].x;
		Y.row(i) = *data[inds[i]].y;
		neighbors(i) = inds[i];
	}
	
	rvec w = dists.array().pow(-3).sqrt();
	
	/*
	 Any neighbor whose weight is infinity is close enough
	 to provide an exact solution.	If any exist, take their
	 average as the solution.  If we don't do this the solve()
	 will fail due to infinite values in Z and V.
	*/
	rvec closeavg = rvec::Zero(ydim);
	int nclose = 0;
	for (int i = 0; i < w.size(); ++i) {
		if (w(i) == INF) {
			closeavg += Y.row(i);
			++nclose;
		}
	}
	if (nclose > 0) {
		for(int i = 0; i < closeavg.size(); ++i) {
			y[i] = closeavg(i) / nclose;
		}
		return true;
	}

	mat coefs;
	linreg_d(FORWARD, X, Y, w, noise_var, coefs, intercept);
	//linreg_d(LASSO, X, Y, cvec(), noise_var, coefs, intercept);
	y = x * coefs + intercept;
	
	lin_coefs.resize(coefs.rows());
	for (int i = 0, iend = coefs.rows(); i < iend; ++i) {
		lin_coefs(i) = coefs(i, 0);
	}
	/*
	double sigma, clen, mean, var;
	sigma = 1e-4;
	clen = 10;
	gpr(X, Y.col(0), x, sigma, clen, mean, var);
	y(0) = mean;
	*/
	return true;
}

void LWR::normalize() {
	if (normalized) {
		return;
	}
	
	xrange = xmax - xmin;
	// can't have division by 0
	for (int i = 0; i < xrange.size(); ++i) {
		if (xrange[i] == 0.0) {
			xrange[i] = 1.0;
		}
	}

	Xnorm.resize(data.size(), data[0].x->size());
	rvec n;
	for (int i = 0, iend = data.size(); i < iend; ++i) {
		norm_vec(*data[i].x, xmin, xrange, n);
		Xnorm.row(i) = n;
	}
	normalized = true;
}

void LWR::serialize(ostream &os) const {
	assert(alloc);  // it doesn't make sense to serialize points we don't own
	serializer(os) << xdim << ydim << xmin << xmax << data;
}

void LWR::unserialize(istream &is) {
	assert(alloc);
	unserializer(is) >> xdim >> ydim >> xmin >> xmax >> data;
	normalized = false;
	normalize();
}
