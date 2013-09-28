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
	serializer(os) << x << y << x_norm << cx << cx_norm;
}

void LWR::example::unserialize(istream &is) {
	unserializer(is) >> x >> y >> x_norm >> cx >> cx_norm;
}

LWR::LWR(bool alloc)
: nnbrs(10), noise_var(1e-15), alloc(alloc), xdim(-1), ydim(-1), normalized(false), center(true)
{}

LWR::~LWR() {
	if (alloc) {
		for (int i = 0, iend = data.size(); i < iend; ++i) {
			delete data[i].x;
			delete data[i].y;
		}
	}
}

void LWR::learn(const rvec &x, const rvec &cx, const rvec &y) {
	example e;
	
	if (xdim < 0) {
		xdim = x.size();
		ydim = y.size();
	} else if (x.size() != xdim || y.size() != ydim) {
		FATAL("LWR dimension mismatch");
	}
	
	if (alloc) {
		e.x = new rvec(x);
		e.y = new rvec(y);
	} else {
		e.x = &x;
		e.y = &y;
	}
	e.cx = cx;
	
	if (data.size() == 0) {
		x_min = x;
		x_max = x;
		x_range.setZero();
		cx_min = cx;
		cx_max = cx;
		cx_range.setZero();
	} else {
		for (int i = 0; i < xdim; ++i) {
			if (x(i) < x_min(i)) {
				x_min(i) = x(i);
				normalized = false;
			} else if (x(i) > x_max(i)) {
				x_max(i) = x(i);
				normalized = false;
			}
			if (cx(i) < cx_min(i)) {
				cx_min(i) = cx(i);
				normalized = false;
			} else if (cx(i) > cx_max(i)) {
				cx_max(i) = cx(i);
				normalized = false;
			}
		}
	}
	
	if (normalized) {
		// otherwise just wait for renormalization
		norm_vec(*e.x, x_min, x_range, e.x_norm);
		norm_vec(e.cx, cx_min, cx_range, e.cx_norm);
	}

	data.push_back(e);
}

bool LWR::predict(const rvec &x, const rvec &cx, rvec &y, rvec &neighbors, rvec &dists, rvec &lin_coefs) {
	if (data.size() < 2) {
		return false;
	}
	
	normalize();
	
	rvec q;
	if (center) {
		norm_vec(cx, cx_min, cx_range, q);
	} else {
		norm_vec(x, x_min, x_range, q);
	}
	
	vector<int> inds;
	nearest_neighbor(q, inds, dists);
	
	mat X(inds.size(), xdim);
	mat Y(inds.size(), ydim);
	neighbors.resize(inds.size());
	for(int i = 0, iend = inds.size(); i < iend; ++i) {
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
	rvec intercept;
	//linreg(FORWARD, X, Y, w, noise_var, false, coefs, intercept);
	//linreg(FORWARD, X, Y, cvec(), noise_var, false, coefs, intercept);
	linreg(FORWARD, X, Y, w, noise_var, false, coefs, intercept);
	y = x * coefs + intercept;
	
	lin_coefs.resize(coefs.rows() + 1);
	for (int i = 0, iend = coefs.rows(); i < iend; ++i) {
		lin_coefs(i) = coefs(i, 0);
	}
	lin_coefs(lin_coefs.size() - 1) = intercept(0);

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
	
	cx_range = cx_max - cx_min;
	// can't have division by 0
	for (int i = 0; i < cx_range.size(); ++i) {
		if (cx_range(i) == 0.0) {
			cx_range(i) = 1.0;
		}
	}

	for (int i = 0, iend = data.size(); i < iend; ++i) {
		norm_vec(data[i].cx, cx_min, cx_range, data[i].cx_norm);
	}

	x_range = x_max - x_min;
	// can't have division by 0
	for (int i = 0; i < x_range.size(); ++i) {
		if (x_range(i) == 0.0) {
			x_range(i) = 1.0;
		}
	}

	for (int i = 0, iend = data.size(); i < iend; ++i) {
		norm_vec(*data[i].x, x_min, x_range, data[i].x_norm);
	}

	normalized = true;
}

void LWR::serialize(ostream &os) const {
	assert(alloc);  // it doesn't make sense to serialize points we don't own
	serializer(os) << xdim << ydim << x_min << x_max << cx_min << cx_max << data << nnbrs << noise_var << center;
}

void LWR::unserialize(istream &is) {
	assert(alloc);
	unserializer(is) >> xdim >> ydim >> x_min >> x_max >> cx_min >> cx_max >> data >> nnbrs >> noise_var >> center;
	normalized = false;
	normalize();
}

// copied from nn.cpp to accommodate data type
void LWR::nearest_neighbor(const rvec &q, vector<int> &indexes, rvec &dists) {
	di_queue nn;
	for (int i = 0, iend = data.size(); i < iend; ++i) {
		double d;
		if (center) {
			d = (q - data[i].cx_norm).squaredNorm();
		} else {
			d = (q - data[i].x_norm).squaredNorm();
		}
		if (nn.size() < nnbrs || d < nn.top().first) {
			nn.push(std::make_pair(d, i));
			if (nn.size() > nnbrs) {
				nn.pop();
			}
		}
	}

	indexes.reserve(nn.size());
	dists.resize(nn.size());
	for (int i = 0; i < dists.size(); ++i) {
		dists(i) = nn.top().first;
		indexes.push_back(nn.top().second);
		nn.pop();
	}
}

void LWR::proxy_get_children(map<string, cliproxy*> &c) {
	c["noise_var"] = new float_proxy(&noise_var, "Expected variance of environment noise.");
	c["neighbors"] = new int_proxy(&nnbrs, "Number of neighbors to use.");
	c["center"] = new bool_proxy(&center, "Center points around target");
}

