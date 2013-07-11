#ifndef LWR_H
#define LWR_H

#include <iostream>
#include "mat.h"
#include "serializable.h"

class LWR : public serializable {
public:
	LWR(int nnbrs, double noise_var, bool alloc);
	~LWR();
	
	void learn(const rvec &x, const rvec &y);
	bool predict(const rvec &x, rvec &y, rvec &neighbors, rvec &dists, rvec &lin_coefs, rvec &intercept);
	void load(std::istream &is);
	void save(std::ostream &os) const;
	
	int size() const { return data.size(); }
	int xsize() const { return xdim; }
	int ysize() const { return ydim; }
	
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
	
private:
	void normalize();
	
	class example : public serializable {
	public:
		rvec const *x;
		rvec const *y;
		
		void serialize(std::ostream &os) const;
		void unserialize(std::istream &is);
	};
	
	std::vector<example> data;
	dyn_mat Xnorm;
	int xdim, ydim, nnbrs;
	rvec xmin, xmax, xrange;
	bool normalized, alloc;
	double noise_var;
};

#endif
