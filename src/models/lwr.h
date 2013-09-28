#ifndef LWR_H
#define LWR_H

#include <iostream>
#include "mat.h"
#include "serializable.h"
#include "cliproxy.h"

class LWR : public serializable, public cliproxy {
public:
	LWR(bool alloc);
	~LWR();
	
	void learn(const rvec &x, const rvec &cx, const rvec &y);
	bool predict(const rvec &x, const rvec &cx, rvec &y, rvec &neighbors, rvec &dists, rvec &lin_coefs);
	void load(std::istream &is);
	void save(std::ostream &os) const;
	
	int size() const { return data.size(); }
	int xsize() const { return xdim; }
	int ysize() const { return ydim; }
	
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
	
private:
	void normalize();
	void nearest_neighbor(const rvec &q, std::vector<int> &indexes, rvec &dists);
	void proxy_get_children(std::map<std::string, cliproxy*> &c);
	
	class example : public serializable {
	public:
		rvec const *x;
		rvec const *y;
		rvec x_norm;
		rvec cx;
		rvec cx_norm;
		
		void serialize(std::ostream &os) const;
		void unserialize(std::istream &is);
	};
	
	std::vector<example> data;
	int xdim, ydim, nnbrs;
	double noise_var;
	rvec x_min, x_max, x_range;
	rvec cx_min, cx_max, cx_range;
	bool normalized, alloc, center;
};

#endif
