#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>
#include <vector>
#include <limits>
#include "common.h"

using namespace std;

void split(const string &s, const string &delim, vector<string> &fields) {
	int start, end = 0;
	fields.clear();
	while (end < s.size()) {
		start = s.find_first_not_of(delim, end);
		if (start == string::npos) {
			return;
		}
		end = s.find_first_of(delim, start);
		if (end == string::npos) {
			end = s.size();
		}
		fields.push_back(s.substr(start, end - start));
	}
}

string getnamespace() {
	char *s;
	if ((s = getenv("SVSNAMESPACE")) == NULL) {
		return "";
	}
	
	string ns(s);
	if (ns.size() > 0 && *ns.rbegin() != '/') {
		ns.push_back('/');
	}
	return ns;
}

ofstream& get_datavis() {
	static bool first = true;
	static const char *path = getenv("SVS_DATA_PIPE");
	static ofstream f;
	if (first) {
		if (path == NULL || access(path, W_OK) < 0) {
			f.open("/dev/null");
		} else {
			f.open(path);
			f << "CLEAR" << endl;
		}
		first = false;
	}
	return f;
}

//ostream &operator<<(ostream &os, const floatvec &v) {
//	copy(v.mem, v.mem + v.sz, ostream_iterator<float>(os, " "));
//	return os;
//}

ostream &operator<<(ostream &os, const namedvec &v) {
	string name;
	for (int i = 0; i < v.size(); ++i) {
		if (!v.get_name(i, name)) {
			assert(false);
		}
		os << name << " " << v.vals[i] << endl;
	}
	return os;
}

vec3 calc_centroid(const ptlist &pts) {
	ptlist::const_iterator i;
	int d;
	vec3 c = vec3::Zero();
	
	for (i = pts.begin(); i != pts.end(); ++i) {
		c += *i;
	}

	return c / pts.size();
}

vec3 project(const vec3 &v, const vec3 &u) {
	float m = u.squaredNorm();
	if (m == 0.) {
		return vec3::Zero();
	}
	return u * (v.dot(u) / m);
}

float dir_separation(const ptlist &a, const ptlist &b, const vec3 &u) {
	int counter = 0;
	ptlist::const_iterator i;
	vec3 p;
	float x, min = numeric_limits<float>::max(), max = -numeric_limits<float>::max();
	for (i = a.begin(); i != a.end(); ++i) {
		p = project(*i, u);
		x = p[0] / u[0];
		if (x < min) {
			min = x;
		}
	}
	for (i = b.begin(); i != b.end(); ++i) {
		p = project(*i, u);
		x = p[0] / u[0];
		if (x > max) {
			max = x;
		}
	}
	
	return max - min;
}

void histogram(const evec &vals, int nbins) {
	assert(nbins > 0);
	float min = vals[0], max = vals[0], binsize, hashes_per;
	int i, b, maxcount = 0;
	vector<int> counts(nbins, 0);
	for (i = 1; i < vals.size(); ++i) {
		if (vals[i] < min) {
			min = vals[i];
		}
		if (vals[i] > max) {
			max = vals[i];
		}
	}
	binsize = (max - min) / (nbins - 1);
	if (binsize == 0) {
		cout << "All values identical (" << min << "), not drawing histogram" << endl;
		return;
	}
	for (i = 0; i < vals.size(); ++i) {
		b = (int) ((vals[i] - min) / binsize);
		assert(b < counts.size());
		counts[b]++;
		if (counts[b] > maxcount) {
			maxcount = counts[b];
		}
	}
	hashes_per = 72.0 / maxcount;
	streamsize p = cout.precision();
	cout.precision(4);
	for (i = 0; i < nbins; ++i) {
		cout << setfill(' ') << setw(5) << min + binsize * i << " - " << setw(5) << min + binsize * (i + 1) << "|";
		cout << setfill('#') << setw((int) (hashes_per * counts[i])) << '/' << counts[i] << endl;
	}
	cout.precision(p);
}

ostream& operator<<(ostream &os, const bbox &b) {
	os << b.min[0] << " " << b.min[1] << " " << b.min[2] << " " << b.max[0] << " " << b.max[1] << " " << b.max[2];
	return os;
}

void save_mat(std::ostream &os, const mat &m) {
	os << "begin_mat " << m.rows() << " " << m.cols() << endl;
	os << m << endl << "end_mat" << endl;
}

void load_mat(std::istream &is, mat &m) {
	string token;
	char *endptr;
	int nrows, ncols;
	double x;
	is >> token;
	assert(token == "begin_mat");
	is >> token;
	nrows = strtol(token.c_str(), &endptr, 10);
	assert(endptr[0] == '\0');
	is >> token;
	ncols = strtol(token.c_str(), &endptr, 10);
	assert(endptr[0] == '\0');
	
	m.resize(nrows, ncols);
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			is >> token;
			x = strtod(token.c_str(), &endptr);
			assert(endptr[0] == '\0');
			m(i, j) = x;
		}
	}
	is >> token;
	assert(token == "end_mat");
}

void save_mat(std::ostream &os, const imat &m) {
	os << "begin_mat " << m.rows() << " " << m.cols() << endl;
	os << m << endl << "end_mat" << endl;
}

void load_mat(std::istream &is, imat &m) {
	string token;
	char *endptr;
	int nrows, ncols;
	int x;
	is >> token;
	assert(token == "begin_mat");
	is >> token;
	nrows = strtol(token.c_str(), &endptr, 10);
	assert(endptr[0] == '\0');
	is >> token;
	ncols = strtol(token.c_str(), &endptr, 10);
	assert(endptr[0] == '\0');
	
	m.resize(nrows, ncols);
	for (int i = 0; i < nrows; ++i) {
		for (int j = 0; j < ncols; ++j) {
			is >> token;
			x = strtol(token.c_str(), &endptr, 10);
			assert(endptr[0] == '\0');
			m(i, j) = x;
		}
	}
	is >> token;
	assert(token == "end_mat");
}

void save_vec(std::ostream &os, const evec &v) {
	os << "begin_vec " << v.size() << endl;
	os << v << endl;
	os << "end_vec" << endl;
}

void load_vec(std::istream &is, evec &v) {
	string token;
	char *endptr;
	int n;
	double x;
	is >> token;
	assert(token == "begin_vec");
	is >> token;
	n = strtol(token.c_str(), &endptr, 10);
	assert(endptr[0] == '\0');
	
	v.resize(n);
	for (int i = 0; i < n; ++i) {
		is >> token;
		x = strtod(token.c_str(), &endptr);
		assert(endptr[0] == '\0');
		v(i) = x;
	}
	is >> token;
	assert(token == "end_vec");
}
