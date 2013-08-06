#include <vector>
#include <map>
#include "numeric_classifier.h"
#include "nn.h"
#include "common.h"
#include "serialize.h"

using namespace std;
using namespace Eigen;

class LDA : public numeric_classifier {
public:
	LDA();
	
	void learn(mat &data, const std::vector<int> &classes);
	int classify(const rvec &x) const;
	bool project(const rvec &x, rvec &p) const;
	void inspect(std::ostream &os) const;
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
	
private:
	mat W, projected;
	rvec J;
	std::vector<int> classes, used_cols;
	bool degenerate;
	int degenerate_class;
};

class sign_classifier : public numeric_classifier {
public:
	sign_classifier();

	void learn(mat &data, const std::vector<int> &classes);
	int classify(const rvec &x) const;
	void inspect(std::ostream &os) const;
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);

private:
	int dim, sgn, size;
};

class dtree_classifier : public numeric_classifier {
public:
	dtree_classifier();
	~dtree_classifier();

	void learn(mat &data, const std::vector<int> &classes);
	int classify(const rvec &x) const;
	void inspect(std::ostream &os) const;
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);

private:
	dtree_classifier(int depth);

	void learn_rec(
		const mat &data,
		const std::vector<int> &classes,
		const std::vector<std::vector<int> > &sorted_dims,
		const std::vector<bool> &ex_usable,
		const std::vector<bool> &dim_usable);
	
	void choose_split(
		const mat &data,
		const std::vector<int> &classes,
		const std::vector<std::vector<int> > &sorted_dims,
		const std::vector<bool> &ex_usable,
		const std::vector<bool> &dim_usable);

	void print(std::ostream &os) const;

	int depth;
	int split_dim;
	double split_val;
	int cls;
	dtree_classifier *left, *right;
};

bool hasnan(const_mat_view m) {
	return (m.array() != m.array()).any();
}

#define BETA 1.0e-10

void clean_data(mat &data, vector<int> &nonuniform_cols) {
	del_uniform_cols(data, data.cols(), nonuniform_cols);
	data.conservativeResize(data.rows(), nonuniform_cols.size());
	/*
	 I used to add a small random offset to each element to try to fix numerical
	 stability. Now I add a constant BETA to the Sw matrix instead.

	mat rand_offsets = mat::Random(data.rows(), data.cols()) / 10000;
	data += rand_offsets;
	*/
}

int largest_class(const vector<int> &c) {
	map<int, int> counts;
	for (int i = 0; i < c.size(); ++i) {
		++counts[c[i]];
	}
	map<int, int>::iterator i;
	int largest_count = -1;
	int largest = -1;
	vector<int> used_cols;
	for (i = counts.begin(); i != counts.end(); ++i) {
		if (largest_count < i->second) {
			largest = i->first;
		}
	}
	return largest;
}

LDA::LDA() : degenerate(false), degenerate_class(9090909)
{}

void LDA::learn(mat &data, const vector<int> &cls) {
	classes = cls;
	clean_data(data, used_cols);
	
	if (data.cols() == 0) {
		cerr << "Degenerate case, no useful classification data." << endl;
		degenerate = true;
		degenerate_class = largest_class(classes);
		return;
	}
	int d = data.cols();
	vector<rvec> class_means;
	vector<int> counts, cmem, norm_classes;
	
	// Remap classes to consecutive integers from 0
	for (int i = 0; i < classes.size(); ++i) {
		bool found = false;
		for (int j = 0; j < norm_classes.size(); ++j) {
			if (norm_classes[j] == classes[i]) {
				class_means[j] += data.row(i);
				++counts[j];
				cmem.push_back(j);
				found = true;
				break;
			}
		}
		if (!found) {
			norm_classes.push_back(classes[i]);
			class_means.push_back(data.row(i));
			counts.push_back(1);
			cmem.push_back(norm_classes.size() - 1);
		}
	}
	
	int C = norm_classes.size();
	
	/*
	 Degenerate case: fewer nonuniform dimensions in source
	 data than there are classes. Don't try to project. This then
	 becomes an ordinary NN classifier.
	*/
	if (d < C - 1) {
		W = mat::Identity(d, d);
		projected = data;
		return;
	}
	
	for (int i = 0; i < C; ++i) {
		class_means[i] /= counts[i];
	}
	
	mat Sw = mat::Zero(d, d);
	for (int i = 0; i < data.rows(); ++i) {
		rvec x = data.row(i) - class_means[cmem[i]];
		Sw += x.transpose() * x;
	}
	
	Sw.diagonal().array() += BETA;
	rvec all_mean = data.colwise().mean();
	mat Sb = mat::Zero(d, d);
	for (int i = 0; i < C; ++i) {
		rvec x = class_means[i] - all_mean;
		Sb += counts[i] * (x.transpose() * x);
	}
	
	mat Swi = Sw.inverse();
	mat S = Swi * Sb;
	assert(!hasnan(S));
	
	/*
	ofstream tmp("/tmp/swsb.ser"), tmp2("/tmp/swi.ser"), tmp3("/tmp/S.ser");
	::serialize(Sw, tmp);
	::serialize(Sb, tmp);
	tmp.close();
	::serialize(Swi, tmp2);
	tmp2.close();
	::serialize(S, tmp3);
	tmp3.close();
	*/
	
	EigenSolver<mat> e;
	e.compute(S);
	VectorXcd eigenvals = e.eigenvalues();
	MatrixXcd eigenvecs = e.eigenvectors();

	W.resize(d, C - 1);
	J.resize(C - 1);
	for (int i = 0; i < C - 1; ++i) {
		int best = -1;
		for (int j = 0; j < eigenvals.size(); ++j) {
			if (best < 0 || eigenvals(j).real() > eigenvals(best).real()) {
				best = j;
			}
		}
		J(i) = eigenvals(best).real();
		for (int j = 0; j < d; ++j) {
			W(j, i) = eigenvecs(j, best).real();
		}
		eigenvals(best) = complex<double>(-1, 0);
	}
	projected = data * W;
}

int LDA::classify(const rvec &x) const {
	rvec p;
	int best;
	
	if (!project(x, p))
		return degenerate_class;
	
	(projected.rowwise() - p).rowwise().squaredNorm().minCoeff(&best);
	return classes[best];
}

bool LDA::project(const rvec &x, rvec &p) const {
	if (degenerate)
		return false;
	
	if (used_cols.size() < x.size()) {
		rvec x1(used_cols.size());
		for (int i = 0; i < used_cols.size(); ++i) {
			x1(i) = x(used_cols[i]);
		}
		p = x1 * W;
	} else {
		p = x * W;
	}
	return true;
}

void LDA::inspect(ostream &os) const {
	if (degenerate) {
		os << "degenerate (" << degenerate_class << ")" << endl;
		return;
	}
	table_printer t;
	for (int i = 0; i < W.rows(); ++i) {
		t.add_row() << used_cols[i];
		for (int j = 0; j < W.cols(); ++j) {
			t << W(i, j);
		}
	}
	t.print(os);
}

void LDA::serialize(ostream &os) const {
	serializer(os) << W << projected << J << classes << used_cols << degenerate << degenerate_class;
}

void LDA::unserialize(istream &is) {
	unserializer(is) >> W >> projected >> J >> classes >> used_cols >> degenerate >> degenerate_class;
}

sign_classifier::sign_classifier() : dim(-1), sgn(0), size(-1) {}

void sign_classifier::learn(mat &data, const vector<int> &classes) {
	if (size == -1) {
		size = data.cols();
	} else {
		assert(size == data.cols());
	}
	cvec cls2(classes.size());
	for (int i = 0, iend = cls2.size(); i < iend; ++i) {
		cls2(i) = (classes[i] == 0 ? -1 : 1);
	}
	
	for (int i = 0, iend = data.rows(); i < iend; ++i) {
		for (int j = 0, jend = data.cols(); j < jend; ++j) {
			data(i, j) = sign(data(i, j));
		}
	}
	
	double best;
	dim = -1;
	for (int i = 0, iend = data.cols(); i < iend; ++i) {
		double dp = cls2.dot(data.col(i));
		if (dim == -1 || fabs(dp) > best) {
			dim = i;
			best = fabs(dp);
			sgn = sign(dp);
		}
	}
}

int sign_classifier::classify(const rvec &x) const {
	assert(size == x.size());
	return max(sgn * sign(x(dim)), 0);
}

void sign_classifier::inspect(std::ostream &os) const {
	os << "dim: " << dim << " sign: " << sgn << endl;
}

void sign_classifier::serialize(ostream &os) const {
	serializer(os) << dim << sgn << size;
}

void sign_classifier::unserialize(istream &is) {
	unserializer(is) >> dim >> sgn >> size;
}

dtree_classifier::dtree_classifier()
: depth(0), split_dim(-1), split_val(NAN), cls(-1), left(NULL), right(NULL)
{}

dtree_classifier::dtree_classifier(int depth)
: depth(depth), split_dim(-1), split_val(NAN), cls(-1), left(NULL), right(NULL)
{}

dtree_classifier::~dtree_classifier() {
	if (left) {
		delete left;
	}
	if (right) {
		delete right;
	}
}

double entropy(const vector<int> &counts, int total) {
	if (total == 0)
		return 0.0;
	
	double e = 0.0;
	for (int i = 0, iend = counts.size(); i < iend; ++i) {
		double p = counts[i] / static_cast<double>(total);
		if (p > 0) {
			e += -p * log2(p);
		}
	}
	return e;
}

double split_entropy(const vector<int> &left_counts, const vector<int> &right_counts, int left_total, int right_total) {
	double total = left_total + right_total;
	double e = 0.0;
	e += (left_total / total) * entropy(left_counts, left_total);
	e += (right_total / total) * entropy(right_counts, right_total);
	return e;
}

int most_common_class(const vector<int> &classes, const vector<bool> &ex_usable) {
	vector<int> counts;
	for (int i = 0, iend = classes.size(); i < iend; ++i) {
		if (ex_usable[i]) {
			if (classes[i] >= counts.size()) {
				counts.resize(classes[i] + 1);
			}
			++counts[classes[i]];
		}
	}
	return argmax(counts);
}

void dtree_classifier::choose_split(
	const mat &data,
	const vector<int> &classes,
	const vector<vector<int> > &sorted_dims,
	const vector<bool> &ex_usable,
	const vector<bool> &dim_usable)
{
	int n = data.rows();
	split_dim = -1;
	split_val = INF;
	double best_entropy = 0.0;
	for (int i = 0, iend = sorted_dims.size(); i < iend; ++i) {
		if (!dim_usable[i])
			continue;
		
		vector<int> left_counts(2), right_counts(2); // 2 classes - 0 and 1
		int left_total = 0, right_total = 0;
		for (int j = 0; j < n; ++j) {
			int a = sorted_dims[i][j];
			if (ex_usable[a]) {
				++right_counts[classes[a]];
				++right_total;
			}
		}
		for (int j = 0; j < n;) {
			int a = sorted_dims[i][j];
			if (!ex_usable[a]) {
				++j;
				continue;
			}

			++left_counts[classes[a]];
			--right_counts[classes[a]];
			++left_total;
			--right_total;

			int k;
			for (k = j + 1; k < n && (data(a,i) == data(sorted_dims[i][k],i) || !ex_usable[sorted_dims[i][k]]); ++k)
				;
			if (k >= n)
				break;

			int b = sorted_dims[i][k];
			assert(data(a,i) != data(b,i) && ex_usable[a] && ex_usable[b]);
			if (classes[a] != classes[b]) {
				double e = split_entropy(left_counts, right_counts, left_total, right_total);
				if (split_dim < 0 || e < best_entropy) {
					split_dim = i;
					split_val = (data(a, i) + data(b, i)) / 2;
					best_entropy = e;
				}
			}
			j = k;
		}
	}
}

void dtree_classifier::learn_rec(
	const mat &data,
	const vector<int> &classes,
	const vector<vector<int> > &sorted_dims,
	const vector<bool> &ex_usable,
	const vector<bool> &dim_usable)
{
	choose_split(data, classes, sorted_dims, ex_usable, dim_usable);
	if (split_dim < 0) {
		cls = most_common_class(classes, ex_usable);
		return;
	}

	vector<bool> left_ex_usable(ex_usable.begin(), ex_usable.end());
	vector<bool> right_ex_usable(ex_usable.begin(), ex_usable.end());
	vector<bool> dim_usable_next(dim_usable.begin(), dim_usable.end());

	dim_usable_next[split_dim] = false;
	for (int i = 0, iend = data.rows(); i < iend; ++i) {
		if (data(i, split_dim) <= split_val) {
			right_ex_usable[i] = false;
		} else {
			left_ex_usable[i] = false;
		}
	}
	left = new dtree_classifier(depth+1);
	right = new dtree_classifier(depth+1);
	left->learn_rec(data, classes, sorted_dims, left_ex_usable, dim_usable_next);
	right->learn_rec(data, classes, sorted_dims, right_ex_usable, dim_usable_next);
}

class dim_sorter {
public:
	dim_sorter(const mat &data, int dim)
	: data(data), dim(dim)
	{}

	bool operator()(int a, int b) const {
		return data(a, dim) < data(b, dim);
	}

private:
	const mat &data;
	int dim;
};

void dtree_classifier::learn(mat &data, const vector<int> &classes) {
	vector<vector<int> > sorted_dims(data.cols());
	for (int i = 0, iend = data.cols(); i < iend; ++i) {
		sorted_dims[i].resize(data.rows());
		for (int j = 0, jend = data.rows(); j < jend; ++j) {
			sorted_dims[i][j] = j;
		}
		sort(sorted_dims[i].begin(), sorted_dims[i].end(), dim_sorter(data, i));
	}

	vector<bool> ex_usable(data.rows(), true);
	vector<bool> dim_usable(data.cols(), true);

	learn_rec(data, classes, sorted_dims, ex_usable, dim_usable);
}

int dtree_classifier::classify(const rvec &x) const {
	if (split_dim < 0) {
		return cls;
	}
	if (x(split_dim) <= split_val) {
		return left->classify(x);
	}
	return right->classify(x);
}

void dtree_classifier::print(ostream &os) const {
	for (int i = 0; i < depth; ++i) {
		os << ' ';
	}
	if (cls >= 0) {
		os << cls << endl;
		return;
	}
	os << split_dim << " <= " << split_val;
	if (left->cls >= 0) {
		os << " : " << left->cls << endl;
	} else {
		os << endl;
		left->print(os);
	}
	for (int i = 0; i < depth; ++i) {
		os << ' ';
	}
	os << split_dim << " >  " << split_val;
	if (right->cls >= 0) {
		os << " : " << right->cls << endl;
	} else {
		os << endl;
		right->print(os);
	}
}

void dtree_classifier::inspect(ostream &os) const {
	print(os);
}

void dtree_classifier::serialize(ostream &os) const {
	serializer(os) << depth << split_dim << split_val << cls << left << right;
}

void dtree_classifier::unserialize(istream &is) {
	unserializer(is) >> depth >> split_dim >> split_val >> cls >> left >> right;
	assert((cls >= 0 && split_dim < 0 && !left && !right) || (cls < 0 && split_dim >= 0 && left && right));
}

numeric_classifier *make_numeric_classifier(const string &type) {
	if (type == "lda") {
		return new LDA;
	} else if (type == "dtree") {
		return new dtree_classifier;
	} else if (type == "sign") {
		return new sign_classifier;
	} else if (type == "none") {
		return NULL;
	}
	assert(false);
	return NULL;
}
