#include <iostream>
#include "common.h"
#include "mat.h"
#include "linear.h"

using namespace std;

void parse_text(istream &is, mat &X, cvec &y) {
	string line;
	vector<rvec> buffer;
	rvec r;
	int ncol = 0;

	while (getline(is, line)) {
		vector<string> fields;
		split(line, " \t", fields);
		if (fields.size() == 0)
			continue;

		if (ncol != 0 && fields.size() != ncol) {
			cerr << "mismatched row lengths" << endl;
			exit(1);
		} else {
			ncol = fields.size();
		}
		r.resize(ncol);
		for (int i = 0, iend = fields.size(); i < iend; ++i) {
			if (!parse_double(fields[i], r(i))) {
				cerr << "invalid double " << fields[i] << endl;
				exit(1);
			}
		}
		buffer.push_back(r);
	}

	X.resize(buffer.size(), ncol - 1);
	y.resize(buffer.size());

	for (int i = 0, iend = buffer.size(); i < iend; ++i) {
		X.row(i) = buffer[i].segment(0, ncol - 1);
		y(i) = buffer[i](ncol-1);
	}
}

void print_function(const cvec &coefs, double intercept) {
	bool first = true;
	for (int i = 0, iend = coefs.size(); i < iend; ++i) {
		if (coefs(i) != 0.0) {
			if (first) {
				cout << coefs(i);
				first = false;
			} else if (coefs(i) < 0) {
				cout << " - " << -coefs(i);
			} else {
				cout << " + " << coefs(i);
			}
			cout << " x" << i;
		}
	}
	if (intercept < 0) {
		cout << " - " << -intercept;
	} else if (intercept > 0) {
		cout << " + " << intercept;
	}
}

double test_error(const mat &X, const cvec &y, const cvec &coefs, double intercept) {
	return (y - ((X * coefs).array() + intercept).matrix()).norm();
}

int main(int argc, char *argv[]) {
	regression_type type;
	mat X;
	cvec y, coefs;
	double intercept, variance = 1e-15;
	ifstream testfile;

	mat y1, coefs1;
	rvec intercept1;

	int a = 1;

	while (a < argc) {
		if (strcmp(argv[a], "-v") == 0) {
			if (++a >= argc || !parse_double(argv[a], variance)) {
				cerr << "invalid variance" << endl;
				exit(1);
			}
		} else if (strcmp(argv[a], "-t") == 0) {
			if (++a >= argc) {
				cerr << "specify test file" << endl;
				exit(1);
			}
			testfile.open(argv[a]);
			if (!testfile) {
				cerr << "error opening " << argv[a] << endl;
				exit(1);
			}
		} else {
			break;
		}
		++a;
	}
	
	if (a >= argc) {
		cerr << "specify algorithm: ols|ridge|lasso|forward" << endl;
		exit(1);
	}
	if (strcmp(argv[a], "ols") == 0) {
		type = OLS;
	} else if (strcmp(argv[a], "ridge") == 0) {
		type = RIDGE;
	} else if (strcmp(argv[a], "lasso") == 0) {
		type = LASSO;
	} else if (strcmp(argv[a], "forward") == 0) {
		type = FORWARD;
	} else {
		cerr << "no such algorithm" << endl;
		exit(1);
	}

	if (++a < argc) {
		ifstream f(argv[a]);
		if (!f) {
			cerr << "error opening " << argv[a] << endl;
			exit(1);
		}
		parse_text(f, X, y);
	} else {
		parse_text(cin, X, y);
	}
	
	y1.resize(y.size(), 1);
	y1.col(0) = y;

	if (!linreg(type, X, y1, cvec(), variance, false, coefs1, intercept1)) {
		cerr << "regression failed" << endl;
		exit(1);
	}
	coefs = coefs1.col(0);
	intercept = intercept1(0);

	print_function(coefs, intercept);
	cout << endl;

	double error;
	if (testfile.is_open()) {
		mat Xtest;
		cvec ytest;
		parse_text(testfile, Xtest, ytest);
		if (Xtest.cols() != coefs.size()) {
			cerr << "test data dim mismatch" << endl;
			exit(1);
		}
		error = test_error(Xtest, ytest, coefs, intercept);
	} else {
		error = test_error(X, y, coefs, intercept);
	}
	cout << "error: " << error << endl;
	return 0;
}

