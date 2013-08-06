#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <vector>
#include <map>
#include "common.h"
#include "mat.h"
#include "numeric_classifier.h"
#include "serialize.h"

using namespace std;

void read_data(const char *path, mat &X, vector<int> &classes);
void run_print(int first, int argc, char *argv[]);
void run_test_set(int first, int argc, char *argv[]);
void run_cross_validation(int first, int argc, char *argv[]);
void run_serialize(int first, int argc, char *argv[]);

bool input_serialized = false;
char *classifier_type = NULL;

int main(int argc, char *argv[]) {
	int i = 1;
	
	if (argc < 3) {
		cerr << "usage: " << argv[0] << " [options] [type] [operation]" << endl;
		cerr << "options = -s (read serialized data)" << endl;
		cerr << "type = lda | sign | dtree" << endl;
		cerr << "operation = print, test_set, cv, serialize" << endl;
		exit(1);
	}
	
	while (i < argc && argv[i][0] == '-') {
		switch (argv[i][1]) {
			case 's':
				input_serialized = true;
				break;
			default:
				cerr << "unrecognized option " << argv[i] << endl;
				exit(1);
				break;
		}
		++i;
	}
	
	classifier_type = argv[i++];
	
	if (strcmp(argv[i], "print") == 0) {
		run_print(++i, argc, argv);
	} else if (strcmp(argv[i], "test_set") == 0) {
		run_test_set(++i, argc, argv);
	} else if (strcmp(argv[i], "cv") == 0) {
		run_cross_validation(++i, argc, argv);
	} else if (strcmp(argv[i], "serialize") == 0) {
		run_serialize(++i, argc, argv);
	} else {
		cerr << "no such operation" << endl;
		return 1;
	}
	
	return 0;
}

void read_data(const char *path, mat &X, vector<int> &classes) {
	string line;
	vector<string> fields;
	vector<vector<double> > data;
	double x;
	
	ifstream input(path);
	
	assert(input);
	
	if (input_serialized) {
		unserialize(X, input);
		unserialize(classes, input);
		return;
	}
	
	while(getline(input, line)) {
		fields.clear();
		split(line, "", fields);
		if (fields.empty()) {
			continue;
		}
		
		grow_vec(data);
		for (int i = 0; i < fields.size(); ++i) {
			if (!parse_double(fields[i], x)) {
				cerr << "non number \"" << fields[i] << "\"" << endl;;
				exit(1);
			}
			data.back().push_back(x);
		}
	}
	X.resize(data.size(), data[0].size() - 1);
	for (int i = 0; i < data.size(); ++i) {
		assert(data[i].size() - 1 == X.cols());
		for (int j = 0; j < data[i].size() - 1; ++j) {
			X(i, j) = data[i][j];
		}
		classes.push_back(static_cast<int>(data[i].back()));
	}
}

void run_print(int first, int argc, char *argv[]) {
	mat data;
	vector<int> classes;
	
	if (first >= argc) {
		cerr << "specify training file" << endl;
		exit(1);
	}
	
	numeric_classifier *cls = make_numeric_classifier(classifier_type);
	read_data(argv[first], data, classes);
	cls->learn(data, classes);
	cls->inspect(cout);
	delete cls;
}

void run_test_set(int first, int argc, char *argv[]) {
	string line;
	vector<string> fields;
	vector<int> train_classes, test_classes;
	mat Xtrain, Xtest;
	
	if (first + 1 >= argc) {
		cerr << "usage: <training file> <test files>" << endl;
		exit(1);
	}
	
	numeric_classifier *cls = make_numeric_classifier(classifier_type);
	read_data(argv[first], Xtrain, train_classes);
	read_data(argv[first+1], Xtest, test_classes);
	
	cls->learn(Xtrain, train_classes);
	
	int correct = 0;
	for (int i = 0; i < Xtest.rows(); ++i) {
		int pred = cls->classify(Xtest.row(i));
		if (pred == test_classes[i])
			++correct;
	}
	cout << correct << " correct out of " << Xtest.rows() << endl;
	delete cls;
}

void run_cross_validation(int first, int argc, char *argv[]) {
	mat data, train;
	vector<int> classes, train_classes, reorder;
	int n, k, chunksize, extra, start, ndata, ncols, correct;
	
	if (first >= argc) {
		cerr << "usage: <training file> [n]" << endl;
		exit(1);
	}
	
	read_data(argv[first], data, classes);
	ndata = data.rows();
	ncols = data.cols();
	
	if (first + 2 < argc) {
		if (!parse_int(argv[first+2], n)) {
			cerr << "invalid n" << endl;
			exit(1);
		}
	} else {
		n = 10;
	}
	
	reorder.resize(ndata);
	for (int i = 0, iend = ndata; i < iend; ++i) {
		reorder[i] = i;
	}
	random_shuffle(reorder.begin(), reorder.end());
	chunksize = data.rows() / n;
	extra = data.rows() - chunksize * n;
	correct = 0;
	start = 0;
	for (int i = 0; i < n; ++i) {
		if (i < extra) {
			k = chunksize + 1;
		} else {
			k = chunksize;
		}
		train.resize(ndata - k, ncols);
		train_classes.resize(ndata - k);
		
		for (int j = 0; j < ndata - k; ++j) {
			if (j < start) {
				train.row(j) = data.row(reorder[j]);
				train_classes[j] = classes[reorder[j]];
			} else {
				train.row(j) = data.row(reorder[j + k]);
				train_classes[j] = classes[reorder[j + k]];
			}
		}

		numeric_classifier *cls = make_numeric_classifier(classifier_type);
		cls->learn(train, train_classes);
		
		for (int j = 0; j < k; ++j) {
			int a = cls->classify(data.row(reorder[start + j]));
			if (a == classes[reorder[start + j]]) {
				correct++;
			}
		}
		delete cls;
		start += k;
	}
	
	cout << correct << " correct out of " << ndata << endl;
}

void run_serialize(int first, int argc, char *argv[]) {
	mat data;
	vector<int> classes;
	
	if (first + 1 >= argc) {
		cerr << "specify training file and output file" << endl;
		exit(1);
	}
	
	numeric_classifier *cls = make_numeric_classifier(classifier_type);
	read_data(argv[first+1], data, classes);
	cls->learn(data, classes);
	
	ofstream out(argv[first + 2]);
	cls->serialize(out);
	out.close();
	
	ifstream input(argv[first+1]);
	cls->unserialize(input);
	cls->inspect(cout);
	
	delete cls;
}
