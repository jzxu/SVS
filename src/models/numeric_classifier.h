#ifndef NUMERIC_CLASSIFIER_H
#define NUMERIC_CLASSIFIER_H

#include <iostream>
#include <vector>
#include "serializable.h"
#include "mat.h"

class numeric_classifier : public serializable {
public:
	virtual ~numeric_classifier() {}
	
	virtual void learn(mat &data, const std::vector<int> &classes) = 0; // data matrix can be modified to avoid copying
	virtual int classify(const rvec &x) const = 0;
	virtual void inspect(std::ostream &os) const = 0;
};

numeric_classifier *make_numeric_classifier(const std::string &type);

#endif

