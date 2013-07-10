#include <iostream>
#include "model.h"
#include "svs.h"

using namespace std;

/* Doesn't do anything */
class null_model : public model {
public:
	null_model(const string &name) : model(name, "null") {}
	
	bool predict_sub(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, bool test, rvec &y, map<string,rvec> &info) { return true; }
	int get_input_size() const { return 0; }
	int get_output_size() const { return 0; }
};

model *_make_null_model_(svs *owner, const string &name) {
	return new null_model(name);
}
