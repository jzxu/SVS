#include <iostream>
#include "lwr.h"
#include "svs.h"
#include "model.h"

using namespace std;
using namespace Eigen;

class lwr_model : public model {
public:
	lwr_model(int nnbrs, double noise_var, const string &name)
	: model(name, "lwr"), lwr(nnbrs, noise_var, true)
	{}
	
	void update() {
		const model_train_inst &i = get_data().get_last_inst();
		lwr.learn(i.x, i.y);
	}
	
	bool predict_sub(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, bool test, rvec &y, map<string,rvec> &info) {
		rvec centered, neighbors, dists, coefs, intercept;
		bool success;
		
		success = lwr.predict(x, y, neighbors, dists, coefs, intercept);
		if (success) {
			info["neighbors"] = neighbors;
			info["distances"] = dists;
			info["coefs"]     = coefs;
			info["intercept"] = intercept;
		}
		return success;
	}
	
	int size() const {
		return lwr.size();
	}
	
	void unserialize_sub(istream &is) {
		lwr.unserialize(is);
	}
	
	void serialize_sub(ostream &os) const {
		lwr.serialize(os);
	}
	
	int get_input_size() const {
		return lwr.xsize();
	}
	
	int get_output_size() const {
		return lwr.ysize();
	}
	
private:
	LWR lwr;
};

model *_make_lwr_model_(svs *owner, const string &name) {
	Symbol *attr;
	wme *nnbrs_wme = NULL, *var_wme = NULL;
	long nnbrs = 50;
	double noise_var = 1e-8;
	string attrstr;
	
	return new lwr_model(nnbrs, noise_var, name);
}
