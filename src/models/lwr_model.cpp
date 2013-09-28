#include <iostream>
#include "lwr.h"
#include "svs.h"
#include "model.h"

using namespace std;
using namespace Eigen;

/*
 Center x on the target
*/
void center_x(const rvec &x, const scene_sig &sig, int target, rvec &cx) {
	cx = x;
	rvec tpos = x.segment(sig[target].start, 3);
	for (int i = 1, iend = sig.size(); i < iend; ++i) { // start at 1 to skip world object
		if (sig[i].id >= 0) {  // not output
			cx.segment(sig[i].start, 3) -= tpos;
		}
	}
}

class lwr_model : public model {
public:
	lwr_model(const string &name)
	: model(name, "lwr"), lwr(true)
	{}
	
	void update() {
		const model_train_inst &i = get_data().get_last_inst();
		rvec cx;
		center_x(i.x, *i.sig, i.target, cx);
		lwr.learn(i.x, cx, i.y);
	}
	
	bool predict_sub(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, bool test, rvec &y, map<string,rvec> &info) {
		rvec cx, neighbors, dists, coefs;
		bool success;
		
		center_x(x, sig, target, cx);
		success = lwr.predict(x, cx, y, neighbors, dists, coefs);
		if (success) {
			info["centered" ] = cx;
			info["neighbors"] = neighbors;
			info["distances"] = dists;
			info["coefs"]     = coefs;
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

	void proxy_get_children(map<string, cliproxy*> &c) {
		model::proxy_get_children(c);
		c["lwr"] = &lwr;
	}
};

model *_make_lwr_model_(svs *owner, const string &name) {
	return new lwr_model(name);
}

