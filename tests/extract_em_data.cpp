#include <iostream>
#include <fstream>
#include "common.h"
#include "model.h"
#include "logger.h"

#define private public
#include "em.h"

using namespace std;

void center_x(const rvec &x, const scene_sig &sig, int target, rvec &cx) {
	cx = x;
	rvec tpos = x.segment(sig[target].start, 3);
	for (int i = 1, iend = sig.size(); i < iend; ++i) { // start at 1 to skip world object
		if (sig[i].id >= 0) {  // not output
			cx.segment(sig[i].start, 3) -= tpos;
		}
	}
}

model_train_data *read_data(const char *path) {
	model_train_data *data = new model_train_data;
	ifstream input(path);
	string line;

	// first line contains model info
	getline(input, line);
	cerr << line << endl;

	data->unserialize(input);
	cerr << data->size() << endl;
	return data;
}

void add_modes_old(EM &em) {
	vector<string> mode_specs(7);

	mode_specs[1] = "1 b1:vx -0.39 b1:vz  1.96e-4";
	mode_specs[2] = "1 b1:vx  0.39 b1:vz -1.96e-4";
	mode_specs[3] = "1 b1:vx";
	mode_specs[4] = "-.95 b1:vx";
	mode_specs[5] = "0";
	mode_specs[6] = "-.17 b1:vx -1.17 b1:vz 5.88e-4";

	for (int i = 1; i < mode_specs.size(); ++i) {
		vector<string> args;
		split(mode_specs[i], " ", args);
		em.cli_add_mode(args, cerr);
	}
	em.proxy_use_sub(vector<string>(), cerr);
}

void add_modes(EM &em) {
	string line;
	while (getline(cin, line)) {
		vector<string> args;
		split(line, " ", args);
		em.cli_add_mode(args, cerr);
	}
	em.proxy_use_sub(vector<string>(), cerr);
}

void get_rels_at_time(const relation_table &source, int t, relation_table &out) {
	relation_table::const_iterator i, iend;
	int_tuple vals(1, t);
	for (i = source.begin(), iend = source.end(); i != iend; ++i) {
		out[i->first] = i->second;
		out[i->first].filter(0, vals, false);
	}
}

int best_mode(double y, rvec &p) {
	int m;
	p(0) = INFINITY;
	if ((p.array() - y).abs().minCoeff(&m) > 1e-10) {
		return -1;
	}
	assert(m != 0);
	return m;
}

void extract_cls_data(const char *train_file, bool center) {
	model_train_data *data = read_data(train_file);
	logger_set log;
	EM em(*data, &log);
	int errors = 0;
	int xcols;

	if (data->size() == 0) {
		xcols = 0;
	} else {
		xcols = data->get_inst(0).x.size() - 9;
	}

	em.add_data(0);
	add_modes(em);

	cout << "% x(" << xcols << ") mode" << endl;
	for (int i = 0, iend = data->size(); i < iend; ++i) {
		const model_train_inst &inst = data->get_inst(i);
		rvec preds, cx;
		relation_table rels;
		int mode;
		
		get_rels_at_time(data->get_all_rels(), i, rels);
		em.all_predictions(inst.target, *inst.sig, rels, inst.x, preds);
		mode = best_mode(inst.y(0), preds);

		if (center) {
			center_x(inst.x, *inst.sig, inst.target, cx);
			for (int j = 9, jend = cx.size(); j < jend; ++j) {
				cout << cx(j) << " ";
			}
		} else {
			for (int j = 9, jend = inst.x.size(); j < jend; ++j) {
				cout << inst.x(j) << " ";
			}
		}
		cout << mode << endl;

		if (mode == -1) {
			errors++;
		}
	}
	cerr << errors << " errors" << endl;
}

void extract_pred_data(const char *train_file, bool center) {
	model_train_data *data = read_data(train_file);
	int xcols;

	if (data->size() == 0) {
		xcols = 0;
	} else {
		xcols = data->get_inst(0).x.size() - 9;
	}
	cout << "% x(" << xcols << ") y" << endl;
	for (int i = 0, iend = data->size(); i < iend; ++i) {
		const model_train_inst &inst = data->get_inst(i);
		rvec preds, cx;
		
		if (center) {
			center_x(inst.x, *inst.sig, inst.target, cx);
			for (int j = 9, jend = cx.size(); j < jend; ++j) {
				cout << cx(j) << " ";
			}
		} else {
			for (int j = 9, jend = inst.x.size(); j < jend; ++j) {
				cout << inst.x(j) << " ";
			}
		}
		cout << inst.y(0) << endl;
	}
}

void usage() {
	cerr << "usage: [-c] cls|pred FILE" << endl;
	exit(1);
}

int main(int argc, char *argv[]) {
	if (argc < 2) {
		usage();
	}
	
	int i = 1;
	bool center = false;
	if (string(argv[i]) == "-c") {
		center = true;
		++i;
	}

	if (i + 2 > argc) {
		usage();
	}

	if (string(argv[i]) == "cls") {
		extract_cls_data(argv[i+1], center);
	} else if (string(argv[i]) == "pred") {
		extract_pred_data(argv[i+1], center);
	} else {
		usage();
	}

	return 0;
}

