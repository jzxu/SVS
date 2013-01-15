#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <list>
#include "common.h"
#include "mat.h"
#include "relation.h"
#include "scene_sig.h"
#include "soar_interface.h"

class svs_state;

class model : public serializable {
public:
	model(const std::string &name, const std::string &type);
	virtual ~model() {}
	
	bool cli_inspect(int first_arg, const std::vector<std::string> &args, std::ostream &os);
	
	std::string get_name() const { return name; }
	std::string get_type() const { return type; }
	
	virtual bool predict(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, rvec &y) = 0;
	virtual int get_input_size() const = 0;
	virtual int get_output_size() const = 0;
	virtual void learn(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, const rvec &y) {}
	virtual void test(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, rvec &y);
	virtual void set_wm_root(Symbol *r) {}
	void serialize(std::ostream &os) const {}
	void unserialize(std::istream &is) {}
	
protected:
	virtual bool cli_inspect_sub(int first_arg, const std::vector<std::string> &args, std::ostream &os) {
		return false;
	};

private:
	std::string name, type;
	std::ofstream predlog;
};

/*
 This class keeps track of how to combine several distinct models to make
 a single prediction.  Its main responsibility is to map the values from
 a single scene vector to the vectors that the individual models expect
 as input, and then map the values of the output vectors from individual
 models back into a single output vector for the entire scene. The mapping
 is specified by the Soar agent at runtime using the assign-model command.
*/
class multi_model {
public:
	typedef std::vector<std::pair<std::string, std::string> > prop_vec;
	
	multi_model(std::map<std::string, model*> *model_db);
	~multi_model();
	
	bool predict(const scene_sig &sig, const relation_table &rels, const rvec &x, rvec &y);
	void learn(const scene_sig &sig, const relation_table &rels, const rvec &x, const rvec &y);
	void test(const scene_sig &sig, const relation_table &rels, const rvec &x, const rvec &y);
	void unassign_model(const std::string &name);
	
	std::string assign_model (
		const std::string &name, 
		const prop_vec &inputs, bool all_inputs,
		const prop_vec &outputs);

	
	bool cli_inspect(int first_arg, const std::vector<std::string> &args, std::ostream &os) const;

private:
	struct model_config {
		std::string name;
		prop_vec xprops;
		prop_vec yprops;
		bool allx;
		model *mdl;
	};
	
	struct test_info {
		scene_sig sig;
		rvec x;
		rvec y;
		rvec pred;
		rvec error;
	};
	
	bool predict_or_test(bool test, const scene_sig &sig, const relation_table &rels, const rvec &x, rvec &y);
	void report_model_config(model_config* c, std::ostream &os) const;
	bool report_error(int i, const std::vector<std::string> &args, std::ostream &os) const;
	
	std::list<model_config*>       active_models;
	std::map<std::string, model*> *model_db;
	
	// record prediction errors
	std::vector<test_info> tests;
};

model *make_model(soar_interface *si, Symbol *root, svs_state *state, const std::string &name, const std::string &type);

#endif