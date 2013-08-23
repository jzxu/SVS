#ifndef MODE_H
#define MODE_H

#include <set>
#include <vector>
#include <utility>
#include "mat.h"
#include "relation.h"
#include "scene_sig.h"
#include "foil.h"
#include "model.h"
#include "cliproxy.h"

class logger_set;

class em_mode : public serializable, public cliproxy {
public:
	em_mode(bool noise, bool manual, const model_train_data &data, logger_set *loggers);
	
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
	
	void add_example(int i, const std::vector<int> &obj_map, double noise_var);
	void del_example(int i);
	double predict(const scene_sig &s, const rvec &x, const std::vector<int> &obj_map) const;
	void largest_const_subset(std::vector<int> &subset);
	const std::set<int> &get_noise(int sigindex) const;
	void get_noise_sigs(std::vector<int> &sigs);
	double calc_error(int target, const scene_sig &sig, const rvec &x, double y, double noise_var, std::vector<int> &best_assign) const;
	bool update_fits(double noise_var);
	
	void get_params(scene_sig &sig, rvec &coefs, double &intercepts) const;
	void set_params(const scene_sig &dsig, int target, const rvec &coefs, double inter);
	bool uniform_sig(int sig, int target) const;
	void get_members(interval_set &m) const;

	int size() const { return members.size(); }
	bool is_new_fit() const { return new_fit; }
	bool is_manual() const { return manual; }
	void reset_new_fit() { new_fit = false; }
	
	const scene_sig &get_sig() const { return sig; }

	bool map_objs(int target, const scene_sig &dsig, const relation_table &rels, std::vector<int> &mapping) const;
	
	int get_num_nonzero_coefs() const;
	
	/* return a string representation of the linear function */
	void get_function_string(std::string &s) const;
	
private:
	interval_set members;

	/*
	 For each member instance, there is a mapping from objects tested by the
	 linear function to object indexes in the member's signature, call it the
	 object map. An object map is stored as a vector m such that m[i] = j, where i is an index
	 into the mode's signature, and j is an index into the instance's signature.
	 The omap_table associates unique omaps with sets of instances that share that
	 omap.

	 I'm assuming that the number of unique omaps will be small.
	*/
	class obj_map_entry : public serializable {
	public:
		std::vector<int> obj_map;
		interval_set     members;  // members with this object map
		
		void serialize(std::ostream &os) const;
		void unserialize(std::istream &is);
	};
	std::vector<obj_map_entry> obj_maps;
	
	bool stale, noise, new_fit, manual;
	const model_train_data &data;
	
	rvec coefficients;
	double intercept;
	int n_nonzero;
	scene_sig sig;
	
	/*
	 Noise data sorted by their Y values. First element in pair is the Y value,
	 second is the index.
	*/
	std::set<std::pair<double, int> > sorted_ys;
	
	/*
	 Each object the model is conditioned on needs to be
	 identifiable with a set of first-order Horn clauses
	 learned with FOIL.
	*/
	mutable std::vector<std::vector<clause> > obj_clauses;
	mutable bool obj_clauses_stale;
	
	logger_set *loggers;

	void learn_obj_clauses(const relation_table &rels) const;
	
	void proxy_get_children(std::map<std::string, cliproxy*> &c);
	void proxy_use_sub(const std::vector<std::string> &args, std::ostream &os);
	void cli_clauses(std::ostream &os) const;
	void cli_members(std::ostream &os) const;
	void cli_sig(std::ostream &os) const;
};

#endif
