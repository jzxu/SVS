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
	double calc_error(int target, const scene_sig &sig, const rvec &x, double y, double noise_var, std::vector<int> &best_assign) const;
	bool update_fits(double noise_var);
	
	void set_params(const scene_sig &dsig, int target, const rvec &coefs, double inter);
	bool unifiable(int sig, int target) const;

	const interval_set &get_members() const { return members; }
	int size() const { return members.size(); }
	bool is_new_fit() const { return new_fit; }
	bool is_manual() const { return manual; }
	void reset_new_fit() { new_fit = false; }
	
	int num_roles() const { return roles.size(); }

	bool map_roles(int target, const scene_sig &dsig, const relation_table &rels, std::vector<int> &mapping) const;
	
	int get_num_nonzero_coefs() const;
	
	/* return a string representation of the linear function */
	void get_function_string(std::string &s) const;
	
	void update_role_classifiers() const;
	
private:
	interval_set members;

	/*
	 Maintain a set of maps from the mode's roles to objects in each example.
	 Since many examples will share the same role map, keep only unique maps and
	 keep track of which member examples use which maps. The role map is stored as
	 a vector m such that m[i] = j, where i is an index into the mode's signature,
	 and j is an index into the instance's signature.
	*/
	class role_map_entry : public serializable {
	public:
		std::vector<int> role_map;
		interval_set     members;  // members with this object map
		
		void serialize(std::ostream &os) const;
		void unserialize(std::istream &is);
	};
	std::vector<role_map_entry> role_maps;
	
	bool stale, noise, new_fit, manual;
	mutable bool role_classifiers_stale;
	const model_train_data &data;
	double intercept;
	int n_nonzero;

	class role : public serializable {
	public:
		std::string type;
		std::vector<std::string> properties;
		rvec coefficients;
		FOIL_result classifier;

		void serialize(std::ostream &os) const;
		void unserialize(std::istream &is);
	};

	std::vector<role> roles;
	
	/*
	 Noise data sorted by their Y values. First element in pair is the Y value,
	 second is the index.
	*/
	std::set<std::pair<double, int> > sorted_ys;
	
	
	logger_set *loggers;

	void proxy_get_children(std::map<std::string, cliproxy*> &c);
	void proxy_use_sub(const std::vector<std::string> &args, std::ostream &os);
	void cli_clauses(const std::vector<std::string> &args, std::ostream &os) const;
	void cli_members(std::ostream &os) const;
};

#endif
