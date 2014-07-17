#ifndef EM_H
#define EM_H

#include <set>
#include <vector>
#include "linear.h"
#include "common.h"
#include "timer.h"
#include "foil.h"
#include "serializable.h"
#include "relation.h"
#include "mat.h"
#include "scene_sig.h"
#include "lda.h"
#include "classifier.h"
#include "mode.h"
#include "cliproxy.h"

class logger_set;

class inst_info : public serializable {
public:
	/*
	 Holds information about how a training data point relates to each mode
	*/
	class mode_info : public serializable {
	public:
		double prob;                // probability that data point belongs to mode
		bool prob_stale;            // does prob need to be update?
		std::vector<int> sig_map;   // map from objects in this instance to variables in mode sig
	
		mode_info() : prob(0), prob_stale(true) {}
		void serialize(std::ostream &os) const;
		void unserialize(std::istream &is);
	};

	int mode;
	std::vector<mode_info> minfo;
	
	inst_info() : mode(0) {}
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
};

class sig_info : public serializable {
public:
	std::vector<int> members;  // indexes of data points with this sig
	std::set<int> noise;       // indexes of noise data with this signature

	sig_info();
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
};

class EM : public serializable, public cliproxy {
public:
	EM(const model_train_data &data, logger_set *loggers);
	EM(const model_train_data &data, logger_set *loggers, bool use_em, bool use_unify, bool learn_new_modes);
	~EM();
	
	void add_data(int t);
	bool run(int maxiters);
	bool predict(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, int &mode, double &y);
	// Return the mode with the model that best fits (x, y)
	int best_mode(int target, const scene_sig &sig, const rvec &x, double y, double &besterror) const;
	
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
	
	// for test_em.cpp
	void print_ptable(std::ostream &os) const;
	void print_modes(std::ostream &os) const;
	void set_noise_var(double v) { noise_var = v; }
	
	int num_modes() const { return modes.size(); }
	
	void get_mode_function_string(int m, std::string &s) const;

	void proxy_use_sub(const std::vector<std::string> &args, std::ostream &os);
private:
	void estep();
	bool mstep();
	void fill_xy(const interval_set &rows, mat &X, mat &Y) const;

	bool unify_or_add_mode();
	int find_linear_subset(mat &X, mat &Y, std::vector<int> &subset, mat &coefs, rvec &inter) const;
	void find_linear_subset_em(const_mat_view X, const_mat_view Y, std::vector<int> &subset) const;
	void find_linear_subset_block(const_mat_view X, const_mat_view Y, std::vector<int> &subset) const;
	em_mode *add_mode(bool manual);
	bool remove_modes();

	int classify(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, std::vector<int> &obj_map);
	
	void proxy_get_children(std::map<std::string, cliproxy*> &c);
	void cli_ptable(std::ostream &os) const;
	void cli_add_mode(const std::vector<std::string> &args, std::ostream &os);

	typedef std::map<const scene_sig*,sig_info*> sig_table;
	const model_train_data &data;
	std::vector<inst_info*> insts;
	std::vector<em_mode*> modes;
	sig_table sigs;
	classifier clsfr;
	
	bool use_em, use_unify, learn_new_modes;

	/*
	 Keeps track of the minimum number of new noise examples needed before we have
	 to check for a possible new mode
	*/
	int check_after;
	int nc_type;
	
	double noise_var;
	mutable timer_set timers;
	logger_set *loggers;
};


#endif
