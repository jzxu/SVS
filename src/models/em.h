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
		double error;               // residual error of using this mode to predict this data point
		bool error_stale;           // does error need to be update?
		std::vector<int> role_map;  // map from objects in this instance to mode roles
	
		mode_info() : error(INF), error_stale(true) {}
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
	EM(const model_train_data &data, logger_set *loggers, bool use_unify, bool learn_new_modes);
	~EM();
	
	void add_data(int t);
	bool run(int maxiters);
	int  predict(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, int mode, double &y, rvec &vote_trace) const;
	void all_predictions(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, rvec &preds) const;
	
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
	
	// for test_em.cpp
	void print_error_table(std::ostream &os) const;
	void print_modes(std::ostream &os) const;
	void set_noise_var(double v) { noise_var = v; }
	
	int num_modes() const { return modes.size(); }
	
	void get_mode_function_string(int m, std::string &s) const;

	void proxy_use_sub(const std::vector<std::string> &args, std::ostream &os);
private:
	struct unify_result {
		bool success;
		cvec coefs;
		double intercept;
		int num_ex;          // num. examples covered, old and new
		int num_uncovered;   // num. examples covered by old mode and not by new mode
		int num_new_covered; // num. noise examples covered by new mode
		int num_coefs;       // num. non-zero coefficients
	};

	void estep();
	bool mstep();
	void fill_xy(const interval_set &rows, mat &X, mat &Y) const;
	void unify(const em_mode *m, const std::vector<int> &new_ex, int sig, int target, unify_result &result) const;
	bool unify_or_add_mode();
	em_mode *add_mode(bool manual);
	bool remove_modes();

	int classify(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, std::vector<int> &obj_map, rvec &vote_trace) const;
	
	void proxy_get_children(std::map<std::string, cliproxy*> &c);
	void cli_error_table(std::ostream &os) const;
	void cli_add_mode(const std::vector<std::string> &args, std::ostream &os);
	void cli_update_classifiers(std::ostream &os);

	typedef std::map<const scene_sig*,sig_info*> sig_table;
	const model_train_data &data;
	std::vector<inst_info*> insts;
	std::vector<em_mode*> modes;
	sig_table sigs;
	classifier clsfr;
	
	bool use_unify, learn_new_modes;

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
