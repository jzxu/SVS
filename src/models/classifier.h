#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include "relation.h"
#include "foil.h"
#include "serializable.h"
#include "scene_sig.h"
#include "model.h"
#include "cliproxy.h"
#include "timer.h"
#include "numeric_classifier.h"

class logger_set;

class clause_info : public serializable {
public:
	clause_info() : nc(NULL), success_rate(0.0) {}
	~clause_info() { if (nc) { delete nc; } }
	
	clause cl;
	relation false_pos;
	relation true_pos;
	numeric_classifier *nc;
	double success_rate;

	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
};

class binary_classifier : public serializable, public cliproxy {
public:
	binary_classifier();
	binary_classifier(logger_set *loggers);
	~binary_classifier();

	int vote(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, int &clause_num, bool &used_nc) const;
	void update(const relation &pos, const relation &neg, const relation_table &rels, const model_train_data &train_data, bool use_foil, bool prune, const std::string &nc_type);
	double get_success_rate() const;
	
	void inspect(std::ostream &os) const;
	void inspect_detailed(std::ostream &os) const;
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);

	void set_loggers(logger_set *loggers) { this->loggers = loggers; }
private:
	std::vector<clause_info> clauses;
	
	relation false_negatives, true_negatives;
	numeric_classifier *neg_nc;
	double neg_success_rate;

	mutable timer_set timers;
	logger_set *loggers;
};

/*
 Uses one-to-one voting to extend binary classification to multiple classes.
*/
class classifier : public serializable, public cliproxy {
public:
	classifier(const model_train_data &data, logger_set *loggers);
	~classifier();
	
	void add_class();
	void del_classes(const std::vector<int> &c);
	
	void update_class(int i, int old_class, int new_class);
	void classify(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, std::vector<int> &votes, rvec &trace) const;
	void update();

	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
	
private:
	class pair_info : public serializable {
	public:
		int cls_i, cls_j;
		binary_classifier *clsfr;
		bool negated;
		
		pair_info() : cls_i(0), cls_j(0), clsfr(NULL), negated(false) {}
		pair_info(int i, int j) : cls_i(i), cls_j(j), clsfr(NULL), negated(false) {}
		~pair_info() { delete clsfr; }
		void serialize(std::ostream &os) const;
		void unserialize(std::istream &is);
	};
	
	class class_info : public serializable {
	public:
		relation mem_rel;
		bool stale;        // has membership changed?
		
		class_info() : mem_rel(2), stale(false) {}
		
		void serialize(std::ostream &os) const;
		void unserialize(std::istream &os);
	};
	
	const model_train_data &data;
	std::list<pair_info*> pairs;
	std::vector<class_info*> classes;
	
	bool foil, prune, context;
	std::string nc_type;
	
	// Option values since last classifier update.
	// If they're different from the current values, force an update.
	bool old_foil, old_prune, old_context;
	std::string old_nc_type;
	
	logger_set *loggers;

	pair_info *find(int i, int j);
	
	void proxy_get_children(std::map<std::string, cliproxy*> &c);
	void proxy_use_sub(const std::vector<std::string> &args, std::ostream &os);
	void cli_nc_type(const std::vector<std::string> &args, std::ostream &os);
};

#endif
