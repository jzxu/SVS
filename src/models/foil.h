#ifndef FOIL_H
#define FOIL_H

#include <vector>
#include "common.h"
#include "serializable.h"
#include "relation.h"

class logger_set;

class literal : public serializable {
public:
	literal() : negate(false) {}
	
	literal(const std::string &name, const int_tuple &args, bool negate)
	: name(name), args(args), negate(negate)
	{}
	
	literal(const literal &l)
	: name(l.name), args(l.args), negate(l.negate)
	{}
	
	literal &operator=(const literal &l) {
		name = l.name;
		args = l.args;
		negate = l.negate;
		return *this;
	}
	
	bool operator==(const literal &l) const { return name == l.name && args == l.args && negate == l.negate; }
	
	int new_vars() const;
	const std::string &get_name() const { return name; }
	const int_tuple &get_args() const { return args; }
	bool negated() const { return negate; }
	
	void set_arg(int i, int v) { args[i] = v; }

	int operator<<(const std::string &s);
	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
	
private:
	std::string name;
	int_tuple args;
	bool negate;

	friend std::ostream &operator<<(std::ostream &os, const literal &l);
};

std::ostream &operator<<(std::ostream &os, const literal &l);

typedef std::vector<literal> clause;

std::ostream &operator<<(std::ostream &os, const clause &c);

class FOIL_result_clause : public serializable {
public:
	clause cl;
	relation true_positives;
	relation false_positives;

	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);
};

class FOIL_result : public serializable {
public:
	std::vector<FOIL_result_clause> clauses;
	relation true_negatives;
	relation false_negatives;
	bool default_class;

	void serialize(std::ostream &os) const;
	void unserialize(std::istream &is);

	void inspect(std::ostream &os) const;
	void inspect_detailed(std::ostream &os) const;
};

bool run_FOIL(const relation &pos, const relation &neg, const relation_table &rels, bool prune, bool track_training, logger_set *loggers, FOIL_result &result);
double FOIL_success_rate(const FOIL_result &result);

class CSP_node;
typedef std::map<int, std::set<int> > var_domains;

/*
 This class holds a CSP_node that is created from the clause and relations so
 that it doesn't have to be reconstructed each time you solve the same CSP with
 a different domain.
*/
class CSP {
public:
	CSP(const clause &c, const relation_table &rels);
	~CSP();
	bool solve(var_domains &domains) const;
	bool solve(int_tuple &domains) const;

private:
	CSP_node *master;
};

void clause_success_rate(const clause &c, const relation &pos, const relation &neg, const relation_table &rels, double &success_rate, double &fp_rate, double &fn_rate);

#endif
