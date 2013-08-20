#include "classifier.h"
#include "serialize.h"
#include "common.h"
#include "logger.h"
#include <algorithm>

using namespace std;

void print_first_arg(const relation &r, ostream &os) {
	interval_set first;
	r.at_pos(0, first);
	os << first;
}

void extract_vec(const int_tuple &t, const rvec &x, const scene_sig &sig, rvec &out) {
	vec3 target_pos;
	out.resize(x.size());
	int end = 0, s, n;
	for (int i = 1, iend = t.size(); i < iend; ++i) {
		bool found = false;
		for (int j = 0, jend = sig.size(); j < jend; ++j) {
			if (sig[j].id == t[i]) {
				s = sig[j].start;
				n = sig[j].props.size();
				found = true;
				break;
			}
		}
		assert(found);
		out.segment(end, n) = x.segment(s, n);

		// Center positions on target
		if (i == 1) {
			target_pos = x.segment(s, 3);
		}
		out.segment(end, 3) -= target_pos;

		end += n;
	}
	out.conservativeResize(end);
}

/*
 positive = class 0, negative = class 1

 For now, only learn numeric classifiers on the properties of the target
 object. This is because it's hard to line up the other objects across
 different signatures: consider that we're trying to distinguish between
 different modes here, and different modes have different sets of relevant
 objects. How do we fit these disparate rows continuous properties into a
 single matrix? Therefore, I'm side-stepping the issue and considering only the
 target object.
*/
numeric_classifier *learn_numeric_classifier(const string &type, const relation &pos, const relation &neg, const model_train_data &data, double &success_rate) {
	const double TEST_RATIO = 0.3;
	const int MIN_BLOCK = 30; // the minimum number of members in pos/neg, train/test

	int npos = pos.size(), nneg = neg.size();
	int ndata = npos + nneg;
	
	/*
	 Split into training and test sets. Try to distribute positives and negatives
	 as evenly as possible. Prioritize maintaining a minimum number of test
	 examples, otherwise estimated success rates will be skewed.
	*/
	int npos_test = std::max(static_cast<int>(npos * TEST_RATIO), MIN_BLOCK);
	int nneg_test = std::max(static_cast<int>(nneg * TEST_RATIO), MIN_BLOCK);
	int npos_train = npos - npos_test;
	int nneg_train = nneg - nneg_test;
	int ntest = npos_test + nneg_test;
	int ntrain = npos_train + nneg_train;
	
	if (npos_train < MIN_BLOCK || nneg_train < MIN_BLOCK) {
		return NULL;
	}

	// figure out matrix columns
	rvec xpart;
	int_tuple t = *pos.begin();
	extract_vec(t, data.get_inst(t[0]).x, *data.get_inst(t[0]).sig, xpart);
	int ncols = xpart.size();
	mat allx(ndata, ncols);
	
	relation::const_iterator i, iend;
	int j = 0;
	for (i = pos.begin(), iend = pos.end(); i != iend; ++i, ++j) {
		t = *i;
		const model_train_inst &inst = data.get_inst(t[0]);
		extract_vec(t, inst.x, *inst.sig, xpart);
		assert(xpart.size() == ncols);
		allx.row(j) = xpart;
	}
	
	for (i = neg.begin(), iend = neg.end(); i != iend; ++i, ++j) {
		t = *i;
		const model_train_inst &inst = data.get_inst(t[0]);
		extract_vec(t, inst.x, *inst.sig, xpart);
		assert(xpart.size() == ncols);
		allx.row(j) = xpart;
	}
	
	vector<int> pos_inds(npos), neg_inds(nneg);
	mat trainx(ntrain, ncols), testx(ntest, ncols);
	vector<int> train_classes(ntrain), test_classes(ntest);

	for (int i = 0; i < npos; ++i) {
		pos_inds[i] = i;
	}
	for (int i = 0; i < nneg; ++i) {
		neg_inds[i] = npos + i;
	}
	std::random_shuffle(pos_inds.begin(), pos_inds.end());
	std::random_shuffle(neg_inds.begin(), neg_inds.end());

	/*
	 Beginning part of randomized pos_inds (neg_inds) will be used as positive
	 (negative) training examples, rest will be used as test examples.
	*/
	for (int i = 0; i < npos_train; ++i) {
		assert(i < trainx.rows() && i < train_classes.size() && i < pos_inds.size() && pos_inds[i] < allx.rows());
		trainx.row(i) = allx.row(pos_inds[i]);
		train_classes[i] = 0;
	}
	for (int i = npos_train, j = 0; i < ntrain; ++i, ++j) {
		assert(i < trainx.rows() && i < train_classes.size() && j < neg_inds.size() && neg_inds[j] < allx.rows());
		trainx.row(i) = allx.row(neg_inds[j]);
		train_classes[i] = 1;
	}
	for (int i = 0, j = npos_train; i < npos_test; ++i, ++j) {
		assert(i < testx.rows() && i < test_classes.size() && j < pos_inds.size() && pos_inds[j] < allx.rows());
		testx.row(i) = allx.row(pos_inds[j]);
		test_classes[i] = 0;
	}
	for (int i = npos_test, j = nneg_train; i < ntest; ++i, ++j) {
		assert(i < testx.rows() && i < test_classes.size() && j < neg_inds.size() && neg_inds[j] < allx.rows());
		testx.row(i) = allx.row(neg_inds[j]);
		test_classes[i] = 1;
	}

	numeric_classifier *nc = make_numeric_classifier(type);
	assert(nc);
	nc->learn(trainx, train_classes);

	int nright = 0, nwrong = 0;
	for (int i = 0; i < ntest; ++i) {
		if (nc->classify(testx.row(i)) == test_classes[i]) {
			nright++;
		} else {
			nwrong++;
		}
	}
	success_rate = nright / static_cast<double>(nright + nwrong);

	return nc;
}

binary_classifier::binary_classifier()
: neg_nc(NULL), loggers(NULL), neg_success_rate(0.0)
{}

binary_classifier::binary_classifier(logger_set *loggers)
: neg_nc(NULL), loggers(loggers), neg_success_rate(0.0)
{}

binary_classifier::~binary_classifier() {
	if (neg_nc) {
		delete neg_nc;
	}
}

void binary_classifier::serialize(ostream &os) const {
	serializer(os) << clauses 
	               << false_negatives << true_negatives
				   << neg_success_rate
	               << (neg_nc != NULL);
	if (neg_nc) {
		neg_nc->serialize(os);
	}
}

void binary_classifier::unserialize(istream &is) {
	bool has_negnc;
	unserializer(is) >> clauses 
	                 >> false_negatives >> true_negatives
				     >> neg_success_rate
	                 >> has_negnc;

	if (neg_nc) {
		delete neg_nc;
	}
	neg_nc = has_negnc ? unserialize_numeric_classifier(is) : NULL;
}

void binary_classifier::inspect(ostream &os) const {
	table_printer t;
	t.add_row() << "#" << "clause" << "Correct" << "Incorrect" << "NumCls?";
	for (int i = 0, iend = clauses.size(); i < iend; ++i) {
		t.add_row() << i << clauses[i].cl << clauses[i].true_pos.size() << clauses[i].false_pos.size() << (clauses[i].nc != NULL);
	}
	t.add_row() << '-' << "NEGATIVE" << true_negatives.size() << false_negatives.size() << (neg_nc != NULL);
	t.print(os);
	os << "success rate: " << get_success_rate() << endl;
}

void binary_classifier::inspect_detailed(ostream &os) const {
	if (clauses.empty()) {
		os << "No clauses" << endl;
	} else {
		for (int k = 0; k < clauses.size(); ++k) {
			os << "Clause: " << clauses[k].cl << endl;
			
			os << "True positives: " << endl;
			clauses[k].true_pos.print_condensed(os);
			os << endl << endl;
			
			os << "False positives: ";
			print_first_arg(clauses[k].false_pos, os);
			os << endl << endl;
			
			if (clauses[k].nc) {
				os << "Numeric classifier:" << endl;
				clauses[k].nc->inspect(os);
				os << endl << endl;
			}
		}
	}
	os << "NEGATIVE:" << endl;
	
	os << "True negatives: ";
	print_first_arg(true_negatives, os);
	os << endl << endl;
	
	os << "False negatives: ";
	print_first_arg(false_negatives, os);
	os << endl << endl;
	
	if (neg_nc) {
		os << "Negative numeric classifier:" << endl;
		neg_nc->inspect(os);
		os << endl;
	}
}

/*
 Return 0 to vote for i, 1 to vote for j
*/
int binary_classifier::vote(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, int &matched_clause, bool &used_nc) const {
	function_timer t(timers.get_or_add("vote"));
	
	int result = 1;
	matched_clause = -1;
	used_nc = false;

	if (!clauses.empty()) {
		var_domains domains;
		domains[0].insert(0);       // rels is only for the current timestep, time should always be 0
		domains[1].insert(sig[target].id);
		
		for (int i = 0, iend = clauses.size(); i < iend; ++i) {
			const clause &cl = clauses[i].cl;
			const numeric_classifier *nc = clauses[i].nc;
			CSP csp(cl, rels);
			var_domains assign = domains;
			if (csp.solve(assign)) {
				loggers->get(LOG_EM) << "matched clause:" << endl << cl << endl;
				var_domains::const_iterator vi, viend;
				int_tuple assign_tuple(assign.size());
				int j;
				for (j = 0, vi = assign.begin(), viend = assign.end(); vi != viend; ++j, ++vi) {
					assert(vi->second.size() == 1);
					int var = vi->first, val = *vi->second.begin();
					loggers->get(LOG_EM) << var << " = " << val << endl;
					assign_tuple[j] = val;
				}
				if (nc) {
					rvec x1;
					extract_vec(assign_tuple, x, sig, x1);
					result = nc->classify(x1);
					loggers->get(LOG_EM) << "NC votes for " << result << endl;
					if (result == 0) {
						matched_clause = i;
						used_nc = true;
						break;
					}
				} else {
					loggers->get(LOG_EM) << "No NC, voting for 0" << endl;
					matched_clause = i;
					used_nc = false;
					result = 0;
					break;
				}
			}
		}
	}
	if (result == 1) {
		// no matched clause, FOIL thinks this is a negative
		if (neg_nc) {
			int_tuple assign_tuple(2);
			rvec x1;

			assign_tuple[0] = 0;
			assign_tuple[1] = sig[target].id;
			extract_vec(assign_tuple, x, sig, x1);
			result = neg_nc->classify(x1);
			loggers->get(LOG_EM) << "No matched clauses, NC votes for " << result << endl;
			used_nc = true;
		} else {
			// no false negatives in training, so this must be a negative
			loggers->get(LOG_EM) << "No matched clauses, no NC, vote for 1" << endl;
			used_nc = false;
		}
	}
	return result;
}

void binary_classifier::update(const relation &mem_i, const relation &mem_j, const relation_table &rels, const model_train_data &data, bool use_foil, bool prune, const string &nc_type) {
	function_timer t(timers.get_or_add("update"));
	
	clauses.clear();
	neg_success_rate = 0.0;
	if (neg_nc) {
		delete neg_nc;
		neg_nc = NULL;
	}
	
	if (use_foil && !mem_i.empty() && !mem_j.empty()) {
		FOIL foil(loggers);
		foil.set_problem(mem_i, mem_j, rels);
		foil.learn(prune, true);

		clauses.resize(foil.num_clauses());
		for (int k = 0, kend = foil.num_clauses(); k < kend; ++k) {
			clauses[k].cl = foil.get_clause(k);
			clauses[k].false_pos = foil.get_false_positives(k);
			clauses[k].true_pos = foil.get_true_positives(k);
			clauses[k].success_rate = clauses[k].true_pos.size() / static_cast<double>(clauses[k].true_pos.size() + clauses[k].false_pos.size());
		}
		false_negatives = foil.get_false_negatives();
		true_negatives = foil.get_true_negatives();
		neg_success_rate = true_negatives.size() / static_cast<double>(true_negatives.size() + false_negatives.size());
	} else {
		/*
		 Don't learn any clauses. Instead consider every member of i a false negative
		 and every member of j a true negative, and let the numeric classifier take care of it.
		*/
		false_negatives = mem_i;
		true_negatives = mem_j;
		neg_success_rate = true_negatives.size() / static_cast<double>(true_negatives.size() + false_negatives.size());
	}
	
	/*
	 For each clause cl in clauses, if cl misclassified any of the
	 members of j in the training set as a member of i (false positive
	 for cl), train a numeric classifier to classify it correctly.
	 
	 Also train a numeric classifier to catch misclassified members of
	 i (false negatives for the entire clause vector).
	*/
	if (nc_type != "none") {
		for (int k = 0, kend = clauses.size(); k < kend; ++k) {
			assert(clauses[k].nc == NULL);
			if (clauses[k].success_rate < .75) {
				double nc_success_rate;
				numeric_classifier *nc = learn_numeric_classifier(nc_type, clauses[k].true_pos, clauses[k].false_pos, data, nc_success_rate);
				if (nc_success_rate > clauses[k].success_rate) {
					clauses[k].nc = nc;
					clauses[k].success_rate = nc_success_rate;
				} else {
					delete nc;
				}
			}
		}
		
		if (neg_success_rate < .75) {
			double nc_success_rate;
			// false_negatives = should be positives, true_negatives = should be negative
			numeric_classifier *nc = learn_numeric_classifier(nc_type, false_negatives, true_negatives, data, nc_success_rate);
			if (nc_success_rate > neg_success_rate) {
				neg_nc = nc;
				neg_success_rate = nc_success_rate;
			} else {
				delete nc;
			}
		}
	}
}

/*
 Weighted average success rate, weighted by the number of instances covered by each clause
*/
double binary_classifier::get_success_rate() const {
	int total = 0;
	double avg_success_rate = 0;
	for (int i = 0, iend = clauses.size(); i < iend; ++i) {
		int n = clauses[i].false_pos.size() + clauses[i].true_pos.size();
		total += n;
		avg_success_rate += clauses[i].success_rate * n;
	}
	int nneg = true_negatives.size() + false_negatives.size();
	total += nneg;
	avg_success_rate += nneg * neg_success_rate;
	return avg_success_rate / total;
}

void clause_info::serialize(ostream &os) const {
	serializer(os) << cl << false_pos << true_pos << success_rate << (nc != NULL);
	if (nc) {
		nc->serialize(os);
	}
}

void clause_info::unserialize(istream &is) {
	bool has_nc;
	unserializer(is) >> cl >> false_pos >> true_pos >> success_rate >> has_nc;
	if (nc) {
		delete nc;
	}
	nc = has_nc ? unserialize_numeric_classifier(is) : NULL;
}

classifier::classifier(const model_train_data &data, logger_set *loggers) 
: data(data), foil(true), prune(true), context(true), nc_type("dtree"), loggers(loggers)
{
	old_foil = foil;
	old_prune = prune;
	old_context = context;
	old_nc_type = nc_type;
}

classifier::~classifier() {
	clear_and_dealloc(pairs);
	clear_and_dealloc(classes);
}

void classifier::proxy_get_children(map<string, cliproxy*> &c) {
	c["use_foil"]    = new bool_proxy(&foil, "Use FOIL for classification.");
	c["use_pruning"] = new bool_proxy(&prune, "Prune FOIL clauses.");
	c["use_context"] = new bool_proxy(&context, "Consider only closest objects in classification.");
	c["nc_type"]     = new memfunc_proxy<classifier>(this, &classifier::cli_nc_type);
	c["dump_foil6"]  = new memfunc_proxy<classifier>(this, &classifier::cli_dump_foil6);
}

void classifier::cli_nc_type(const vector<string> &args, ostream &os) {
	static const char *names[] = { "none", "dtree", "lda", "sign" };
	
	if (args.empty()) {
		os << nc_type << endl;
	} else {
		nc_type = args[0];
		for (int i = 0, iend = sizeof(names) / sizeof(names[0]); i < iend; ++i) {
			if (args[0] == names[i]) {
				nc_type = names[i];
				return;
			}
		}
		os << "invalid numeric classifier type" << endl;
	}
}

void classifier::add_class() {
	int c = classes.size();
	for (int i = 1, iend = classes.size(); i < iend; ++i) {
		assert(i != 0 && c != 0);
		pairs.push_back(new pair_info(i, c));
	}
	classes.push_back(new class_info);
}

void classifier::del_classes(const vector<int> &c) {
	vector<int> class_map(classes.size());
	int n = 1;

	class_map[0] = 0;
	for (int i = 1, iend = classes.size(); i < iend; ++i) {
		if (has(c, i)) {
			if (!classes[i]->mem_rel.empty()) {
				cout << "ERROR: deleting non-empty class" << endl;
				cout << classes[i]->mem_rel << endl;
				FATAL("");
			}
			delete classes[i];
			class_map[i] = -1;
		} else {
			if (n != i) {
				classes[n] = classes[i];
			}
			class_map[i] = n++;
		}
	}

	assert(n == classes.size() - c.size());
	classes.resize(n);
	
	std::list<pair_info*>::iterator i, iend;
	for (i = pairs.begin(), iend = pairs.end(); i != iend; ) {
		pair_info &p = **i;
		p.cls_i = class_map[p.cls_i];
		p.cls_j = class_map[p.cls_j];
		assert(p.cls_i != 0 && p.cls_j != 0);
		if (p.cls_i == -1 || p.cls_j == -1) {
			delete *i;
			i = pairs.erase(i);
		} else {
			++i;
		}
	}
}

void classifier::update_class(int i, int old_class, int new_class) {
	assert(0 <= new_class && new_class < classes.size());
	if (old_class == new_class) {
		return;
	}
	
	const model_train_inst &inst = data.get_inst(i);
	int target = (*inst.sig)[inst.target].id;
	int_tuple old_t(2);
	old_t[0] = i; old_t[1] = target;
	if (old_class >= 0) {
		class_info *oldc = classes[old_class];
		assert(oldc->mem_rel.contains(old_t));
		oldc->mem_rel.del(i, target);
		oldc->stale = true;
	}
	
	class_info *newc = classes[new_class];
	newc->mem_rel.add(i, target);
	newc->stale = true;
}

classifier::pair_info *classifier::find(int i, int j) {
	if (i >= j) {
		return NULL;
	}

	std::list<pair_info*>::iterator pi, pend;
	for (pi = pairs.begin(), pend = pairs.end(); pi != pend; ++pi) {
		if ((**pi).cls_i == i && (**pi).cls_j == j) {
			return *pi;
		}
	}
	return NULL;
}

void classifier::update() {
	std::list<pair_info*>::iterator i, iend;
	bool options_changed = false;
	
	if (foil != old_foil || prune != old_prune || context != old_context || nc_type != old_nc_type) {
		options_changed = true;
		old_foil = foil; old_prune = prune; old_context = context; old_nc_type = nc_type;
	}
	
	for (i = pairs.begin(), iend = pairs.end(); i != iend; ++i) {
		pair_info *p = *i;
		class_info *ci = classes[p->cls_i], *cj = classes[p->cls_j];
		if (!options_changed && !ci->stale && !cj->stale) {
			continue;
		}
		
		if (p->clsfr) {
			delete p->clsfr;
		}
		binary_classifier *pos_cls, *neg_cls;
		const relation_table *rels;

		if (context) {
			rels = &data.get_context_rels();
		} else {
			rels = &data.get_all_rels();
		}

		pos_cls = new binary_classifier(loggers);
		pos_cls->update(ci->mem_rel, cj->mem_rel, *rels, data, foil, prune, nc_type);
		p->clsfr = pos_cls;
		p->negated = false;
		if (pos_cls->get_success_rate() < .99) {
			neg_cls = new binary_classifier(loggers);
			neg_cls->update(cj->mem_rel, ci->mem_rel, *rels, data, foil, prune, nc_type);
			if (neg_cls->get_success_rate() > pos_cls->get_success_rate()) {
				p->clsfr = neg_cls;
				p->negated = true;
				delete pos_cls;
			} else {
				delete neg_cls;
			}
		}
	}
	
	/*
	 have to wait until all pairs have been examined before setting stale to false
	 for each class
	*/
	for (int i = 0, iend = classes.size(); i < iend; ++i) {
		classes[i]->stale = false;
	}
}

void classifier::classify(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, vector<int> &votes, rvec &vote_trace) const {
	const_cast<classifier*>(this)->update();
	votes.clear();
	votes.resize(classes.size(), 0);
	
	const relation_table *r;
	if (context) {
		relation_table *r1 = new relation_table;
		get_context_rels(sig[target].id, rels, *r1);
		r = r1;
	} else {
		r = &rels;
	}
	
	/*
	 vote_trace is "formatted" in blocks of 5 values. Each block corresponds to
	 the voting record for one pair of modes. The elements of each block are:

	   1. class 0 - integer [1 .. max mode]
	   2. class 1 - integer [1 .. max mode]
	   3. vote (0 for class 0, 1 for class 1)
	   4. index of the clause matched
	   5. whether a numeric classifier was used (1 = yes)
	*/
	vote_trace.resize(pairs.size() * 5);

	std::list<pair_info*>::const_iterator i, iend;
	int j;
	for (i = pairs.begin(), iend = pairs.end(), j = 0; i != iend; ++i, j += 5) {
		const pair_info &p = **i;
		assert(p.cls_i > 0 && p.cls_j > 0);
		int clause_index;
		bool used_nc;
		loggers->get(LOG_EM) << "VOTE " << p.cls_i << " " << p.cls_j << endl;
		int winner = p.clsfr->vote(target, sig, *r, x, clause_index, used_nc);
		if (p.negated) {
			winner = 1 - winner;
		}
		if (winner == 0) {
			++votes[p.cls_i];
		} else if (winner == 1) {
			++votes[p.cls_j];
		} else {
			FATAL("illegal winner");
		}
		vote_trace(j + 0) = p.cls_i;
		vote_trace(j + 1) = p.cls_j;
		vote_trace(j + 2) = winner;
		vote_trace(j + 3) = clause_index;
		vote_trace(j + 4) = used_nc;
	}
	
	if (context) {
		delete r;
	}
}

void classifier::proxy_use_sub(const vector<string> &args, ostream &os) {
	update();
	
	if (args.empty()) {
		// print summary of all classifiers
		for (int i = 1, iend = classes.size(); i < iend; ++i) {
			for (int j = i + 1, jend = classes.size(); j < jend; ++j) {
				pair_info *p = find(i, j);
				assert(p && p->clsfr);
				if (p->negated) {
					os << "=== FOR MODES " << i << "/" << j << " (Negated) ===" << endl;
				} else {
					os << "=== FOR MODES " << i << "/" << j << " ===" << endl;
				}
				p->clsfr->inspect(os);
				os << endl;
			}
		}
		return;
	}
	
	int i, j;
	pair_info *p;
	if (args.size() != 2) {
		os << "Specify two modes" << endl;
		return;
	}
	
	if (!parse_int(args[0], i) || !parse_int(args[1], j) || !(p = find(i, j))) {
		os << "invalid modes, make sure i < j" << endl;
		return;
	}
	
	if (p->negated) {
		os << "Negated" << endl;
	}
	p->clsfr->inspect_detailed(os);
}

void classifier::cli_dump_foil6(const vector<string> &args, ostream &os) const {
	int m1, m2;
	if (args.size() != 2 || 
	    !parse_int(args[0], m1) || 
	    !parse_int(args[1], m2) ||
	    m1 < 0 || m1 >= classes.size() || m2 < 0 || m2 >= classes.size() || m1 == m2) 
	{
		os << "Specify 2 modes" << endl;
		return;
	}
	
	if (m1 > m2) {
		swap(m1, m2);
	}
	
	FOIL foil(loggers);
	if (context) {
		foil.set_problem(classes[m1]->mem_rel, classes[m2]->mem_rel, data.get_context_rels());
	} else {
		foil.set_problem(classes[m1]->mem_rel, classes[m2]->mem_rel, data.get_all_rels());
	}
	foil.dump_foil6(os);
}

void classifier::serialize(ostream &os) const {
	serializer(os) << pairs << classes;
}

void classifier::unserialize(istream &is) {
	unserializer(is) >> pairs >> classes;

	std::list<pair_info*>::iterator i, iend;
	for (i = pairs.begin(), iend = pairs.end(); i != iend; ++i) {
		if ((**i).clsfr) {
			(**i).clsfr->set_loggers(loggers);
		}
	}
}

void classifier::pair_info::serialize(ostream &os) const {
	assert(cls_i > 0 && cls_j > 0);
	serializer(os) << cls_i << cls_j << negated << clsfr;
}

void classifier::pair_info::unserialize(istream &is) {
	unserializer(is) >> cls_i >> cls_j >> negated >> clsfr;
	assert(cls_i > 0 && cls_j > 0);
}

void classifier::class_info::serialize(ostream &os) const {
	serializer(os) << mem_rel << stale;
}

void classifier::class_info::unserialize(istream &is) {
	unserializer(is) >> mem_rel >> stale;
}
