#include "mode.h"
#include "em.h"
#include "serialize.h"
#include "params.h"

using namespace std;

const regression_type REGRESSION_ALG = FORWARD;

/*
 Generate all possible combinations of sets of items
*/
template <typename T>
class multi_combination_generator {
public:
	multi_combination_generator(const std::vector<std::vector<T> > &elems, bool allow_repeat)
	: elems(elems), indices(elems.size(), 0), allow_repeat(allow_repeat), finished(false)
	{
		empty = false;
		if (elems.empty()) {
			empty = true;
		} else {
			for (int i = 0; i < elems.size(); ++i) {
				if (elems[i].empty()) {
					empty = true;
					break;
				}
			}
		}
	}

	void reset() {
		finished = false;
		fill(indices.begin(), indices.end(), 0);
	}

	bool next(std::vector<T> &comb) {
		if (empty) {
			return false;
		}
		
		comb.resize(elems.size());
		std::set<int> s;
		while (!finished) {
			bool has_repeat = false;
			s.clear();
			for (int i = 0; i < elems.size(); ++i) {
				comb[i] = elems[i][indices[i]];
				if (!allow_repeat) {
					std::pair<std::set<int>::iterator, bool> p = s.insert(comb[i]);
					if (!p.second) {
						has_repeat = true;
						break;
					}
				}
			}
			increment(0);
			if (allow_repeat || !has_repeat) {
				return true;
			}
		}
		return false;
	}

private:
	void increment(int i) {
		if (i >= elems.size()) {
			finished = true;
		} else if (++indices[i] >= elems[i].size()) {
			indices[i] = 0;
			increment(i + 1);
		}
	}

	const std::vector<std::vector<T> > &elems;
	std::vector<int> indices;
	bool allow_repeat, finished, empty;
};

em_mode::em_mode(bool noise, bool manual, const model_train_data &data, logger_set *loggers) 
: noise(noise), manual(manual), data(data), new_fit(true), n_nonzero(-1), loggers(loggers), intercept(INF)
{
	if (noise) {
		stale = false;
	} else {
		stale = true;
	}
	
}

void em_mode::set_params(const scene_sig &dsig, int target, const rvec &coefs, double inter) {
	n_nonzero = 0;
	intercept = inter;
	roles.clear();
	if (coefs.size() > 0) {
		vector<int> role_map;
		role target_role;
		target_role.type = dsig[target].type;
		target_role.properties = dsig[target].props;
		target_role.coefficients = coefs.segment(dsig[target].start, dsig[target].props.size());
		roles.push_back(target_role);
		role_map.push_back(target);
		for (int i = 0; i < dsig.size(); ++i) {
			if (i == target) {
				continue;
			}
			int start = dsig[i].start;
			int end = start + dsig[i].props.size();
			bool is_role = false;
			for (int j = start; j < end; ++j) {
				if (coefs(j) != 0.0) {
					++n_nonzero;
					is_role = true;
				}
			}
			if (is_role) {
				role r;
				r.type = dsig[i].type;
				r.properties = dsig[i].props;
				r.coefficients = coefs.segment(start, end - start);
				roles.push_back(r);
				role_map.push_back(i);
			}
		}
		
		if (role_maps.size() == 0) {
			role_map_entry e;
			e.role_map = role_map;
			role_maps.push_back(e);
		} else {
			assert(role_maps.size() == 1);        // all existing members must have the same signature
			role_maps[0].role_map = role_map;
		}
	}
	new_fit = true;
	role_classifiers_stale = true;
}

/*
 Upon return, mapping[i] will contain the position in dsig that holds the
 object to be mapped to the i'th role. Again, the mapping vector will hold
 indexes, not ids.
*/
bool em_mode::map_roles(int target, const scene_sig &dsig, const relation_table &rels, vector<int> &mapping) const {
	vector<bool> used(dsig.size(), false);
	used[target] = true;
	mapping.resize(roles.empty() ? 1 : roles.size(), -1);
	mapping[0] = target;  // target always maps to target
	
	update_role_classifiers();
	
	var_domains domains;
	// 0 = time, 1 = target, 2 = object we're searching for
	domains[0].insert(0);
	domains[1].insert(dsig[target].id);
	
	for (int i = 1, iend = roles.size(); i < iend; ++i) {
		const FOIL_result &clsfr = roles[i].classifier;
		set<int> &d = domains[2];
		d.clear();
		for (int j = 0; j < dsig.size(); ++j) {
			if (!used[j] && dsig[j].type == roles[i].type) {
				d.insert(dsig[j].id);
			}
		}
		if (d.empty()) {
			return false;
		} else if (d.size() == 1 || clsfr.clauses.empty()) {
			mapping[i] = dsig.find_id(*d.begin());
		} else {
			bool found = false;
			for (int j = 0, jend = clsfr.clauses.size(); j < jend; ++j) {
				CSP csp(clsfr.clauses[j].cl, rels);
				if (csp.solve(domains)) {
					assert(d.size() == 1);
					mapping[i] = dsig.find_id(*d.begin());
					found = true;
					break;
				}
			}
			if (!found) {
				return false;
			}
		}
		used[mapping[i]] = true;
	}
	return true;
}

/*
 pos_obj and neg_obj can probably be cached and updated as data points
 are assigned to modes.
*/
void em_mode::update_role_classifiers() const {
	if (!role_classifiers_stale) {
		return;
	}
	
	const relation_table &rels = data.get_all_rels();
	for (int i = 1; i < roles.size(); ++i) {   // 0 is always target, no need to map
		string type = roles[i].type;
		relation pos_obj(3), neg_obj(3);
		int_tuple objs(2);

		for (int j = 0, jend = role_maps.size(); j < jend; ++j) {
			const vector<int> &m = role_maps[j].role_map;
			const interval_set &mem = role_maps[j].members;
			assert(m.size() == roles.size());
			interval_set::const_iterator k, kend;
			for (k = mem.begin(), kend = mem.end(); k != kend; ++k) {
				const model_train_inst &d = data.get_inst(*k);
				const scene_sig &dsig = *d.sig;

				int o = dsig[m[i]].id;
				objs[0] = dsig[d.target].id;
				objs[1] = o;
				pos_obj.add(*k, objs);
				for (int l = 0, lend = dsig.size(); l < lend; ++l) {
					if (dsig[l].type == type && l != d.target && l != m[i]) {
						objs[1] = dsig[l].id;
						neg_obj.add(*k, objs);
					}
				}
			}
		}
		
		FOIL_result *r = const_cast<FOIL_result*>(&roles[i].classifier);
		if (!run_FOIL(pos_obj, neg_obj, rels, true, true, loggers, *r)) {
			// respond to this situation appropriately
			cerr << "FOIL failed for role " << i << endl;
		}
	}
	role_classifiers_stale = false;
}

void em_mode::proxy_get_children(map<string, cliproxy*> &c) {
	c["clauses"] = new memfunc_proxy<em_mode>(this, &em_mode::cli_clauses);
	c["members"] = new memfunc_proxy<em_mode>(this, &em_mode::cli_members);
}

void em_mode::proxy_use_sub(const vector<string> &args, ostream &os) {
	if (noise) {
		os << "noise" << endl;
	} else {
		string func;
		get_function_string(func);
		os << "function" << endl << func << endl << endl;

		os << "role maps" << endl;
		for (int i = 0, iend = role_maps.size(); i < iend; ++i) {
			int d = role_maps[i].members.ith(0);
			const scene_sig &dsig = *data.get_inst(d).sig;
			const vector<int> &omap = role_maps[i].role_map;

			os << role_maps[i].members << endl;
			for (int j = 0, jend = omap.size(); j < jend; ++j) {
				os << j << " -> " << dsig[omap[j]].name << endl;
			}
			os << endl;
		}
	}
}

void em_mode::cli_clauses(const vector<string> &args, ostream &os) const {
	update_role_classifiers();
	
	if (!args.empty()) {
		int i;
		if (!parse_int(args[0], i) || i < 0 || i >= roles.size()) {
			os << "specify valid role (1 - " << roles.size() - 1 << ")" << endl;
			return;
		}
		roles[i].classifier.inspect_detailed(os);
		os << endl;
	} else {
		for (int i = 1, iend = roles.size(); i < iend; ++i) {
			os << "ROLE " << i << endl;
			roles[i].classifier.inspect(os);
			os << endl << endl;
		}
	}
}

void em_mode::cli_members(ostream &os) const {
	os << members << endl;
}

/*
 The fields noise, data, and sigs are initialized in the constructor, and
 therefore not (un)serialized.
*/
void em_mode::serialize(ostream &os) const {
	serializer(os) << stale << new_fit << members << role_maps << roles << sorted_ys
	               << intercept << n_nonzero << manual << role_classifiers_stale;
}

void em_mode::unserialize(istream &is) {
	unserializer(is) >> stale >> new_fit >> members >> role_maps >> roles >> sorted_ys
	                 >> intercept >> n_nonzero >> manual >> role_classifiers_stale;
}

double em_mode::calc_error(int target, const scene_sig &dsig, const rvec &x, double y, double noise_var, vector<int> &role_map) const {
	if (noise) {
		return INF;
	}
	
	role_map.clear();

	if (roles.empty()) {
		// constant prediction
		return fabs(y - intercept);
	}
	
	/*
	 See if any of the existing mappings result in acceptable error
	*/
	for (int i = 0, iend = role_maps.size(); i < iend; ++i) {
		const vector<int> &m = role_maps[i].role_map;
		/* check if it's legal */
		if (m[0] != target) {
			continue;
		}
		bool legal = true;
		for (int j = 0, jend = m.size(); j < jend; ++j) {
			if (roles[j].type != dsig[m[j]].type) {
				legal = false;
				break;
			}
		}
		if (!legal) {
			continue;
		}
		double error = fabs(y - predict(dsig, x, m));
		//if (error < NUM_STDEVS_THRESH * sqrt(noise_var)) {
		if (error < 1e-15) {
			role_map = m;
			return error;
		}
	}

	/*
	 No existing role map results in acceptable error, so now try all possible
	 assignments.
	*/
	
	/*
	 Create the input table for the combination generator to generate
	 all possible assignments. possibles[i] should be a list of all
	 object indices that can be assigned to position i in the model
	 signature.
	*/
	vector<vector<int> > possibles(roles.size());
	possibles[0].push_back(target);
	for (int i = 1; i < roles.size(); ++i) {
		for (int j = 0; j < dsig.size(); ++j) {
			if (dsig[j].type == roles[i].type && j != target) {
				possibles[i].push_back(j);
			}
		}
	}
	multi_combination_generator<int> gen(possibles, false);
	
	/*
	 Iterate through all assignments and find the one that gives lowest error
	*/
	vector<int> rm;
	double best_error = INF;
	while (gen.next(rm)) {
		double error = fabs(y - predict(dsig, x, rm));
		if (error < best_error) {
			best_error = error;
			role_map = rm;
		}
	}
	return best_error;
}

bool em_mode::update_fits(double noise_var) {
	if (!stale || manual) {
		return false;
	}
	if (members.empty()) {
		roles.clear();
		intercept = 0.0;
		return false;
	}
	int xcols = 0;
	for (int i = 0, iend = roles.size(); i < iend; ++i) {
		xcols += roles[i].coefficients.size();
	}
	
	mat X(members.size(), xcols), Y(members.size(), 1);
	int j = 0;
	for (int i = 0, iend = role_maps.size(); i < iend; ++i) {
		const vector<int> &m = role_maps[i].role_map;
		const interval_set &members = role_maps[i].members;
		assert(m.size() == roles.size());
		interval_set::const_iterator k, kend;
		for (k = members.begin(), kend = members.end(); k != kend; ++k) {
			const model_train_inst &d = data.get_inst(*k);
			rvec x(xcols);
			int s = 0;
			for (int mi = 0, miend = m.size(); mi < miend; ++mi) {
				const scene_sig::entry &e = (*d.sig)[m[mi]];
				int n = e.props.size();
				x.segment(s, n) = d.x.segment(e.start, n);
				s += n;
			}
			assert(s == xcols);
			X.row(j) = x;
			Y.row(j++) = d.y;
		}
	}
	mat coefs;
	rvec inter;
	linreg(REGRESSION_ALG, X, Y, cvec(), noise_var, false, coefs, inter);
	intercept = inter(0);
	rvec coefs1 = coefs.col(0);

	n_nonzero = 0;
	int last = 0;
	rvec c;
	for (int i = 0, iend = roles.size(), s = 0; i < iend; ++i) {
		int n = roles[i].coefficients.size();
		c = coefs1.segment(s, n);
		s += n;
		int k = (c.array() != 0.0).count();
		if (i == 0 || k > 0) {
			if (last < i) {
				roles[last] = roles[i];
			}
			roles[last].coefficients = c;
			++last;
			n_nonzero += k;
		}
	}

	stale = false;
	new_fit = true;
	return true;
}

double em_mode::predict(const scene_sig &dsig, const rvec &x, const vector<int> &role_map) const {
	double sum = intercept;
	assert(role_map.size() == roles.size());
	for (int i = 0, iend = role_map.size(); i < iend; ++i) {
		const scene_sig::entry &e = dsig[role_map[i]];
		assert(roles[i].type == e.type);
		sum += roles[i].coefficients.dot(x.segment(e.start, e.props.size()));
	}
	return sum;
}

void em_mode::add_example(int t, const vector<int> &ex_role_map, double noise_var) {
	assert(!members.contains(t) && ex_role_map.size() == roles.size());
	
	const model_train_inst &d = data.get_inst(t);
	members.insert(t);
	
	bool found = false;
	for (int i = 0, iend = role_maps.size(); i < iend; ++i) {
		if (role_maps[i].role_map == ex_role_map) {
			role_maps[i].members.insert(t);
			found = true;
			break;
		}
	}
	if (!found) {
		role_map_entry e;
		e.role_map = ex_role_map;
		e.members.insert(t);
		role_maps.push_back(e);
	}

	if (noise) {
		sorted_ys.insert(make_pair(d.y(0), t));
	} else {
		double py = predict(*d.sig, d.x, ex_role_map);
		if (fabs(d.y(0) - py) > sqrt(noise_var) * NUM_STDEVS_THRESH) {
			stale = true;
		}
	}
	role_classifiers_stale = true;
}

void em_mode::del_example(int t) {
	const model_train_inst &d = data.get_inst(t);

	members.erase(t);
	for (int i = 0, iend = role_maps.size(); i < iend; ++i) {
		role_maps[i].members.erase(t);
	}
	if (noise) {
		sorted_ys.erase(make_pair(d.y(0), t));
	}
	role_classifiers_stale = true;
	stale = true;
}

void em_mode::largest_const_subset(vector<int> &subset) {
	vector<int> s;
	set<pair<double, int> >::const_iterator i;
	double last = NAN;
	subset.clear();
	for (i = sorted_ys.begin(); i != sorted_ys.end(); ++i) {
		if (i->first == last) {
			s.push_back(i->second);
		} else {
			if (s.size() > subset.size()) {
				subset = s;
			}
			last = i->first;
			s.clear();
			s.push_back(i->second);
		}
	}
	if (s.size() > subset.size()) {
		subset = s;
	}
}

bool em_mode::unifiable(int sig, int target) const {
	bool uniform_sig = true;
	interval_set::const_iterator i, iend;
	for (i = members.begin(), iend = members.end(); i != iend; ++i) {
		const model_train_inst &d = data.get_inst(*i);
		if (d.sig_index != sig || d.target != target) {
			uniform_sig = false;
			break;
		}
	}
	return !manual && uniform_sig && role_maps.size() == 1;
}

int em_mode::get_num_nonzero_coefs() const {
	if (noise) {
		return numeric_limits<int>::max();
	}
	assert(n_nonzero >= 0);
	return n_nonzero;
}

void em_mode::get_function_string(string &s) const {
	if (noise) {
		s = "noise";
		return;
	}
	
	stringstream ss;
	int k = 0;
	bool first = true;
	for (int i = 0, iend = roles.size(); i < iend; ++i) {
		const role &r = roles[i];
		for (int j = 0, jend = r.coefficients.size(); j < jend; ++j, ++k) {
			double c = r.coefficients(j);
			if (c == 0.0) {
				continue;
			}
			if (first) {
				if (approx_equal(c, -1.0, SAME_THRESH)) {
					ss << "-";
				} else if (!approx_equal(c, 1.0, SAME_THRESH)) {
					ss << c << " * ";
				}
				first = false;
			} else if (c < 0.0) {
				ss << " - ";
				if (!approx_equal(c, -1.0, SAME_THRESH)) {
					ss << fabs(c) << " * ";
				}
			} else {
				ss << " + ";
				if (!approx_equal(c, 1.0, SAME_THRESH)) {
					ss << c << " * ";
				}
			}
			ss << i << ":" << r.type << ":" << r.properties[j];
		}
	}
	
	if (first) {
		ss << intercept;
	} else if (intercept > 0.0) {
		ss << " + " << intercept;
	} else if (intercept < 0.0) {
		ss << " - " << -intercept;
	}
	s = ss.str();
}

void em_mode::role_map_entry::serialize(ostream &os) const {
	serializer(os) << role_map << members;
}

void em_mode::role_map_entry::unserialize(istream &is) {
	unserializer(is) >> role_map >> members;
}

void em_mode::role::serialize(ostream &os) const {
	serializer(os) << type << properties << coefficients << classifier;
}

void em_mode::role::unserialize(istream &is) {
	unserializer(is) >> type >> properties >> coefficients >> classifier;
}

