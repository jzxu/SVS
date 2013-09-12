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
: noise(noise), manual(manual), data(data), new_fit(true), n_nonzero(-1), loggers(loggers)
{
	if (noise) {
		stale = false;
	} else {
		stale = true;
	}
	
}

void em_mode::get_params(scene_sig &sig, rvec &coefs, double &inter) const {
	sig = this->sig;
	coefs = coefficients;
	inter = intercept;
}

void em_mode::set_params(const scene_sig &dsig, int target, const rvec &coefs, double inter) {
	n_nonzero = 0;
	intercept = inter;
	if (coefs.size() == 0) {
		coefficients.resize(0);
	} else {
		// find relevant objects (with nonzero coefficients)
		vector<int> relevant_objs;
		relevant_objs.push_back(target);
		for (int i = 0; i < dsig.size(); ++i) {
			if (i == target) {
				continue;
			}
			int start = dsig[i].start;
			int end = start + dsig[i].props.size();
			bool relevant = false;
			for (int j = start; j < end; ++j) {
				if (coefs(j) != 0.0) {
					++n_nonzero;
					relevant = true;
				}
			}
			if (relevant) {
				relevant_objs.push_back(i);
			}
		}
		
		int end = 0;
		coefficients.resize(coefs.size());
		sig.clear();
		for (int i = 0; i < relevant_objs.size(); ++i) {
			const scene_sig::entry &e = dsig[relevant_objs[i]];
			sig.add(e);
			int start = e.start, n = e.props.size();
			coefficients.segment(end, n) = coefs.segment(start, n);
			end += n;
		}
		coefficients.conservativeResize(end);
		
		if (obj_maps.size() == 0) {
			obj_map_entry e;
			e.obj_map = relevant_objs;
			obj_maps.push_back(e);
		} else {
			assert(obj_maps.size() == 1);        // all existing members must have the same signature
			obj_maps[0].obj_map = relevant_objs;
		}
	}
	new_fit = true;
}

/*
 Upon return, mapping[i] will contain the position in dsig that holds the
 object to be mapped to the i'th variable in the model signature. Again, the
 mapping vector will hold indexes, not ids.
*/
bool em_mode::map_objs(int target, const scene_sig &dsig, const relation_table &rels, vector<int> &mapping) const {
	vector<bool> used(dsig.size(), false);
	used[target] = true;
	mapping.resize(sig.empty() ? 1 : sig.size(), -1);
	mapping[0] = target;  // target always maps to target
	
	update_obj_clauses();
	
	var_domains domains;
	// 0 = time, 1 = target, 2 = object we're searching for
	domains[0].insert(0);
	domains[1].insert(dsig[target].id);
	
	for (int i = 1; i < sig.size(); ++i) {
		set<int> &d = domains[2];
		d.clear();
		for (int j = 0; j < dsig.size(); ++j) {
			if (!used[j] && dsig[j].type == sig[i].type) {
				d.insert(dsig[j].id);
			}
		}
		if (d.empty()) {
			return false;
		} else if (d.size() == 1 || obj_clauses[i].empty()) {
			mapping[i] = dsig.find_id(*d.begin());
		} else {
			bool found = false;
			for (int j = 0, jend = obj_clauses[i].size(); j < jend; ++j) {
				CSP csp(obj_clauses[i][j], rels);
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
void em_mode::update_obj_clauses() const {
	if (!obj_clauses_stale) {
		return;
	}
	
	const relation_table &rels = data.get_all_rels();
	obj_clauses.resize(sig.size());
	for (int i = 1; i < sig.size(); ++i) {   // 0 is always target, no need to map
		string type = sig[i].type;
		relation pos_obj(3), neg_obj(3);
		int_tuple objs(2);

		for (int j = 0, jend = obj_maps.size(); j < jend; ++j) {
			const vector<int> &m = obj_maps[j].obj_map;
			const interval_set &mem = obj_maps[j].members;
			assert(m.size() == sig.size());
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
		
		FOIL foil(loggers);
		foil.set_problem(pos_obj, neg_obj, rels);
		if (!foil.learn(true, false)) {
			// respond to this situation appropriately
			FATAL("FOIL failed");
		}
		obj_clauses[i].resize(foil.num_clauses());
		for (int j = 0, jend = foil.num_clauses(); j < jend; ++j) {
			obj_clauses[i][j] = foil.get_clause(j);
		}
	}
	obj_clauses_stale = false;
}

void em_mode::proxy_get_children(map<string, cliproxy*> &c) {
	c["clauses"] = new memfunc_proxy<em_mode>(this, &em_mode::cli_clauses);
	c["members"] = new memfunc_proxy<em_mode>(this, &em_mode::cli_members);
	c["sig"]     = new memfunc_proxy<em_mode>(this, &em_mode::cli_sig);
}

void em_mode::proxy_use_sub(const vector<string> &args, ostream &os) {
	if (noise) {
		os << "noise" << endl;
	} else {
		string func;
		get_function_string(func);
		os << "function" << endl << func << endl << endl;

		os << "object maps" << endl;
		for (int i = 0, iend = obj_maps.size(); i < iend; ++i) {
			int d = obj_maps[i].members.ith(0);
			const scene_sig &dsig = *data.get_inst(d).sig;
			const vector<int> &omap = obj_maps[i].obj_map;

			os << obj_maps[i].members << endl;
			for (int j = 0, jend = omap.size(); j < jend; ++j) {
				os << j << " -> " << dsig[omap[j]].name << endl;
			}
			os << endl;
		}
	}
}

void em_mode::cli_clauses(ostream &os) const {
	table_printer t;
	update_obj_clauses();
	t.add_row() << 0 << "target";
	for (int j = 1; j < obj_clauses.size(); ++j) {
		t.add_row() << j;
		if (obj_clauses[j].empty()) {
			t << "empty";
		} else {
			for (int k = 0; k < obj_clauses[j].size(); ++k) {
				if (k > 0) {
					t.add_row().skip(1);
				}
				t << obj_clauses[j][k];
			}
		}
	}
	t.print(os);
}

void em_mode::cli_members(ostream &os) const {
	os << members << endl;
}

void em_mode::cli_sig(ostream &os) const {
	sig.print(os);
}

/*
 The fields noise, data, and sigs are initialized in the constructor, and
 therefore not (un)serialized.
*/
void em_mode::serialize(ostream &os) const {
	serializer(os) << stale << new_fit << members << obj_maps << sig << obj_clauses << sorted_ys
	               << coefficients << intercept << n_nonzero << manual << obj_clauses_stale;
}

void em_mode::unserialize(istream &is) {
	unserializer(is) >> stale >> new_fit >> members >> obj_maps >> sig >> obj_clauses >> sorted_ys
	                 >> coefficients >> intercept >> n_nonzero >> manual >> obj_clauses_stale;
}

double em_mode::assignment_error(const scene_sig &dsig, const rvec &x, double y, double noise_var, const vector<int> &assign) const {
	int xlen = sig.dim();
	rvec xc(xlen);
	int s = 0;
	assert(assign.size() == sig.size());
	for (int i = 0, iend = assign.size(); i < assign.size(); ++i) {
		const scene_sig::entry &e = dsig[assign[i]];
		int l = e.props.size();
		assert(sig[i].props.size() == l);
		xc.segment(s, l) = x.segment(e.start, l);
		s += l;
	}
	assert(s == xlen);
	double py = xc.dot(coefficients) + intercept;
	return fabs(y - py);
}

double em_mode::calc_error(int target, const scene_sig &dsig, const rvec &x, double y, double noise_var, vector<int> &best_assign) const {
	if (noise) {
		return INF;
	}
	
	best_assign.clear();

	if (sig.empty()) {
		// should be constant prediction
		assert(coefficients.size() == 0);
		return fabs(y - intercept);
	}
	
	/*
	 See if any of the existing mappings result in acceptable error
	*/
	for (int i = 0, iend = obj_maps.size(); i < iend; ++i) {
		const vector<int> &m = obj_maps[i].obj_map;
		/* check if it's legal */
		if (m[0] != target) {
			continue;
		}
		bool legal = true;
		for (int j = 0, jend = m.size(); j < jend; ++j) {
			if (sig[j].type != dsig[m[j]].type) {
				legal = false;
				break;
			}
		}
		if (!legal) {
			continue;
		}
		double error = assignment_error(dsig, x, y, noise_var, m);
		//if (error < NUM_STDEVS_THRESH * sqrt(noise_var)) {
		if (error < 1e-15) {
			best_assign = m;
			return error;
		}
	}

	/*
	 No existing object map results in acceptable error, so now try all possible
	 assignments.
	*/
	
	/*
	 Create the input table for the combination generator to generate
	 all possible assignments. possibles[i] should be a list of all
	 object indices that can be assigned to position i in the model
	 signature.
	*/
	vector<vector<int> > possibles(sig.size());
	possibles[0].push_back(target);
	for (int i = 1; i < sig.size(); ++i) {
		for (int j = 0; j < dsig.size(); ++j) {
			if (dsig[j].type == sig[i].type && j != target) {
				possibles[i].push_back(j);
			}
		}
	}
	multi_combination_generator<int> gen(possibles, false);
	
	/*
	 Iterate through all assignments and find the one that gives lowest error
	*/
	vector<int> assign;
	double best_error = INF;
	while (gen.next(assign)) {
		double error = assignment_error(dsig, x, y, noise_var, assign);
		if (error < best_error) {
			best_error = error;
			best_assign = assign;
		}
	}
	return best_error;
}

bool em_mode::update_fits(double noise_var) {
	if (!stale || manual) {
		return false;
	}
	if (members.empty()) {
		coefficients.resize(0);
		intercept = 0.0;
		return false;
	}
	int xcols = 0;
	for (int i = 0; i < sig.size(); ++i) {
		xcols += sig[i].props.size();
	}
	
	mat X(members.size(), xcols), Y(members.size(), 1);
	int j = 0;
	for (int i = 0, iend = obj_maps.size(); i < iend; ++i) {
		const vector<int> &m = obj_maps[i].obj_map;
		const interval_set &members = obj_maps[i].members;
		assert(m.size() == sig.size());
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
	coefficients = coefs.col(0);
	intercept = inter(0);
	stale = false;
	new_fit = true;
	return true;
}

double em_mode::predict(const scene_sig &dsig, const rvec &x, const vector<int> &ex_omap) const {
	if (coefficients.size() == 0) {
		return intercept;
	}
	
	assert(ex_omap.size() == sig.size());
	rvec xc(x.size());
	int xsize = 0;
	for (int j = 0, jend = ex_omap.size(); j < jend; ++j) {
		const scene_sig::entry &e = dsig[ex_omap[j]];
		int n = e.props.size();
		xc.segment(xsize, n) = x.segment(e.start, n);
		xsize += n;
	}
	xc.conservativeResize(xsize);
	return xc.dot(coefficients) + intercept;
}

void em_mode::add_example(int t, const vector<int> &ex_obj_map, double noise_var) {
	assert(!members.contains(t) && ex_obj_map.size() == sig.size());
	
	const model_train_inst &d = data.get_inst(t);
	members.insert(t);
	
	bool found = false;
	for (int i = 0, iend = obj_maps.size(); i < iend; ++i) {
		if (obj_maps[i].obj_map == ex_obj_map) {
			obj_maps[i].members.insert(t);
			found = true;
			break;
		}
	}
	if (!found) {
		obj_map_entry e;
		e.obj_map = ex_obj_map;
		e.members.insert(t);
		obj_maps.push_back(e);
	}

	if (noise) {
		sorted_ys.insert(make_pair(d.y(0), t));
	} else {
		double py = predict(*d.sig, d.x, ex_obj_map);
		if (fabs(d.y(0) - py) > sqrt(noise_var) * NUM_STDEVS_THRESH) {
			stale = true;
		}
	}
	obj_clauses_stale = true;
}

void em_mode::del_example(int t) {
	const model_train_inst &d = data.get_inst(t);

	members.erase(t);
	for (int i = 0, iend = obj_maps.size(); i < iend; ++i) {
		obj_maps[i].members.erase(t);
	}
	if (noise) {
		sorted_ys.erase(make_pair(d.y(0), t));
	}
	obj_clauses_stale = true;
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
	return !manual && uniform_sig && obj_maps.size() == 1;
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
	for (int i = 0, iend = sig.size(); i < iend; ++i) {
		for (int j = 0, jend = sig[i].props.size(); j < jend; ++j, ++k) {
			double c = coefficients(k);
			if (c == 0.0) {
				continue;
			} else if (first) {
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
			ss << i << ":" << sig[i].type << ":" << sig[i].props[j];
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

void em_mode::obj_map_entry::serialize(std::ostream &os) const {
	serializer(os) << obj_map << members;
}

void em_mode::obj_map_entry::unserialize(std::istream &is) {
	unserializer(is) >> obj_map >> members;
}

