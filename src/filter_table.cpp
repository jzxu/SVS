#include "filter_table.h"
#include "scene.h"
#include "relation.h"

using namespace std;

filter_table& get_filter_table() {
	static filter_table inst;
	return inst;
}

filter_table_entry *intersect_fill_entry();
filter_table_entry *distance_fill_entry();
filter_table_entry *centroid_distance_fill_entry();
filter_table_entry *distance_xyz_fill_entry();
filter_table_entry *smaller_fill_entry();
filter_table_entry *linear_fill_entry();
filter_table_entry *bbox_fill_entry();
filter_table_entry *bbox_int_fill_entry();
filter_table_entry *bbox_contains_fill_entry();
filter_table_entry *ontop_fill_entry();
filter_table_entry *north_of_fill_entry();
filter_table_entry *south_of_fill_entry();
filter_table_entry *east_of_fill_entry();
filter_table_entry *west_of_fill_entry();
filter_table_entry *x_aligned_fill_entry();
filter_table_entry *y_aligned_fill_entry();
filter_table_entry *z_aligned_fill_entry();
filter_table_entry *above_fill_entry();
filter_table_entry *below_fill_entry();
filter_table_entry *node_fill_entry();
filter_table_entry *all_nodes_fill_entry();
filter_table_entry *node_centroid_fill_entry();
filter_table_entry *compare_fill_entry();
filter_table_entry *absval_fill_entry();
filter_table_entry *vec3_fill_entry();
filter_table_entry *max_fill_entry();
filter_table_entry *closest_fill_entry();
filter_table_entry *has_property_fill_entry();
filter_table_entry *occlusion_fill_entry();

filter_table_entry *robot_driving_fill_entry();
filter_table_entry *robot_turning_right_fill_entry();
filter_table_entry *robot_turning_left_fill_entry();

filter_table::filter_table() {
	add(intersect_fill_entry());
	add(distance_fill_entry());
	add(centroid_distance_fill_entry());
	add(distance_xyz_fill_entry());
	add(smaller_fill_entry());
	add(linear_fill_entry());
	add(bbox_fill_entry());
	add(bbox_int_fill_entry());
	add(bbox_contains_fill_entry());
	add(ontop_fill_entry());
	add(north_of_fill_entry());
	add(south_of_fill_entry());
	add(east_of_fill_entry());
	add(west_of_fill_entry());
	add(x_aligned_fill_entry());
	add(y_aligned_fill_entry());
	add(z_aligned_fill_entry());
	add(above_fill_entry());
	add(below_fill_entry());
	add(node_fill_entry());
	add(all_nodes_fill_entry());
	add(node_centroid_fill_entry());
	add(compare_fill_entry());
	add(absval_fill_entry());
	add(vec3_fill_entry());
	add(max_fill_entry());
	add(closest_fill_entry());
	add(has_property_fill_entry());
	add(occlusion_fill_entry());
	
	add(robot_driving_fill_entry());
	add(robot_turning_right_fill_entry());
	add(robot_turning_left_fill_entry());
}

void filter_table::proxy_get_children(map<string, cliproxy*> &c) {
	c["timers"] = &timers;
	
	map<string, filter_table_entry*>::iterator i, iend;
	for (i = t.begin(), iend = t.end(); i != iend; ++i) {
		c[i->first] = i->second;
	}
}

template <typename T>
class single_combination_generator {
public:
	single_combination_generator(const std::vector<T> &elems, int n, bool ordered, bool allow_repeat)
	: elems(elems), indices(n), nelems(elems.size()), n(n), 
	  ordered(ordered), allow_repeat(allow_repeat), finished(false)
	{
		assert(n <= nelems);
		reset();
	}

	void reset() {
		if (!ordered && !allow_repeat) {
			for (int i = 0; i < n; ++i) {
				indices[i] = n - i - 1;
			}
		} else {
			fill(indices.begin(), indices.end(), 0);
		}
	}

	bool next(std::vector<T> &comb) {
		if (nelems == 0 || n == 0) {
			return false;
		}
		
		comb.resize(n);
		std::set<int> s;
		while (!finished) {
			bool has_repeat = false;
			s.clear();
			for (int i = 0; i < n; ++i) {
				comb[i] = elems[indices[i]];
				if (!has_repeat && !allow_repeat && ordered) {
					/*
					 incrementing technique guarantees no
					 repeats in the case ordered = false
					 and allow_repeats = false
					*/
					std::pair<std::set<int>::iterator, bool> p = s.insert(indices[i]);
					if (!p.second) {
						has_repeat = true;
					}
				}
			}
			increment(nelems - 1, 0);
			if (allow_repeat || !has_repeat) {
				return true;
			}
		}
		return false;
	}

private:
	int increment(int max, int i) {
		if (i == n - 1) {
			if (++indices[i] > max) {
				finished = true;
			}
			return indices[i];
		}
		if (++indices[i] > max) {
			if (ordered) {
				increment(max, i + 1);
				indices[i] = 0;
			} else {
				if (allow_repeat) {
					// maintain indices[i] >= indices[i+1]
					indices[i] = increment(max, i + 1);
				} else {
					// maintain indices[i] > indices[i+1]
					indices[i] = increment(max - 1, i + 1) + 1;
				}
			}
		}
		return indices[i];
	}

	const std::vector<T> &elems;
	std::vector<int> indices;
	int n, nelems;
	bool ordered, allow_repeat, finished;
};

filter* filter_table::make_filter(const string &pred, Symbol *root, soar_interface *si, scene *scn, filter_input *input) const
{
	map<std::string, filter_table_entry*>::const_iterator i = t.find(pred);
	if (i == t.end() || i->second->create == NULL) {
		return NULL;
	}
	return (*(i->second->create))(root, si, scn, input);
}

void filter_table::get_all_atoms(scene *scn, vector<string> &atoms) const {
	vector<const sgnode*> all_nodes;
	vector<string> all_node_names;
	scn->get_all_nodes(all_nodes);
	all_node_names.resize(all_nodes.size() - 1);
	for (int i = 1; i < all_nodes.size(); ++i) {  // don't use world node
		all_node_names[i-1] = all_nodes[i]->get_name();
	}
	
	map<string, filter_table_entry*>::const_iterator i, iend;
	for(i = t.begin(), iend = t.end(); i != iend; ++i) {
		const filter_table_entry *e = i->second;
		if (e->calc != NULL) {
			vector<string> args;
			single_combination_generator<string> gen(all_node_names, e->parameters.size(), e->ordered, e->allow_repeat);
			while (gen.next(args)) {
				stringstream ss;
				ss << e->name << "(";
				for (int j = 0; j < args.size() - 1; ++j) {
					ss << args[j] << ",";
				}
				ss << args.back() << ")";
				atoms.push_back(ss.str());
				args.clear();
			}
		}
	}
}

void filter_table::get_predicates(vector<string> &preds) const {
	map<string, filter_table_entry*>::const_iterator i, iend;
	for (i = t.begin(), iend = t.end(); i != iend; ++i) {
		preds.push_back(i->first);
	}
}

bool filter_table::get_params(const string &pred, vector<string> &p) const {
	map<string, filter_table_entry*>::const_iterator i = t.find(pred);
	if (i == t.end()) {
		return false;
	}
	p = i->second->parameters;
	return true;
}

void filter_table::add(filter_table_entry *e) {
	assert(t.find(e->name) == t.end());
	t[e->name] = e;
}

void filter_table::update_relations(const scene *scn, const vector<int> &dirty, int time, relation_table &rt) const {
	vector<const sgnode *> nodes;
	scn->get_all_nodes(nodes);
	nodes.erase(nodes.begin());
	
	map<string, filter_table_entry*>::const_iterator i, iend;
	for(i = t.begin(), iend = t.end(); i != iend; ++i) {
		const filter_table_entry *e = i->second;
		if (e->calc != NULL && nodes.size() >= e->parameters.size()) {
			relation &r = rt[e->name];
			if (r.arity() == 0) {
				// +1 for the time argument
				r.reset(e->parameters.size() + 1);
			}
			vector<const sgnode*> args;
			vector<int> arg_ids;
			single_combination_generator<const sgnode*> gen(nodes, e->parameters.size(), e->ordered, e->allow_repeat);
			while (gen.next(args)) {
				bool params_dirty = false;
				arg_ids.resize(args.size());
				for (int j = 0, jend = args.size(); j < jend; ++j) {
					arg_ids[j] = args[j]->get_id();
					if (has(dirty, args[j]->get_id())) {
						params_dirty = true;
					}
				}
				if (params_dirty) {
					timer &t = timers.get_or_add(i->first.c_str());
					t.start();
					bool pos = (*e->calc)(scn, args);
					t.stop();
					if (pos) {
						if (e->ordered) {
							r.add(time, arg_ids);
						} else {
							// true for all permutations
							single_combination_generator<int> gen2(arg_ids, arg_ids.size(), true, e->allow_repeat);
							tuple perm;
							while (gen2.next(perm)) {
								r.add(time, perm);
							}
						}
					}
				}
				args.clear();
			}
		}
	}
}

filter_table_entry::filter_table_entry()
: create(NULL), calc(NULL), ordered(false), allow_repeat(false)
{
	set_help("Reports information about this filter type.");
}

void filter_table_entry::proxy_use_sub(const vector<string> &args, ostream &os) {
	os << "parameters:";
	for (int i = 0, iend = parameters.size(); i < iend; ++i) {
		os << " " << parameters[i];
	}
	os << endl;
	if (ordered) {
		os << "ordered ";
	}
	if (allow_repeat) {
		os << "repeat ";
	}
	if (calc) {
		os << "basic";
	}
	os << endl;
}
