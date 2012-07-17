#include <stdlib.h>
#include <ctype.h>
#include <sstream>
#include <limits>

#include "command.h"
#include "filter.h"
#include "filter_table.h"
#include "svs.h"
#include "scene.h"
#include "soar_interface.h"

using namespace std;

bool is_reserved_param(const string &name) {
	return name == "result" || name == "parent";
}

/* Remove weird characters from string */
void cleanstring(string &s) {
	string::iterator i;
	for (i = s.begin(); i != s.end();) {
		if (!isalnum(*i) && *i != '.' && *i != '-' && *i != '_') {
			i = s.erase(i);
		} else {
			++i;
		}
	}
}

command::command(svs_state *state, Symbol *cmd_root)
: state(state), si(state->get_svs()->get_soar_interface()), root(cmd_root), 
  subtree_size(0), prev_max_time(-1), status_wme(NULL), first(true)
{
	timers.add("update");
}

command::~command() {}

bool command::changed() {
	int size, max_time;
	parse_substructure(size, max_time);
	if (first || size != subtree_size || max_time > prev_max_time) {
		first = false;
		subtree_size = size;
		prev_max_time = max_time;
		return true;
	}
	return false;
}

void command::parse_substructure(int &size, int &max_time) {
	tc_num tc;
	stack< Symbol *> to_process;
	wme_list childs;
	wme_list::iterator i;
	Symbol *parent, *v;
	int tt;
	string attr;

	tc = si->new_tc_num();
	size = 0;
	max_time = -1;
	to_process.push(root);
	while (!to_process.empty()) {
		parent = to_process.top();
		to_process.pop();
		
		si->get_child_wmes(parent, childs);
		for (i = childs.begin(); i != childs.end(); ++i) {
			if (parent == root) {
				if (si->get_val(si->get_wme_attr(*i), attr) && 
				    (attr == "result" || attr == "status"))
				{
					/* result wmes are added by svs */
					continue;
				}
			}
			v = si->get_wme_val(*i);
			tt = si->get_timetag(*i);
			size++;
			
			if (tt > max_time) {
				max_time = tt;
			}

			if (si->is_identifier(v) && si->get_tc_num(v) != tc) {
				si->set_tc_num(v, tc);
				to_process.push(v);
			}
		}
	}
}

bool command::get_str_param(const string &name, string &val) {
	wme_list children;
	wme_list::iterator i;
	string attr, v;
	
	si->get_child_wmes(root, children);
	for(i = children.begin(); i != children.end(); ++i) {
		if (si->get_val(si->get_wme_attr(*i), attr)) {
			if (name != attr) {
				continue;
			}
			if (si->get_val(si->get_wme_val(*i), v)) {
				val = v;
				return true;
			}
		}
	}
	return false;
}

void command::set_status(const string &s) {
	if (curr_status == s) {
		return;
	}
	if (status_wme) {
		si->remove_wme(status_wme);
	}
	status_wme = si->make_wme(root, "status", s);
	curr_status = s;
}

command *_make_extract_command_(svs_state *state, Symbol *root);
command *_make_add_node_command_(svs_state *state, Symbol *root);
command *_make_create_model_command_(svs_state *state, Symbol *root);
command *_make_assign_model_command_(svs_state *state, Symbol *root);
command *_make_property_command_(svs_state *state, Symbol *root);
command *_make_seek_command_(svs_state *state, Symbol *root);
command *_make_random_control_command_(svs_state *state, Symbol *root);
command *_make_manual_control_command_(svs_state *state, Symbol *root);

command* make_command(svs_state *state, wme *w) {
	string name;
	Symbol *id;
	soar_interface *si;
	
	si = state->get_svs()->get_soar_interface();
	if (!si->get_val(si->get_wme_attr(w), name)) {
		return NULL;
	}
	if (!si->is_identifier(si->get_wme_val(w))) {
		return NULL;
	}
	id = si->get_wme_val(w);
	if (name == "extract") {
		return _make_extract_command_(state, id);
	} else if (name == "add_node") {
		return _make_add_node_command_(state, id);
	} else if (name == "seek") {
		return _make_seek_command_(state, id);
	} else if (name == "random_control") {
		return _make_random_control_command_(state, id);
	} else if (name == "manual_control") {
		return _make_manual_control_command_(state, id);
	} else if (name == "create-model") {
		return _make_create_model_command_(state, id);
	} else if (name == "assign-model") {
		return _make_assign_model_command_(state, id);
	} else if (name == "property") {
		return _make_property_command_(state, id);
	}
	return NULL;
}

/*
 Example input:
 
 (<ot> ^type on-top ^a <ota> ^b <otb>)
 (<ota> ^type node ^name box1)
 (<otb> ^type node ^name box2)
*/
filter *parse_filter_spec(soar_interface *si, Symbol *root, scene *scn) {
	wme_list children, params;
	wme_list::iterator i;
	Symbol* cval;
	string strval, pname, ftype, itype;
	long intval;
	float floatval;
	filter_input *input;
	bool fail;
	filter *f;
	
	fail = false;
	si->get_child_wmes(root, children);
	for (i = children.begin(); i != children.end(); ++i) {
		if (!si->get_val(si->get_wme_attr(*i), pname)) {
			continue;
		}
		cval = si->get_wme_val(*i);
		if (pname == "type") {
			if (!si->get_val(cval, ftype)) {
				return NULL;
			}
		} else if (pname == "input-type") {
			if (!si->get_val(cval, itype)) {
				return NULL;
			}
		} else if (pname != "status" && pname != "result") {
			params.push_back(*i);
		}
	}
	
	// The combine type check is a bit of a hack
	if (itype == "concat" || ftype == "combine") {
		input = new concat_filter_input();
	} else if (params.size() == 0) {
		input = new null_filter_input();
	} else {
		input = new product_filter_input();
	}
	
	for (i = params.begin(); i != params.end(); ++i) {
		if (!si->get_val(si->get_wme_attr(*i), pname)) {
			continue;
		}
		cval = si->get_wme_val(*i);
		if (si->get_val(cval, strval)) {
			input->add_param(pname, new const_filter<string>(strval));
		} else if (si->get_val(cval, intval)) {
			input->add_param(pname, new const_filter<int>(intval));
		} else if (si->get_val(cval, floatval)) {
			input->add_param(pname, new const_filter<float>(floatval));
		} else {
			filter *cf;
			// must be identifier
			if ((cf = parse_filter_spec(si, cval, scn)) == NULL) {
				fail = true;
				break;
			}
			input->add_param(pname, cf);
		}
	}
	
	if (!fail) {
		if (ftype == "combine") {
			f = new passthru_filter(input);
		} else {
			f = get_filter_table().make_filter(ftype, scn, input);
		}
	}
	
	if (fail || ftype == "" || f == NULL) {
		delete input;
		return NULL;
	}
	return f;
}
