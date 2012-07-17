#include <string>
#include "command.h"
#include "scene.h"
#include "filter.h"
#include "svs.h"

using namespace std;

class add_node_command : public command, public sgnode_listener {
public:
	add_node_command(svs_state *state, Symbol *root)
	: command(state, root), root(root), scn(state->get_scene()), node_filter(NULL), parent(NULL)
	{
		si = state->get_svs()->get_soar_interface();
	}
	
	~add_node_command() {
		reset();
	}
	
	string description() {
		return string("add-node");
	}
	
	bool update_sub() {
		wme *parent_wme, *gen_wme;
		bool c = changed();
		
		if (!parent || c) {
			reset();
			if (!si->find_child_wme(root, "parent", parent_wme) ||
			    !si->find_child_wme(root, "node", gen_wme))
			{
				set_status("missing parameters");
				return false;
			}
			
			string pname;
			if (!si->get_val(si->get_wme_val(parent_wme), pname)) {
				set_status("parent name must be a string");
				return false;
			}
			if (!parent || parent->get_name() != pname) {
				if (parent) {
					parent->unlisten(this);
				}
				
				if (!(parent = scn->get_node(pname))) {
					set_status("parent node doesn't exist");
					return false;
				}
				parent->listen(this);
			}
			
			if ((node_filter = parse_filter_spec(si, si->get_wme_val(gen_wme), scn)) == NULL) {
				set_status("incorrect node filter syntax");
				return false;
			}
		}
		if (node_filter) {
			if (!proc_changes()) {
				return false;
			}
			set_status("success");
			return true;
		}
		return false;
	}
	
	bool early() { return false; }
	
	void node_update(sgnode *n, sgnode::change_type t, int added_child) {
		if (t == sgnode::DELETED) {
			parent = NULL;
		}
	}

private:
	bool proc_changes() {
		filter_result *res;
		
		if (!node_filter->update()) {
			set_status(node_filter->get_error());
			return false;
		}
		res = node_filter->get_result();
		for (int i = res->first_added(); i < res->num_current(); ++i) {
			if (!add_node(res->get_current(i))) {
				return false;
			}
		}
		for (int i = 0; i < res->num_removed(); ++i) {
			if (!del_node(res->get_removed(i))) {
				return false;
			}
		}
		res->clear_changes();
		return true;
	}
	
	bool add_node(filter_val *v) {
		sgnode *n;
		stringstream ss;
		if (!get_filter_val(v, n)) {
			set_status("filter result must be a node");
			return false;
		}
		if (!scn->add_node(parent->get_name(), n)) {
			ss << "error adding node " << n->get_name() << " to scene";
			set_status(ss.str());
			return false;
		}
		return true;
	}
	
	bool del_node(filter_val *v) {
		sgnode *n;
		stringstream ss;
		if (!get_filter_val(v, n)) {
			set_status("filter result must be a node");
			return false;
		}
		if (!scn->del_node(n->get_name())) {
			ss << "error deleting node " << n->get_name() << " from scene";
			set_status(ss.str());
			return false;
		}
		return true;
	}
	
	void reset() {
		if (parent) {
			parent->unlisten(this);
		}
		parent = NULL;
		
		if (node_filter) {
			delete node_filter;
			node_filter = NULL;
		}
	}
	
	scene             *scn;
	Symbol            *root;
	soar_interface    *si;
	sgnode            *parent;
	filter            *node_filter;
};

command *_make_add_node_command_(svs_state *state, Symbol *root) {
	return new add_node_command(state, root);
}
