#include <iostream>
#include "command.h"
#include "filter.h"
#include "svs.h"
#include "soar_interface.h"

using namespace std;

class extract_command : public command, public filter_input_listener {
public:
	extract_command(svs_state *state, Symbol *root, bool once)
	: command(state, root), root(root), state(state), fltr(NULL), res(NULL), res_root(NULL), first(true), once(once)
	{
		si = state->get_svs()->get_soar_interface();
	}
	
	~extract_command() {
		if (fltr) {
			delete fltr;
		}
	}
	
	string description() {
		return string("extract");
	}
	
	bool update_sub() {
		if (!res_root) {
			res_root = si->get_wme_val(si->make_id_wme(root, "result"));
		}
		
		if (changed()) {
			clear_results();
			if (fltr) {
				delete fltr;
			}
			
			fltr = parse_filter_spec(state->get_svs()->get_soar_interface(), root, state->get_scene());
			if (!fltr) {
				set_status("incorrect filter syntax");
				return false;
			}
			res = fltr->get_result();
			fltr->listen_for_input(this);
			first = true;
		}
		
		if (fltr && (!once || first)) {
			if (!fltr->update()) {
				clear_results();
				return false;
			}
			update_results();
			res->clear_changes();
			first = false;
		}
		return true;
	}
	
	bool early() { return false; }
	
	void reset_results() {
		clear_results();
		for (int i = 0; i < res->num_current(); ++i) {
			handle_result(res->get_current(i));
		}
		res->clear_changes();
	}
	
	void update_results() {
		wme *w;
		
		for (int i = res->first_added(); i < res->num_current(); ++i) {
			handle_result(res->get_current(i));
		}
		for (int i = 0; i < res->num_removed(); ++i) {
			filter_val *fv = res->get_removed(i);
			record r;
			if (!map_pop(records, fv, r)) {
				assert(false);
			}
			si->remove_wme(r.rec_wme);
		}
		for (int i = 0; i < res->num_changed(); ++i) {
			handle_result(res->get_changed(i));
		}
	}
	
	void clear_results() {
		record_map::iterator i;
		for (i = records.begin(); i != records.end(); ++i) {
			si->remove_wme(i->second.rec_wme);
		}
		records.clear();
	}
	
private:
	Symbol *make_filter_val_sym(filter_val *v) {
		int iv;
		double fv;
		bool bv;
		
		if (get_filter_val(v, iv)) {
			return si->make_sym(iv);
		}
		if (get_filter_val(v, fv)) {
			return si->make_sym(fv);
		}
		if (get_filter_val(v, bv)) {
			return si->make_sym(bv ? "t" : "f");
		}
		return si->make_sym(v->get_string());
	}
	
	bool sym_reps_filter_val(Symbol *s, const filter_val *fv) {
		long fiv, siv;
		double ffv, sfv;
		bool fbv;
		string str;
		if (get_filter_val(fv, fiv)) {
			return (si->get_val(s, siv) && siv == fiv);
		}
		if (get_filter_val(fv, ffv)) {
			return (si->get_val(s, sfv) && sfv == ffv);
		}
		if (get_filter_val(fv, fbv)) {
			return (si->get_val(s, str) && ((fbv && str == "t") || (!fbv && str == "f")));
		}
		return (si->get_val(s, str) && str == fv->get_string());
	}
	
	wme *make_value_wme(filter_val *v, Symbol *root) {
		return si->make_wme(root, "value", make_filter_val_sym(v));
	}
	
	void update_param_struct(const filter_params *p, Symbol *pid) {
		filter_params::const_iterator j;
		for (j = p->begin(); j != p->end(); ++j) {
			wme *pwme = NULL;
			if (!si->find_child_wme(pid, j->first, pwme) ||
			    !sym_reps_filter_val(si->get_wme_val(pwme), j->second))
			{
				if (pwme) {
					si->remove_wme(pwme);
				}
				si->make_wme(pid, j->first, make_filter_val_sym(j->second));
			}
		}
	}
	
	void make_record(filter_val *result) {
		record r;
		r.rec_wme = si->make_id_wme(res_root, "record");
		r.rec_id = si->get_wme_val(r.rec_wme);
		r.val_wme = make_value_wme(result, r.rec_id);
		r.params_wme = si->make_id_wme(r.rec_id, "params");
		fltr->get_result_params(result, r.params);
		if (r.params) {
			update_param_struct(r.params, si->get_wme_val(r.params_wme));
		}
		records[result] = r;
	}
	
	void handle_result(filter_val *result) {
		record *r;
		if (r = map_get(records, result)) {
			si->remove_wme(r->val_wme);
			r->val_wme = make_value_wme(result, r->rec_id);
		} else {
			make_record(result);
		}
	}
	
	void handle_ctlist_change(const filter_params *p) {
		record_map::iterator i;
		for (i = records.begin(); i != records.end(); ++i) {
			if (i->second.params == p) {
				Symbol *pid = si->get_wme_val(i->second.params_wme);
				update_param_struct(p, pid);
				return;
			}
		}
		assert(false);
	}
	
	Symbol         *root;
	Symbol         *res_root;
	Symbol         *pos_root;  // identifier for positive atoms
	Symbol         *neg_root;  // identifier for negative atoms
	svs_state      *state;
	soar_interface *si;
	filter         *fltr;
	filter_result  *res;
	bool            first, once;
	
	struct record {
		const filter_params *params;
		wme *rec_wme;
		wme *val_wme;
		wme *params_wme;
		Symbol *rec_id;
	};
	
	typedef map<filter_val*, record> record_map;
	record_map records;
};

command *_make_extract_command_(svs_state *state, Symbol *root) {
	return new extract_command(state, root, false);
}

command *_make_extract_once_command_(svs_state *state, Symbol *root) {
	return new extract_command(state, root, true);
}

