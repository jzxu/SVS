#include <stdlib.h>
#include <ctype.h>
#include <sstream>
#include <limits>

#include "command.h"
#include "command_table.h"
#include "svs.h"
#include "scene.h"
#include "soar_interface.h"

using namespace std;

command_table& get_command_table(){
  static command_table inst;
  return inst;
}

command *_make_extract_command_(svs_state *state, Symbol *root);
command *_make_project_command_(svs_state *state, Symbol *root);
command *_make_extract_once_command_(svs_state *state, Symbol *root);
command *_make_add_node_command_(svs_state *state, Symbol *root);
command *_make_create_model_command_(svs_state *state, Symbol *root);
command *_make_assign_model_command_(svs_state *state, Symbol *root);
command *_make_property_command_(svs_state *state, Symbol *root);
command *_make_seek_command_(svs_state *state, Symbol *root);
command *_make_random_control_command_(svs_state *state, Symbol *root);
command *_make_copy_node_command_(svs_state *state, Symbol *root);
command *_make_del_node_command_(svs_state *state, Symbol *root);

command_table::command_table(){
  table["extract"] = &_make_extract_command_;
	table["project"] = &_make_project_command_;
	table["extract_once"] = &_make_extract_once_command_;
	table["add_node"] = &_make_add_node_command_;
	table["seek"] = &_make_seek_command_;
	table["random_control"] = &_make_random_control_command_;
	table["create-model"] = &_make_create_model_command_;
	table["assign-model"] = &_make_assign_model_command_;
	table["property"] = &_make_property_command_;
	table["copy_node"] = &_make_copy_node_command_;
	table["del_node"] = &_make_del_node_command_;
}

command* command_table::make_command(svs_state *state, wme *w) {
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
  
  map<string, make_command_fp*>::iterator i = table.find(name);
  if(i != table.end()){
    return i->second(state, id);
  } else {
    return NULL;
  }
}

