#include <string>
#include <vector>
#include "filter_table.h"
#include "scene.h"
#include "common.h"

using namespace std;

string get_action(const sgnode *n) {
	string action;
	if (!map_get(n->get_string_properties(), string("action"), action)) {
		return "";
	}
	return action;
}

bool robot_driving(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 1);
	return get_action(args[0]) == "drive";
}

bool robot_turning_right(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 1);
	return get_action(args[0]) == "turn_right";
}

bool robot_turning_left(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 1);
	return get_action(args[0]) == "turn_left";
}

filter_table_entry *robot_driving_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "robot_driving";
	e->calc = &robot_driving;
	e->parameters.push_back("robot");
	return e;
}

filter_table_entry *robot_turning_right_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "robot_turning_right";
	e->calc = &robot_turning_right;
	e->parameters.push_back("robot");
	return e;
}

filter_table_entry *robot_turning_left_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "robot_turning_left";
	e->calc = &robot_turning_left;
	e->parameters.push_back("robot");
	return e;
}
