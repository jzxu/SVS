#include "filter_table.h"
#include "scene.h"

bool robot_driving(const scene *scn, const vector<const sgnode*> &args) {
	rvec props;
	scn->get_properties(props);
	return props(18) == 1.0;
}

bool robot_turning_right(const scene *scn, const vector<const sgnode*> &args) {
	rvec props;
	scn->get_properties(props);
	return props(18) == 1.1;
}

bool robot_turning_left(const scene *scn, const vector<const sgnode*> &args) {
	rvec props;
	scn->get_properties(props);
	return props(18) == 1.2;
}

filter_table_entry *robot_driving_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "robot_driving";
	e->calc = &robot_driving;
	e->parameters.push_back("dummy");
	return e;
}

filter_table_entry *robot_turning_right_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "robot_turning_right";
	e->calc = &robot_turning_right;
	e->parameters.push_back("dummy");
	return e;
}

filter_table_entry *robot_turning_left_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "robot_turning_left";
	e->calc = &robot_turning_left;
	e->parameters.push_back("dummy");
	return e;
}
