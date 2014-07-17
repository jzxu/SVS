#include <vector>
#include <string>
#include "sgnode.h"
#include "scene.h"
#include "filter.h"
#include "filter_table.h"
#include "common.h"

using namespace std;

/*
 Percentage of the smaller of two bodies that they can overlap before being
 considered overlapping.
*/
const double OVERLAP_MARGIN = .05;

/*
Calculates whether a bounding box is located to the left (-1), overlapping
(0), or right (1) of another bounding box along the specified axis.
*/
bool direction(const sgnode *a, const sgnode *b, int axis, int comp) {
	assert(0 <= axis && axis < 3);

	int dir;
	vec3 amin, amax, bmin, bmax;
	double asize, bsize, margin;

	a->get_bounds().get_vals(amin, amax);
	b->get_bounds().get_vals(bmin, bmax);

	asize = amax(axis) - amin(axis);
	bsize = bmax(axis) - bmin(axis);
	margin = OVERLAP_MARGIN * ( asize < bsize ? asize : bsize );

	/*
	 dir = [-1, 0, 1] if a is [less than, overlapping,
	 greater than] b in that dimension.
	*/
	if (amax[axis] <= bmin[axis] + margin) {
		dir = -1;
	} else if (bmax[axis] <= amin[axis] + margin) {
		dir = 1;
	} else {
		dir = 0;
	}
	return comp == dir;
}

// AM: Note, changed so it uses the volume as reported by the scale
// This assumes unit sized bounding volumes or spheres as the vertices
bool size_comp(const sgnode *a, const sgnode *b) {
	int i, dir[3];
    vec3 sa = a->get_trans('s');
    vec3 sb = b->get_trans('s');
    double va = sa[0] * sa[1] * sa[2];
    double vb = sb[0] * sb[1] * sb[2];

    return (va * 1.05) < vb;
	//vec3 amin, amax, bmin, bmax;

	//a->get_bounds().get_vals(amin, amax);
	//b->get_bounds().get_vals(bmin, bmax);
	//float adiag = (amax-amin).norm();
	//float bdiag = (bmax-bmin).norm();
	//float atopsquare = (amax[0]-amin[0])*(amax[1]-amin[1]);
	//float btopsquare = (bmax[0]-bmin[0])*(bmax[1]-bmin[1]);

	//return (atopsquare*1.05) < btopsquare;
}

bool linear_comp(const sgnode *a, const sgnode *b, const sgnode *c) {

/*	for now ignore z
1/2[x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
*/
	vec3 ca = a->get_centroid();
	vec3 cb = b->get_centroid();
	vec3 cc = c->get_centroid();

	float tri_area = 0.5*(ca[0]*(cb[1]-cc[1]) + cb[0]*(cc[1]-ca[1]) +
						  cc[0]*(ca[1]-cb[1]));
	if (tri_area < 0)
		tri_area = -tri_area;
	//expecting b parameter to be in middle or fail
	if (((ca - cb).norm() >= (ca-cc).norm()) ||
		((cb - cc).norm() >= (ca-cc).norm()))
		tri_area = 1;
	return (tri_area < 0.000001);
}

bool north_of(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return direction(args[0], args[1], 1, 1);
}
bool smaller(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return size_comp(args[0], args[1]);
}
bool linear(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 3);
	return linear_comp(args[0], args[1], args[2]);
}

bool south_of(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return direction(args[0], args[1], 1, -1);
}

bool east_of(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return direction(args[0], args[1], 0, 1);
}

bool west_of(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return direction(args[0], args[1], 0, -1);
}

bool x_aligned(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return direction(args[0], args[1], 0, 0);
}

bool y_aligned(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return direction(args[0], args[1], 1, 0);
}

bool z_aligned(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return direction(args[0], args[1], 2, 0);
}

bool above(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return direction(args[0], args[1], 2, 1);
}

bool below(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return direction(args[0], args[1], 2, -1);
}

/*
Filter version
*/

class direction_filter : public typed_select_filter<const sgnode*>{ 
public:
	direction_filter(Symbol *root, soar_interface *si, filter_input *input, int axis, int comp)
	  : typed_select_filter<const sgnode*>(root, si, input), axis(axis), comp(comp) {}

  bool compute(const filter_params *p, bool null_out, const sgnode* &out, bool &select, bool &changed) {
		const sgnode *a, *b;
		double top,bot;
		if (!get_filter_param(this, p, "a", a)) {
			return false;
		}
		if (!get_filter_param(this, p, "b", b)) {
			return false;
		}
		//top bot, parameters optional
		top = 100.0;
		bot = 0.0;

		bool newres = direction(a, b, axis, comp);

		if (newres && (comp != 0))
		{
			get_filter_param(this, p, "top", top);
			get_filter_param(this, p, "bot", bot);
			vec3 amin, amax, bmin, bmax, ac, bc;
			double dist;

			ac = a->get_centroid();
			bc = b->get_centroid();
			bbox ba = a->get_bounds();
			bbox bb = b->get_bounds();
			ba.get_vals(amin, amax);
			bb.get_vals(bmin, bmax);

			if (amax[axis] <= bmin[axis])
				dist = abs(amax[axis] - bmin[axis]);
			else if (bmax[axis] <= amin[axis])
				dist = abs(bmax[axis] - amin[axis]);
			/*
			else if ((amax[axis] < bmax[axis] && amax[axis] > bmin[axis]) ||
					 (bmax[axis] < amax[axis] && bmax[axis] > amin[axis]) ||
					 (amax[axis] == bmax[axis]) || (bmin[axis] == amin[axis]))
				dist = 0.0;
			*/
			else
				dist = 0.0;
			newres = ((dist >= bot) && (dist <= top));
		}


		//res = newres;
		changed = true;
		out = b;
		select = newres;
		/*
		if (newres && out == NULL) {
			
			out = new filter_val_c<const sgnode*>(b);
		} else if(newres && a_val != out) {
			
			set_filter_val(out, b);
		} else if(!newres && out != NULL) {
			// We no longer are selecting the value, make it null
			out = NULL;
		} else {
			// the value didn't actually changed
			changed = false;
		}
		*/
		//delete a_val;
		return true;
	}

	virtual int getAxis() {
		return axis;
	}
	virtual int getComp() {
		return comp;
	}

private:
	int axis, comp;
};





class size_comp_filter : public select_filter{//typed_map_filter<bool> {
public:
	size_comp_filter(Symbol *root, soar_interface *si, filter_input *input)
	: select_filter(root, si, input) {}

	bool compute(const filter_params *p, filter_val*& out, bool &changed) {//bool adding, bool &res, bool &changed) {
		const sgnode *a, *b;

		if (!get_filter_param(this, p, "a", a)) {
			return false;
		}
		if (!get_filter_param(this, p, "b", b)) {
			return false;
		}

		bool newres = size_comp(a, b);
		changed = false;
		filter_val* a_val = new filter_val_c<const sgnode*>(b);
		if(newres && out == NULL){
			// Create a new filter val
			out = new filter_val_c<const sgnode*>(b);
		} else if(newres && a_val != out){
			// The value has changed
			set_filter_val(out, b);
		} else if(!newres && out != NULL){
			// We no longer are selecting the value, make it null
			out = NULL;
		} else {
			// the value didn't actually changed
			changed = false;
		}
		delete a_val;

		//res = newres;
		return true;
	}
};
class linear_comp_filter : public select_filter{//typed_map_filter<bool> {
public:
	linear_comp_filter(Symbol *root, soar_interface *si, filter_input *input)
		: select_filter(root, si, input) {}

	bool compute(const filter_params *p, filter_val*& out, bool &changed) {//bool adding, bool &res, bool &changed) {
		const sgnode *a, *b, *c;

		if (!get_filter_param(this, p, "a", a)) {
			return false;
		}
		if (!get_filter_param(this, p, "b", b)) {
			return false;
		}
		if (!get_filter_param(this, p, "c", c)) {
			return false;
		}

	    bool newres = linear_comp(a, b, c);
	changed = false;
		filter_val* a_val = new filter_val_c<const sgnode*>(b);
		if(newres && out == NULL){
			// Create a new filter val
			out = new filter_val_c<const sgnode*>(b);
		} else if(newres && a_val != out){
			// The value has changed
			set_filter_val(out, b);
		} else if(!newres && out != NULL){
			// We no longer are selecting the value, make it null
			out = NULL;
		} else {
			// the value didn't actually changed
			changed = false;
		}
		delete a_val;

	//res = newres;
		return true;
	}
};

filter *make_north_of(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new direction_filter(root, si, input, 1, 1);
}
filter *make_smaller(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new size_comp_filter(root, si, input);
}
filter *make_linear(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new linear_comp_filter(root, si, input);
}

filter *make_south_of(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new direction_filter(root, si, input, 1, -1);
}

filter *make_east_of(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new direction_filter(root, si, input, 0, 1);
}

filter *make_west_of(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new direction_filter(root, si, input, 0, -1);
}

filter *make_x_aligned(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new direction_filter(root, si, input, 0, 0);
}

filter *make_y_aligned(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new direction_filter(root, si, input, 1, 0);
}

filter *make_z_aligned(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new direction_filter(root, si, input, 2, 0);
}

filter *make_above(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new direction_filter(root, si, input, 2, 1);
}

filter *make_below(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new direction_filter(root, si, input, 2, -1);
}


filter_table_entry *smaller_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "smaller-than";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = true;
	e->allow_repeat = false;
	e->create = &make_smaller;
	e->calc = &smaller;
	return e;
}

filter_table_entry *linear_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "linear-with";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->parameters.push_back("c");
	e->ordered = true;
	e->allow_repeat = false;
	e->create = &make_linear;
	e->calc = &linear;
	return e;
}

filter_table_entry *north_of_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "y-greater-than";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = true;
	e->allow_repeat = false;
	e->create = &make_north_of;
	e->calc = &north_of;
	return e;
}

filter_table_entry *south_of_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "y-less-than";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = true;
	e->allow_repeat = false;
	e->create = &make_south_of;
	e->calc = &south_of;
	return e;
}

filter_table_entry *east_of_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "x-greater-than";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = true;
	e->allow_repeat = false;
	e->create = &make_east_of;
	e->calc = &east_of;
	return e;
}

filter_table_entry *west_of_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "x-less-than";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = true;
	e->allow_repeat = false;
	e->create = &make_west_of;
	e->calc = &west_of;
	return e;
}

filter_table_entry *x_aligned_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "x-aligned";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = false;
	e->allow_repeat = false;
	e->create = &make_x_aligned;
	e->calc = &x_aligned;
	return e;
}

filter_table_entry *y_aligned_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "y-aligned";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = false;
	e->allow_repeat = false;
	e->create = &make_y_aligned;
	e->calc = &y_aligned;
	return e;
}

filter_table_entry *z_aligned_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "z-aligned";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = false;
	e->allow_repeat = false;
	e->create = &make_z_aligned;
	e->calc = &z_aligned;
	return e;
}

filter_table_entry *above_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "z-greater-than";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = true;
	e->allow_repeat = false;
	e->create = &make_above;
	e->calc = &above;
	return e;
}

filter_table_entry *below_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "z-less-than";
	e->parameters.push_back("a");
	e->parameters.push_back("b");
	e->ordered = true;
	e->allow_repeat = false;
	e->create = &make_below;
	e->calc = &below;
	return e;
}

