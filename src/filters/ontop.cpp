#include "filter.h"
#include "filter_table.h"
#include "common.h"
#include "scene.h"
#include "mat.h"

using namespace std;

/*
 Returns the percentage of the segment [a1, a2] that is overlapped by the
 segment [b1,b2]. This diagram shows the rationale of the calculation:

 a1                      a2
  |                      |
   <-ld->|        |<-rd->
         b1      b2
*/
double overlap_ratio(double a1, double a2, double b1, double b2) {
	double ld = max(0.0, b1 - a1);
	double rd = max(0.0, a2 - b2);
	double o = a2 - a1 - ld - rd;
	return o / (a2 - a1);
}

/*
 Conditions for T to be on top of B:
 - T and B must intersect
 - the bottom of T (z-axis) must be "close to" the top of B
 - B must significantly overlap T in the x and y axes.

 This definition is based on bounding boxes so is inherently flawed. But it
 works well enough for now.
*/
bool ontop(const sgnode *tn, const sgnode *bn) {
	vec3 tmin, tmax, bmin, bmax;
	
	const bbox &tb = tn->get_bounds();
	const bbox &bb = bn->get_bounds();
	
	tb.get_vals(tmin, tmax);
	bb.get_vals(bmin, bmax);
	double h1 = tmax(2) - tmin(2), h2 = bmax(2) - bmin(2);
	double margin = min(h1, h2) * .05;
	if (tmin[2] < bmax[2] - margin) {
		// too far inside
		return false;
	}
	
	if (overlap_ratio(tmin(0), tmax(0), bmin(0), bmax(0)) < 0.1 ||
	    overlap_ratio(tmin(1), tmax(1), bmin(1), bmax(1)) < 0.1)
	{
		return false;
	}
	return intersects(tn, bn);
}

bool standalone(const scene *scn, const vector<const sgnode*> &args) {
	assert(args.size() == 2);
	return ontop(args[0], args[1]);
}

class ontop_filter : public typed_map_filter<bool> {
public:
	ontop_filter(Symbol* root, soar_interface *si, filter_input *input) 
	: typed_map_filter<bool>(root, si, input)
	{}

	bool compute(const filter_params *params, bool adding, bool &res, bool &changed) {
		const sgnode *tn, *bn;
		if (!get_filter_param(this, params, "top", tn) || 
		    !get_filter_param(this, params, "bottom", bn))
		{
			return false;
		}
		bool newres = ontop(tn, bn);
		changed = (newres != res);
		res = newres;
		return true;
	}
};

filter* make_ontop_filter(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
	return new ontop_filter(root, si, input);
}

filter_table_entry *ontop_fill_entry() {
	filter_table_entry *e = new filter_table_entry;
	e->name = "on-top";
	e->parameters.push_back("top");
	e->parameters.push_back("bottom");
	e->ordered = true;
	e->allow_repeat = false;
	e->create = &make_ontop_filter;
	e->calc = &standalone;
	return e;
}
