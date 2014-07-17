#include "filter.h"
#include "filter_table.h"
#include "scene.h"
#include "common.h"

using namespace std;

class distance_xyz_filter : public typed_map_filter<float> {
public:
    distance_xyz_filter(Symbol *root, soar_interface *si, filter_input *input) 
	: typed_map_filter<float>(root, si, input){}
    
    bool compute(const filter_params *params, bool adding, float &res, 
		 bool &changed) {
	float newres;
	const sgnode *a, *b;
	int axis = -1;
	vec3 amin, amax, bmin, bmax, ac, bc;
	double dist;
	
	if (!get_filter_param(this, params, "a", a) ||
	    !get_filter_param(this, params, "b", b) ||
	    !get_filter_param(this, params, "axis", axis))
	{
		changed = false;
	    return false;
	}
	
	ac = a->get_centroid();
	bc = b->get_centroid();
	bbox ba = a->get_bounds();
	bbox bb = b->get_bounds();
	ba.get_vals(amin, amax);
	bb.get_vals(bmin, bmax);
	
	if (amax[axis] <= bmin[axis])
	{
	    dist = abs(amax[axis] - bmin[axis]);
	}
	else if (bmax[axis] <= amin[axis])
	{
	    dist = abs(bmax[axis] - amin[axis]);
	}
	else if ((amax[axis] < bmax[axis] && amax[axis] > bmin[axis]) ||
		 (bmax[axis] < amax[axis] && bmax[axis] > amin[axis]) ||
		 (amax[axis] == bmax[axis]) || (bmin[axis] == amin[axis]))
	{
	    dist = 0.0; 
	}
	else
	{
	    std::cout << "Error: Object locations/axes info inconsistent" 
		      << std::endl;
	    dist = 0.0;
	}
	
	newres = dist;
	if (changed = (newres != res)) {
	    res = newres;
	}
	return true;
    }
};

filter *make_distance_xyz_filter(Symbol *root, soar_interface *si, scene *scn, filter_input *input) {
    return new distance_xyz_filter(root, si, input);
}

filter_table_entry *distance_xyz_fill_entry() {
    filter_table_entry *e = new filter_table_entry;
    e->name = "distance_xyz";
    e->create = &make_distance_xyz_filter;
    return e;
}
