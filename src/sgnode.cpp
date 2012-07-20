#include <assert.h>
#include <list>
#include <vector>
#include <algorithm>
#include "sgnode.h"

using namespace std;

typedef vector<sgnode*>::iterator childiter;
typedef vector<sgnode*>::const_iterator const_childiter;

sgnode::sgnode(std::string name, bool group)
: name(name), parent(NULL), group(group), trans_dirty(true), shape_dirty(true),
  pos(0.0, 0.0, 0.0), rot(0.0, 0.0, 0.0), scale(1.0, 1.0, 1.0)
{}


sgnode::~sgnode() {
	if (parent) {
		parent->detach_child(this);
	}
	send_update(sgnode::DELETED);
}

void sgnode::set_trans(char type, const vec3 &t) {
	switch (type) {
		case 'p':
			if (pos != t) {
				pos = t;
				set_transform_dirty();
			}
			break;
		case 'r':
			if (rot != t) {
				rot = t;
				set_transform_dirty();
			}
			break;
		case 's':
			if (scale != t) {
				scale = t;
				set_transform_dirty();
			}
			break;
		default:
			assert(false);
	}
}

void sgnode::set_trans(const vec3 &p, const vec3 &r, const vec3 &s) {
	if (pos != p || rot != r || scale != s) {
		pos = p;
		rot = r;
		scale = s;
		set_transform_dirty();
	}
}

vec3 sgnode::get_trans(char type) const {
	switch (type) {
		case 'p':
			return pos;
		case 'r':
			return rot;
		case 's':
			return scale;
		default:
			assert (false);
	}
	return vec3();
}

void sgnode::get_trans(vec3 &p, vec3 &r, vec3 &s) const {
	p = pos;
	r = rot;
	s = scale;
}

void sgnode::set_transform_dirty() {
	trans_dirty = true;
	if (parent) {
		parent->set_shape_dirty();
	}
	set_transform_dirty_sub();
	send_update(sgnode::TRANSFORM_CHANGED);
}

void sgnode::set_shape_dirty() {
	shape_dirty = true;
	if (parent) {
		parent->set_shape_dirty();
	}
	send_update(sgnode::SHAPE_CHANGED);
}

void sgnode::update_transform() {
	if (!trans_dirty) {
		return;
	}
	
	ltransform = transform3('p', pos) * transform3('r', rot) * transform3('s', scale);
	if (parent) {
		parent->update_transform();
		wtransform = parent->wtransform * ltransform;
	} else {
		wtransform = ltransform;
	}
}

/* if updates result in observers removing themselves, the iteration may
 * screw up, so make a copy of the std::list first */
void sgnode::send_update(sgnode::change_type t, int added) {
	std::list<sgnode_listener*>::iterator i;
	std::list<sgnode_listener*> c;
	std::copy(listeners.begin(), listeners.end(), back_inserter(c));
	for (i = c.begin(); i != c.end(); ++i) {
		(**i).node_update(this, t, added);
	}
}

void sgnode::listen(sgnode_listener *o) {
	listeners.push_back(o);
}

void sgnode::unlisten(sgnode_listener *o) {
	listeners.remove(o);
}

const bbox &sgnode::get_bounds() const {
	if (shape_dirty || trans_dirty) {
		const_cast<sgnode*>(this)->update_shape();
	}
	return bounds;
}

vec3 sgnode::get_centroid() const {
	if (shape_dirty || trans_dirty) {
		const_cast<sgnode*>(this)->update_shape();
	}
	return centroid;
}

void sgnode::set_bounds(const bbox &b) {
	bounds = b;
	centroid = bounds.get_centroid();
	shape_dirty = false;
}

const transform3 &sgnode::get_world_trans() const {
	if (trans_dirty) {
		const_cast<sgnode*>(this)->update_transform();
	}
	return wtransform;
}

group_node *sgnode::as_group() {
	group_node *g = dynamic_cast<group_node*>(this);
	assert(g);
	return g;
}

const group_node *sgnode::as_group() const {
	const group_node *g = dynamic_cast<const group_node*>(this);
	assert(g);
	return g;
}

sgnode *sgnode::clone() const {
	sgnode *c = clone_sub();
	c->set_trans(pos, rot, scale);
	return c;
}

group_node::~group_node() {
	childiter i;
	for (i = children.begin(); i != children.end(); ++i) {
		(**i).parent = NULL;  // so it doesn't try to detach itself
		delete *i;
	}
}

sgnode* group_node::clone_sub() const {
	group_node *c = new group_node(get_name());
	const_childiter i;
	for(i = children.begin(); i != children.end(); ++i) {
		c->attach_child((**i).clone());
	}
	return c;
}

sgnode* group_node::get_child(int i) {
	if (0 <= i && i < children.size()) {
		return children[i];
	}
	return NULL;
}

const sgnode *group_node::get_child(int i) const {
	if (0 <= i && i < children.size()) {
		return children[i];
	}
	return NULL;
}

void group_node::walk(vector<sgnode*> &result) {
	childiter i;
	result.push_back(this);
	for(i = children.begin(); i != children.end(); ++i) {
		(**i).walk(result);
	}
}

bool group_node::attach_child(sgnode *c) {
	children.push_back(c);
	c->parent = this;
	c->set_transform_dirty();
	set_shape_dirty();
	send_update(sgnode::CHILD_ADDED, children.size() - 1);
	
	return true;
}

void group_node::update_shape() {
	if (children.empty()) {
		vec3 c = get_world_trans()(vec3(0.0,0.0,0.0));
		set_bounds(bbox(c));
		return;
	}
	
	bbox b = children[0]->get_bounds();
	for (int i = 1; i < children.size(); ++i) {
		b.include(children[i]->get_bounds());
	}
	set_bounds(b);
}

void group_node::detach_child(sgnode *c) {
	childiter i;
	for (i = children.begin(); i != children.end(); ++i) {
		if (*i == c) {
			children.erase(i);
			set_shape_dirty();
			return;
		}
	}
}

void group_node::set_transform_dirty_sub() {
	for (childiter i = children.begin(); i != children.end(); ++i) {
		(**i).set_transform_dirty();
	}
}

convex_node::convex_node(const string &name, const ptlist &points)
: geometry_node(name), points(points), dirty(true)
{}

sgnode *convex_node::clone_sub() const {
	return new convex_node(get_name(), points);
}

void convex_node::update_shape() {
	set_bounds(bbox(get_world_points()));
}

void convex_node::set_transform_dirty_sub() {
	dirty = true;
}

const ptlist &convex_node::get_local_points() const {
	return points;
}

void convex_node::set_local_points(const ptlist &pts) {
	if (points != pts) {
		points = pts;
		dirty = true;
		set_shape_dirty();
	}
}

const ptlist &convex_node::get_world_points() const {
	if (dirty) {
		convex_node *nonconst = const_cast<convex_node*>(this);
		nonconst->world_points.clear();
		nonconst->world_points.reserve(points.size());
		transform(points.begin(), points.end(), back_inserter(nonconst->world_points), get_world_trans());
		nonconst->dirty = false;
	}
	return world_points;
}

void convex_node::get_shape_sgel(string &s) const {
	stringstream ss;
	ss << "v ";
	for (int i = 0; i < points.size(); ++i) {
		ss << points[i] << " ";
	}
	s = ss.str();
}

ball_node::ball_node(const string &name, double radius)
: geometry_node(name), radius(radius)
{}

void ball_node::get_shape_sgel(string &s) const {
	stringstream ss;
	ss << "b " << radius;
	s = ss.str();
}

sgnode *ball_node::clone_sub() const {
	return new ball_node(get_name(), radius);
}

/*
 This will overestimate the bounding box right now.
*/
void ball_node::update_shape() {
	transform3 t = get_world_trans();
	bbox bb(t(vec3(-radius,-radius,-radius)));
	bb.include(t(vec3(-radius,-radius, radius)));
	bb.include(t(vec3(-radius, radius,-radius)));
	bb.include(t(vec3(-radius, radius, radius)));
	bb.include(t(vec3( radius,-radius,-radius)));
	bb.include(t(vec3( radius,-radius, radius)));
	bb.include(t(vec3( radius, radius,-radius)));
	bb.include(t(vec3( radius, radius, radius)));
	set_bounds(bb);
}

void ball_node::set_radius(double r) {
	radius = r;
	set_shape_dirty();
}
