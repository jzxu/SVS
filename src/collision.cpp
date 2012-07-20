#include <iostream>
#include "collision.h"
#include "sgnode.h"

using namespace std;

btConvexHullShape *ptlist_to_hullshape(const ptlist &pts) {
	btConvexHullShape *s = new btConvexHullShape();
	for (int i = 0; i < pts.size(); ++i) {
		s->addPoint(btVector3(pts[i][0], pts[i][1], pts[i][2]));
	}
	s->setMargin(btScalar(0.0001));
	return s;
}

btCollisionShape *get_node_shape(const sgnode *n) {
	btCollisionShape *shape;
	const convex_node *cn = dynamic_cast<const convex_node*>(n);
	if (cn) {
		shape = ptlist_to_hullshape(cn->get_local_points());
		return shape;
	}
	const ball_node *bn = dynamic_cast<const ball_node*>(n);
	if (bn) {
		shape = new btSphereShape(bn->get_radius());
		return shape;
	}
	/*
	 If this is a group node, should make a compound shape in the
	 future.
	*/
	assert(false);
	return NULL;
}

void update_transforms(sgnode *n, btCollisionObject *cobj) {
	vec3 rpy = n->get_trans('r');
	btQuaternion q;
	q.setEuler(rpy[0], rpy[1], rpy[2]);
	cobj->getWorldTransform().setOrigin(to_btvec(n->get_trans('p')));
	cobj->getWorldTransform().setRotation(q);
	cobj->getCollisionShape()->setLocalScaling(to_btvec(n->get_trans('s')));
}

collision_detector::collision_detector()
: config(NULL), dispatcher(NULL), broadphase(NULL), cworld(NULL), dirty(true)
{
	timers.add("add_node");
	timers.add("del_node");
	timers.add("update_transform");
	timers.add("update_points");
	timers.add("update");
	timers.add("collision");
}

void collision_detector::init() {
	config = new btDefaultCollisionConfiguration();
	dispatcher = new btCollisionDispatcher(config);
	broadphase = new btDbvtBroadphase();
	cworld = new btCollisionWorld(dispatcher, broadphase, config);
	dirty = true;
}

collision_detector::~collision_detector() {
	delete cworld;
	delete broadphase;
	delete dispatcher;
	delete config;
}

void collision_detector::add_node(sgnode *n) {
	function_timer t(timers.get(ADD_NODE_T));
	assert(object_map.find(n) == object_map.end());
	
	if (!cworld) {
		init();
	}
	
	btCollisionObject *cobj = new btCollisionObject();
	cobj->setUserPointer(static_cast<void*>(n));
	cobj->setCollisionShape(get_node_shape(n));
	update_transforms(n, cobj);
	cworld->addCollisionObject(cobj);
	object_map[n] = cobj;
	dirty = true;
}

void collision_detector::del_node(sgnode *n) {
	function_timer t(timers.get(DEL_NODE_T));
	assert(object_map.find(n) != object_map.end());
	btCollisionObject *cobj = object_map[n];
	cworld->removeCollisionObject(cobj);
	delete cobj->getCollisionShape();
	delete cobj;
	object_map.erase(n);
	dirty = true;
}

void collision_detector::update_transform(sgnode *n) {
	function_timer t(timers.get(UPDATE_TRANSFORM_T));
	
	assert(object_map.find(n) != object_map.end());
	btCollisionObject *cobj = object_map[n];
	update_transforms(n, cobj);
	dirty = true;
}

void collision_detector::update_points(sgnode *n) {
	function_timer t(timers.get(UPDATE_POINTS_T));
	
	assert(object_map.find(n) != object_map.end());
	btCollisionObject *cobj = object_map[n];
	delete cobj->getCollisionShape();
	cobj->setCollisionShape(get_node_shape(n));
	dirty = true;
}

const collision_table &collision_detector::get_collisions() {
	function_timer t(timers.get(UPDATE_T));
	
	if (dirty) {
		results.clear();
		if (!cworld) {
			return results;
		}
		timers.start(COLLISION_T);
		cworld->performDiscreteCollisionDetection();
		timers.stop(COLLISION_T);
		int num_manifolds = dispatcher->getNumManifolds();
		for (int i = 0; i < num_manifolds; ++i) {
			btPersistentManifold *m = dispatcher->getManifoldByIndexInternal(i);
			if (m->getNumContacts() > 0) {
				btCollisionObject *a = static_cast<btCollisionObject*>(m->getBody0());
				btCollisionObject *b = static_cast<btCollisionObject*>(m->getBody1());
				sgnode *na = static_cast<sgnode*>(a->getUserPointer());
				sgnode *nb = static_cast<sgnode*>(b->getUserPointer());
				results.insert(make_pair(na, nb));
			}
			m->clearManifold();
		}
		dirty = false;
	}
	return results;
}

