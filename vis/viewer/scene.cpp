#include <cstdlib>
#include <cassert>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include <osg/Node>
#include <osg/Group>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/PolygonMode>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>
#include <osg/Shape>
#include <osg/Material>
#include <osgText/Font>
#include <osgText/Text>

#include "scene.h"

using namespace std;
using namespace osg;

const char *FONT = "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf";
const double AXIS_RADIUS = 0.005;
const double AXIS_LEN = 1.0;
 
/*
 Execute qhull to calculate the convex hull of pts.
*/
int qhull(const vector<Vec3> &pts, vector<vector<int> > &facets) {
	char *end;
	
	FILE *p = popen("qhull i >/tmp/qhull", "w");
	fprintf(p, "3\n%ld\n", pts.size());
	for (int i = 0; i < pts.size(); ++i) {
		fprintf(p, "%f %f %f\n", pts[i][0], pts[i][1], pts[i][2]);
	}
	int ret = pclose(p);
	if (ret != 0) {
		return ret;
	}
	
	ifstream output("/tmp/qhull");
	string line;
	getline(output, line);
	int nfacets = strtol(line.c_str(), &end, 10);
	if (*end != '\0') {
		return 1;
	}
	while (getline(output, line)) {
		const char *start = line.c_str();
		vector<int> facet;
		while (true) {
			int x = strtol(start, &end, 10);
			if (end == start) {
				break;
			}
			facet.push_back(x);
			start = end;
		}
		facets.push_back(facet);
	}
	assert (facets.size() == nfacets);
	return 0;
}

Quat to_quaternion(const Vec3 &rpy) {
	double halfroll = rpy[0] / 2;
	double halfpitch = rpy[1] / 2;
	double halfyaw = rpy[2] / 2;

	double sin_r2 = sin( halfroll );
	double sin_p2 = sin( halfpitch );
	double sin_y2 = sin( halfyaw );

	double cos_r2 = cos( halfroll );
	double cos_p2 = cos( halfpitch );
	double cos_y2 = cos( halfyaw );

	return Quat(sin_r2 * cos_p2 * cos_y2 - cos_r2 * sin_p2 * sin_y2,
	            cos_r2 * sin_p2 * cos_y2 + sin_r2 * cos_p2 * sin_y2,
	            cos_r2 * cos_p2 * sin_y2 - sin_r2 * sin_p2 * cos_y2,
	            cos_r2 * cos_p2 * cos_y2 + sin_r2 * sin_p2 * sin_y2);
}

node::node(const string &name, const string &parent) 
: name(name), parent(parent), trans(new PositionAttitudeTransform()), group(new Group)
{
	if (name != "world") {
		create_label();
	}
	trans->addChild(group);
}

node::node(const string &name, const string &parent, const vector<Vec3> &verts)
: name(name), parent(parent), trans(new PositionAttitudeTransform())
{
	create_label();
	create_axes();
	if (verts.size() == 0) {
		group = new Group;
		trans->addChild(group);
	} else {
		leaf = new Geode;
		set_vertices(verts);
		scribe = new osgFX::Scribe;
		scribe->addChild(leaf);
		scribe->setWireframeColor(Vec4(0.0, 0.0, 0.0, 1.0));
		trans->addChild(scribe);
	}
}

void node::create_axes() {
	axes = new Group;
	for (int i = 0; i < 3; ++i) {
		ref_ptr<Geode> g = new Geode;
		ref_ptr<Material> m = new Material;
		ref_ptr<Cylinder> c;
		
		switch (i) {
		case 0:
			c = new Cylinder(Vec3(0.5, 0, 0), AXIS_RADIUS, AXIS_LEN);
			c->setRotation(Quat(PI / 2, Vec3(0, 1, 0)));
			m->setDiffuse( Material::FRONT, Vec4(1, 0, 0, 1));
			break;
		case 1:
			c = new Cylinder(Vec3(0, 0.5, 0), AXIS_RADIUS, AXIS_LEN);
			c->setRotation(Quat(PI / 2, Vec3(1, 0, 0)));
			m->setDiffuse( Material::FRONT, Vec4(0, 1, 0, 1));
			break;
		case 2:
			c = new Cylinder(Vec3(0, 0, 0.5), AXIS_RADIUS, AXIS_LEN);
			m->setDiffuse( Material::FRONT, Vec4(0, 0, 1, 1));
			break;
		}
		g->addDrawable(new ShapeDrawable(c));
		g->getOrCreateStateSet()->setAttribute(m, StateAttribute::ON | StateAttribute::OVERRIDE);
		axes->addChild(g);
	}
	trans->addChild(axes);
}

void node::create_label() {
	ref_ptr<osgText::Text> txt = new osgText::Text;
	txt->setText(name);
	txt->setFont(FONT);
	txt->setAxisAlignment(osgText::Text::SCREEN);
	txt->setColor(Vec4(1.f, 0.f, 0.f, 1.f));
    txt->setCharacterSizeMode(osgText::Text::SCREEN_COORDS);
    txt->setCharacterSize(24);
    
	label = new Geode;
	ref_ptr<StateSet> ss = label->getOrCreateStateSet();
	ss->setMode(GL_DEPTH_TEST, StateAttribute::OFF);
	/*
	ss->setRenderingHint( StateSet::TRANSPARENT_BIN );
	ss->setRenderBinDetails(11, "RenderBin");
	*/
	label->addDrawable(txt);
	trans->addChild(label);
}

void node::add_child(node &n) {
	assert(group.valid());
	group->addChild(n.trans);
}

void node::remove_child(node &n) {
	group->removeChild(n.trans.get());
}

void node::set_vertices(const vector<Vec3> &verts) {
	ref_ptr<Geometry> g = new Geometry;
	ref_ptr<Vec3Array> v = new Vec3Array;
	copy(verts.begin(), verts.end(), back_inserter(*v));
	g->setVertexArray(v);
	
	if (verts.size() == 1) {
		g->addPrimitiveSet(new DrawArrays(GL_POINTS, 0, 1));
	} else if (verts.size() == 2) {
		g->addPrimitiveSet(new DrawArrays(GL_LINES, 0, 2));
	} else if (verts.size() == 3) {
		g->addPrimitiveSet(new DrawArrays(GL_TRIANGLES, 0, 3));
	} else {
		vector<vector<int> > facets;
		if (qhull(verts, facets) != 0) {
			cerr << "error executing qhull" << endl;
			exit(1);
		}
		ref_ptr<DrawElementsUInt> triangles = new DrawElementsUInt(GL_TRIANGLES);
		ref_ptr<DrawElementsUInt> quads = new DrawElementsUInt(GL_QUADS);
		for (int i = 0; i < facets.size(); ++i) {
			if (facets[i].size() == 3) {
				copy(facets[i].begin(), facets[i].end(), back_inserter(*triangles));
			} else if (facets[i].size() == 4) {
				copy(facets[i].begin(), facets[i].end(), back_inserter(*quads));
			} else {
				assert(false);
			}
		}
		
		if (!triangles->empty()) {
			g->addPrimitiveSet(triangles);
		}
		if (!quads->empty()) {
			g->addPrimitiveSet(quads);
		}
	}
	leaf->removeDrawable(0);
	leaf->addDrawable(g);
}

bool node::is_group() {
	return group.valid();
}

void node::toggle_axes() {
	if (axes) {
		axes->setNodeMask(~axes->getNodeMask());
	}
}

scene::scene() {
	node *w = new node("world", "");
	nodes["world"] = w;
	ref_ptr<PositionAttitudeTransform> r = w->trans;
	//r->getOrCreateStateSet()->setMode(GL_LIGHTING, StateAttribute::OFF);
}

bool parse_vec3(vector<string> &f, int &p, Vec3 &x) {
	if (p + 3 > f.size()) {
		p = f.size();
		return false;
	}
	for (int i = 0; i < 3; ++i) {
		char *end;
		x[i] = strtod(f[p + i].c_str(), &end);
		if (*end != '\0') {
			p += i;
			return false;
		}
	}
	p += 3;
	return true;
}

bool parse_verts(vector<string> &f, int &p, vector<Vec3> &verts) {
	if (p >= f.size() || f[p] != "v") {
		return true;
	}
	++p;
	while (p < f.size()) {
		Vec3 v;
		int old = p;
		if (!parse_vec3(f, p, v)) {
			if (p > old) {
				return false;
			}
			break;
		}
		verts.push_back(v);
	}
	return true;
}

bool parse_transforms(vector<string> &f, int &p, PositionAttitudeTransform &trans) {	
	Vec3 t;
	char type;
	
	while (p < f.size()) {
		if (f[p].size() != 1 || f[p].find_first_of("prs") != 0) {
			return true;
		}
		type = f[p++][0];
		if (!parse_vec3(f, p, t)) {
			return false;
		}
		switch (type) {
			case 'p':
				trans.setPosition(t);
				break;
			case 'r':
				trans.setAttitude(to_quaternion(t));
				break;
			case 's':
				trans.setScale(t);
				break;
			default:
				assert(false);
		}
	}
	return true;
}

// f[0] is node name, f[1] is parent name
int scene::parse_add(vector<string> &f) {
	if (f.size() < 2) {
		return f.size();
	}
	
	if (nodes.find(f[0]) != nodes.end()) {
		return 0;
	}
	if (nodes.find(f[1]) == nodes.end() || !nodes[f[1]]->is_group()) {
		return 1;
	}
	
	int p = 2;
	vector<Vec3> verts;
	if (!parse_verts(f, p, verts)) {
		return p;
	}

	node *n = new node(f[0], f[1], verts);
	if (!parse_transforms(f, p, *(n->trans))) {
		delete n;
		return p;
	}
	nodes[f[0]] = n;
	nodes[f[1]]->add_child(*n);
	return -1;
}

int scene::parse_change(vector<string> &f) {
	if (f.size() < 2) {
		return f.size();
	}
	
	if (nodes.find(f[0]) == nodes.end()) {
		return 0;
	}
	
	node *n = nodes[f[0]];

	int p = 1;
	vector<Vec3> verts;
	if (!parse_verts(f, p, verts)) {
		return p;
	}
	if (!verts.empty() && !n->is_group()) {
		n->set_vertices(verts);
	}
	
	if (!parse_transforms(f, p, *(n->trans))) {
		return p;
	}
	
	return -1;
}

int scene::parse_del(vector<string> &f) {
	if (f.size() != 1) {
		return f.size();
	}
	if (nodes.find(f[0]) == nodes.end()) {
		return 0;
	}

	node *n = nodes[f[0]];
	nodes[n->parent]->remove_child(*n);
	
	// remove n and all offspring from nodes map
	vector<string> offspring;
	offspring.push_back(f[0]);
	for (int i = 0; i < offspring.size(); ++i) {
		delete nodes[offspring[i]];
		nodes.erase(offspring[i]);
		// look for children
		map<string, node*>::iterator j;
		for (j = nodes.begin(); j != nodes.end(); ++j) {
			if (j->second->parent == offspring[i]) {
				offspring.push_back(j->first);
			}
		}
		
	}
	
	return -1;
}

void scene::update(const vector<string> &fields) {
	char cmd;
	int errfield;
	
	if (fields.size() == 0) {
		return;
	}
	if (fields[0].size() != 1 || fields[0].find_first_of("acd") != 0) {
		cerr << "known commands are a, c, or d" << endl;
		return;
	}
	
	cmd = fields[0][0];
	vector<string> rest;
	for (int i = 1; i < fields.size(); ++i) {
		rest.push_back(fields[i]);
	}
	
	switch(cmd) {
		case 'a':
			errfield = parse_add(rest);
			break;
		case 'c':
			errfield = parse_change(rest);
			break;
		case 'd':
			errfield = parse_del(rest);
			break;
		default:
			return;
	}
	
	if (errfield >= 0) {
		cerr << "error in field " << errfield + 1 << endl;
	}
}

Group* scene::get_root() {
	return nodes["world"]->trans.get();
}

void scene::toggle_axes() {
	node_table::iterator i;
	for (i = nodes.begin(); i != nodes.end(); ++i) {
		i->second->toggle_axes();
	}
}