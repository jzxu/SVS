#include <iostream>
#include <cstdlib>
#include "drawer.h"
#include "common.h"

#ifdef _WIN32
#include <windows.h>
#endif

using namespace std;

void write_shape_string(const sgnode *n, ostream &os) {
	const convex_node *cn = dynamic_cast<const convex_node *>(n);
	if (cn) {
		const ptlist &pts = cn->get_local_points();
		os << "v ";
		for (int i = 0; i < pts.size(); ++i) {
			os << pts[i] << " ";
		}
		return;
	}
}

drawer::drawer() {
	string path = get_option("display");
	if (path.empty()) {
#ifndef _WIN32
		path = "/tmp/viewer";
#else
		char* temp_string = new char[MAX_PATH+1];
		GetTempPath(MAX_PATH+1, temp_string);
		path = temp_string;
		path += "viewer";
		delete temp_string;
#endif
	}
	sock.connect(path);
	if (sock.connected()) {
		sock.send("clear\n");
	}
}

drawer::~drawer() {
	sock.disconnect();
}

void drawer::add(const string &scn, const sgnode *n) {
	if (!sock.connected()) {
		return;
	}
	stringstream ss;
	string shape_str;
	n->get_shape_sgel(shape_str);
	
	ss << scn << " a " << n->get_name() << " " << n->get_parent()->get_name() << " " << shape_str
	   << " p " << n->get_trans('p')
	   << " r " << n->get_trans('r')
	   << " s " << n->get_trans('s') << endl;
	
	sock.send(ss.str());
}

void drawer::del(const string &scn, const sgnode *n) {
	if (!sock.connected()) {
		return;
	}
	
	stringstream ss;
	ss << scn << " d " << n->get_name() << endl;
	sock.send(ss.str());
}

void drawer::change(const string &scn, const sgnode *n, int props) {
	if (!sock.connected()) {
		return;
	}
	
	stringstream ss;
	ss << scn << " c " << n->get_name() << " ";
	if (props & SHAPE) {
		write_shape_string(n, ss);
	}
	if (props & POS) {
		ss << " p " << n->get_trans('p');
	}
	if (props & ROT) {
		ss << " r " << n->get_trans('r');
	}
	if (props & SCALE) {
		ss << " s " << n->get_trans('s');
	}
	ss << endl;
	sock.send(ss.str());
}
