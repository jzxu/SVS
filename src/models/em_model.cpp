#include <cstdlib>
#include <vector>
#include <sstream>
#include <fstream>
#include "model.h"
#include "em.h"
#include "filter_table.h"
#include "params.h"
#include "svs.h"
#include "scene.h"
#include "serialize.h"
#include "drawer.h"

using namespace std;

const int MAXITERS = 50;

void error_color(double error, double color[]) {
	double maxerror = 1e-3;
	color[0] = color[1] = color[2] = 0.0;
	if (is_nan(error)) {
		return;
	} else if (error > maxerror) {
		color[0] = 1.0;
	} else {
		color[0] = error / maxerror;
		color[1] = 1 - (error / maxerror);
	}
}

void draw_predictions(drawer *d, int mode, int nmodes, double pred, double real, const string &name, EM *em) {
	static double mode_colors[][3] = {
		{ 0.0, 0.0, 0.0 },
		{ 1.0, 0.0, 1.0 },
		{ 0.0, 1.0, 1.0 },
		{ 1.0, 1.0, 0.0 },
		{ 1.0, 0.5, 0.0 },
		{ 0.5, 0.0, 0.5 },
	};
	static int ncolors = sizeof(mode_colors) / sizeof(mode_colors[0]);

	const double stretch = 20.0;
	const int mode_text_x = 50;
	const int xmode_text_y = 460;
	const int zmode_text_y = 410;
	const int font_size = 12;
	
	static double vx = NAN, vz = NAN, vxerror, vzerror;
	static int xmode = 0, zmode = 0, xnmodes = 0, znmodes = 0;
	static bool init = true;
	
	stringstream ss;
	
	if (init) {
		ss << "layer 1 l 0 n 0\n"
		   << "layer 2 l 0 n 0 f 1\n"
		   << "* +b1:vx_mode_header t b1:vx l 2 c 1 1 1 p " << mode_text_x << " " << xmode_text_y << " 0\n"
		   << "* +b1:vz_mode_header t b1:vz l 2 c 1 1 1 p " << mode_text_x << " " << zmode_text_y << " 0\n"
		   << "* +vx_pred_line l 1 w 2\n"
		   << "* +vz_pred_line l 1 w 2\n"
		   << "* +pred_line    l 1 w 2\n";
		
		init = false;
	}

	int old_nmodes = 0, old_mode = 0, y = 0;
	if (name == "b1:vx") {
		old_nmodes = xnmodes;
		old_mode = xmode;
		xnmodes = nmodes;
		xmode = mode;
		y = xmode_text_y;
		vx = pred * stretch;
		vxerror = abs(real - pred);
	} else if (name == "b1:vz") {
		old_nmodes = znmodes;
		old_mode = zmode;
		znmodes = nmodes;
		zmode = mode;
		y = zmode_text_y;
		vz = pred * stretch;
		vzerror = abs(real - pred);
	}
	
	/* draw mode text */
	for (int i = 0; i < nmodes; ++i) {
		string t;
		em->get_mode_function_string(i, t);
		y -= font_size;
		ss << "* +" << name << "_mode_" << i
		   << " t \"" << t << "\""
		   << " c 1 1 1 p " << mode_text_x + font_size * 4 << " " << y << " 0 l 2\n";
	}
	
	for (int i = nmodes; i < old_nmodes; ++i) {
		ss << "* -" << name << "_mode_" << i << "\n";
	}
	
	/* set color for selected mode */
	ss << "* " << name << "_mode_" << old_mode << " c 1 1 1\n";
	ss << "* " << name << "_mode_" << mode << " c 1 1 0\n";
	
	double cx[3], cz[3], cp[3];
	
	error_color(vxerror, cx);
	error_color(vzerror, cz);
	error_color(vxerror + vzerror / 2, cp);
	
	bool vx_valid = !is_nan(vx);
	bool vz_valid = !is_nan(vz);
	
	ss << "* vx_pred_line v 0 0 0 " << (vx_valid ? vx : 1000.0) << " 0 0"
	   << " i 0 1 c " << cx[0] << " " << cx[1] << " " << cx[2] << "\n";
	ss << "* vz_pred_line v 0 0 0 0 0 " << (vz_valid ? vz : 1000.0)
	   << " i 0 1 c " << cz[0] << " " << cz[1] << " " << cz[2] << "\n";
	
	if (vx_valid && vz_valid) {
		ss << "* pred_line v 0 0 0 " << vx << " 0 " << vz << " i 0 1";
		ss << " c " << cp[0] << " " << cp[1] << " " << cp[2] << "\n";
	} else {
		ss << "* pred_line v 0 0 0 0 0 0 i 0 1\n";
	}
	d->send(ss.str());
}

class EM_model : public model {
public:
	EM_model(svs *owner, const string &name)
	: model(name, "em"), em(get_data(), owner->get_loggers())
	{
		draw = owner->get_drawer();
	}

	bool predict_sub(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, bool test, rvec &y, map<string, rvec> &info)  {
		int mode;
		double real_y;
		rvec all_preds, mode_info, vote_trace;
		bool success;

		real_y = y(0);
		mode = 0;
		mode = em.predict(target, sig, rels, x, mode, y(0), vote_trace);
		if (test) {
			em.all_predictions(target, sig, rels, x, all_preds);
			mode_info.resize(all_preds.size());
			mode_info(0) = mode;
			for (int i = 1, iend = all_preds.size(); i < iend; ++i) {
				mode_info(i) = fabs(real_y - all_preds(i));
			}
			info["mode"] = mode_info;
			info["votes"] = vote_trace;
		}
		if (mode > 0) {
			draw_predictions(draw, mode, em.num_modes(), y(0), real_y, get_name(), &em);
		}
		return mode > 0;
	}
	
	int get_input_size() const {
		return -1;
	}
	
	int get_output_size() const {
		return 1;
	}

	void update() {
		assert(get_data().get_last_inst().y.size() == 1);
		em.add_data(get_data().size() - 1);
		em.run(MAXITERS);
	}
	
	void proxy_get_children(map<string, cliproxy*> &c) {
		model::proxy_get_children(c);
		c["em"] = &em;
	}
	
private:
	EM em;
	drawer *draw;
	
	void serialize_sub(ostream &os) const {
		em.serialize(os);
	}
	
	void unserialize_sub(istream &is) {
		em.unserialize(is);
	}
};

model *make_em_model(const string &name, svs *owner) {
	return new EM_model(owner, name);
}
