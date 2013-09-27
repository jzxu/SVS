#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <limits>
#include <iomanip>
#include <memory>
#include "linear.h"
#include "em.h"
#include "common.h"
#include "params.h"
#include "mat.h"
#include "serialize.h"
#include "drawer.h"
#include "logger.h"

#define DBGCOUNT(n) { static int count = 0; fprintf(stderr, "%s %d\n", n, count++); }
//#define DBGCOUNT(n)

using namespace std;
using namespace Eigen;

/* Box-Muller method */
double randgauss(double mean, double std) {
	double x1, x2, w;
	do {
		x1 = 2.0 * ((double) rand()) / RAND_MAX - 1.0;
		x2 = 2.0 * ((double) rand()) / RAND_MAX - 1.0;
		w = x1 * x1 + x2 * x2;
	} while (w >= 1.0);
	
	w = sqrt((-2.0 * log(w)) / w);
	return mean + std * (x1 * w);
}

void kernel1(const cvec &d, cvec &w) {
	w.resize(d.size());
	for (int i = 0; i < d.size(); ++i) {
		w(i) = exp(-d(i));
	}
}

void kernel2(const cvec &d, cvec &w, double p) {
	const double maxw = 1.0e9;
	w.resize(d.size());
	for (int i = 0; i < d.size(); ++i) {
		if (d(i) == 0.0) {
			w(i) = maxw;
		} else {
			w(i) = min(maxw, pow(d(i), p));
		}
	}
}

void predict(const mat &C, const rvec &intercepts, const rvec &x, rvec &y) {
	y = (x * C) + intercepts;
}

/*
 Upon return, X and Y will contain the training data, Xtest and Ytest the
 testing data.
*/
void split_data(
	const_mat_view X,
	const_mat_view Y,
	const vector<int> &use,
	int ntest,
	mat &Xtrain, mat &Xtest, 
	mat &Ytrain, mat &Ytest)
{
	int ntrain = use.size() - ntest;
	vector<int> test;
	sample(ntest, 0, use.size(), test);
	sort(test.begin(), test.end());
	
	int train_end = 0, test_end = 0, i = 0;
	for (int j = 0; j < use.size(); ++j) {
		if (i < test.size() && j == test[i]) {
			Xtest.row(test_end) = X.row(use[j]);
			Ytest.row(test_end) = Y.row(use[j]);
			++test_end;
			++i;
		} else {
			Xtrain.row(train_end) = X.row(use[j]);
			Ytrain.row(train_end) = Y.row(use[j]);
			++train_end;
		}
	}
	assert(test_end == ntest && train_end == ntrain);
}

template <typename T>
struct indirect_cmp {
	indirect_cmp(const vector<T> &v) : v(v) {}
	
	bool operator()(int a, int b) const {
		return v[a] < v[b];
	}
	
	const vector<T> &v;
};

template <typename T>
void get_ordering(const vector<T> &v, vector<int> &order) {
	order.resize(v.size());
	for (int i = 0, iend = order.size(); i < iend; ++i) {
		order[i] = i;
	}
	sort(order.begin(), order.end(), indirect_cmp<T>(v));
}

void erase_inds(vector<int> &v, const vector<int> &inds) {
	int i = 0, j = 0;
	for (int k = 0; k < v.size(); ++k) {
		if (i < inds.size() && k == inds[i]) {
			++i;
		} else {
			if (j < k) {
				v[j] = v[k];
			}
			++j;
		}
	}
	assert(i == inds.size() && j == v.size() - inds.size());
	v.resize(j);
}

/*
 Check if any examples (rows) in X and Y are causing spurious regressors to be
 included in the linear function that describes the data, as defined by coefs.
 This is done in a way similar to cross-validation: some portion of the data is
 held out, a linear regression is performed, and the coefficients obtained are
 compared to the coefs parameter. If any of the coefficients differ in sign,
 that is, negative, 0, positive, then this indicates the presence of spurious
 regressors.

 The alternative to checking sign agreement is to check the Euclidean distance
 between the coefficients. But this would have to be compared against an
 arbitrary threshold, which would be yet another free parameter in the system.

 Normal cross-validation doesn't work well in this case, because we expect that
 only a few (most likely just one) data points are necessitating the spurious
 regressors. Therefore, the cross-validated error would still be very low,
 because only the error on a few data points would be high. Maybe normal CV
 will work if we consider the max error rather than average error.
*/
bool find_spurious_regressors(const mat &X, const mat &Y, double noise_var, const mat &coefs) {
	int holdout_size, holdin_size, holdout_start;
	mat Xin, Yin;
	mat coefs_in;
	rvec inter_in;
	cvec dummy;
	int num_nonzero = 0;

	for (int i = 0, iend = coefs.rows(); i < iend; ++i) {
		if (coefs(i, 0) != 0) {
			++num_nonzero;
		}
	}

	holdout_size = X.rows() * .3;
	if (holdout_size < 2) {
		holdout_size = 2;
		assert(holdout_size < X.rows());
	}
	holdin_size = X.rows() - holdout_size;
	holdout_start = 0;
	while (holdout_start < X.rows()) {
		/*
		 If X.rows() doesn't divide evenly by holdout_size, then the last iteration
		 may result in fewer than holdin_size hold-ins. Manually adjust for this.
		*/
		if (holdout_start + holdout_size > X.rows()) {
			holdout_start = X.rows() - holdout_size;
		}
		int top_size = holdout_start;
		int bottom_size = holdin_size - top_size;
		Xin.resize(holdin_size, X.cols());
		Yin.resize(holdin_size, Y.cols());
		Xin.topRows(top_size) = X.topRows(top_size);
		Yin.topRows(top_size) = Y.topRows(top_size);
		Xin.bottomRows(bottom_size) = X.bottomRows(bottom_size);
		Yin.bottomRows(bottom_size) = Y.bottomRows(bottom_size);

		if (!linreg(FORWARD, Xin, Yin, dummy, noise_var, false, coefs_in, inter_in)) {
			FATAL("linear regression failed");
		}
		
		assert(coefs_in.rows() == coefs.rows());
		int n = 0;
		for (int i = 0, iend = coefs_in.rows(); i < iend; ++i) {
			if (coefs_in(i,0) != 0.0) {
				++n;
			}
		}
		if (n > num_nonzero) {
			return true;
		}
		holdout_start += holdout_size;
	}
	return false;
}

int ransac_iters(int ninliers, int mss_size, int ndata) {
	double q = pow(ninliers / static_cast<double>(ndata), mss_size);
	double lq = log(1 - q);
	if (lq == 0) {
		return RANSAC_MAX_ITERS;
	}
	double i1 = RANSAC_LOG_ALARM_RATE / lq + 1.0;  // don't cast here to avoid overflow
	assert(i1 >= 0.0);
	if (i1 > RANSAC_MAX_ITERS) {
		return RANSAC_MAX_ITERS;
	}
	return static_cast<int>(floor(i1));
}

/*
 Gets a random sample that tries to have at least two different values for each
 dimension in diff_cols.

 Each element of diff_cols that does get two distinct values will be set to -1
 upon return
*/
int ransac_sample(const_mat_view X, const_mat_view Y, const vector<int> &diff_cols, int n, int offset, vector<int> &out) {
	int nrows = X.rows(), ncols = X.cols(), nout = 0;
	vector<bool> used(nrows);
	vector<int> candidates;

	for (int i = 0, iend = diff_cols.size(); i < iend && nout < n; ++i) {
		if (diff_cols[i] < 0)
			continue;

		int c = diff_cols[i];
		double x = (nout == 0 ? NAN : X(out.back() - offset, c));
		candidates.clear();
		for (int j = 0; j < nrows; ++j) {
			if (!used[j] && X(j, c) != x) {
				candidates.push_back(j);
			}
		}
		if (!candidates.empty()) {
			int r = candidates[rand() % candidates.size()];
			out.push_back(offset + r);
			used[r] = true;
			++nout;
		}
	}
	if (nout < n) {
		candidates.clear();
		for (int i = 0; i < nrows; ++i) {
			if (!used[i]) {
				candidates.push_back(offset + i);
			}
		}
		int n1 = min(n - nout, static_cast<int>(candidates.size()));
		sample(n1, candidates, out);
		nout += n1;
	}
	return nout;
}

/*
 Added the check_spurious parameter to allow only checking for spurious
 regressors when looking for linear relationships in noise data, but not during
 unification. This is because during unification, it's often the case that the
 existing mode will have a large number of examples, while the new mode will
 have very few members. In this case, extra legitimate coefficients introduced
 during the unification will likely be considered spurious, and discarded. We
 want to avoid doing this.
*/
void ransac(const mat &X, const mat &Y, double noise_var, int size_thresh, int split, bool check_spurious,
            vector<int> &subset, mat &coefs, rvec &intercept)
{
	vector<int> mss, fit_set, nonuniform_cols;
	mat Xmss, Ymss, Yp, C;
	cvec dummy, error;
	rvec inter;
	
	int ndata = X.rows(), ncols = X.cols();
	int mss_size;
	int iters;
	double error_thresh = sqrt(noise_var) * NUM_STDEVS_THRESH;
	
	/* Find non-uniform columns */
	for (int i = 0, iend = X.cols(); i < iend; ++i) {
		if (!uniform(X.col(i))) {
			nonuniform_cols.push_back(i);
		}
	}

	mss_size = nonuniform_cols.size();
	if (ndata <= mss_size) {
		split = 0;
		iters = 1;
	} else {
		iters = ransac_iters(mss_size, mss_size, ndata);
	}
	
	mss.reserve(mss_size);
	fit_set.reserve(ndata);
	subset.clear();
	
	for (int i = 0; i < iters; ++i) {
		mss.clear();
		if (split == 0) {
			ransac_sample(X, Y, nonuniform_cols, mss_size, 0, mss);
		} else {
			// try to sample from the two sides of the split as evenly as possible
			int n1 = split, n2 = ndata - split;
			int m1 = mss_size / 2;
			int m2 = mss_size - m1;

			const_mat_view X1(X.block(0, 0, n1, ncols)), X2(X.block(split, 0, n2, ncols));
			const_mat_view Y1(Y.block(0, 0, n1, 1)), Y2(Y.block(split, 0, n2, 1));
			if (n1 < m1) {
				for (int j = 0; j < n1; ++j) {
					mss.push_back(j);
				}
				ransac_sample(X2, Y2, nonuniform_cols, mss_size - n1, split, mss);
			} else if (n2 < m2) {
				for (int j = split; j < ndata; ++j) {
					mss.push_back(j);
				}
				ransac_sample(X1, Y1, nonuniform_cols, mss_size - n2, 0, mss);
			} else {
				ransac_sample(X1, Y1, nonuniform_cols, m1, 0, mss);
				ransac_sample(X2, Y2, nonuniform_cols, m2, split, mss);
			}
		}
		
		assert(mss.size() == mss_size);
		
		pick_rows(X, mss, Xmss);
		pick_rows(Y, mss, Ymss);
		static int dbgcount = 0;
		if (!linreg(FORWARD, Xmss, Ymss, dummy, noise_var, false, C, inter)) {
			FATAL("linear regression failed");
		}
		Yp = (X * C).rowwise() + inter;
		error = (Y - Yp).cwiseAbs().rowwise().sum();
		
		fit_set.clear();
		for (int j = 0; j < ndata; ++j) {
			if (error(j) <= error_thresh) {
				fit_set.push_back(j);
			}
		}
		if (fit_set.size() > 2 && fit_set.size() > subset.size()) {
			mat Xsub, Ysub;
			pick_rows(X, fit_set, Xsub);
			pick_rows(Y, fit_set, Ysub);
			if (!check_spurious || !find_spurious_regressors(Xsub, Ysub, noise_var, C)) {
				subset = fit_set;
				coefs = C;
				intercept = inter;
				if (subset.size() >= size_thresh) {
					return;
				}
				iters = ransac_iters(subset.size(), mss_size, ndata);
			}
		}
	}
}

template <typename T>
void remove_from_vector(const vector<int> &inds, vector <T> &v) {
	int i = 0, j = 0;
	for (int k = 0; k < v.size(); ++k) {
		if (i < inds.size() && k == inds[i]) {
			++i;
		} else {
			if (k > j) {
				v[j] = v[k];
			}
			j++;
		}
	}
	assert(v.size() - inds.size() == j);
	v.resize(j);
}

EM::EM(const model_train_data &data, logger_set *loggers)
: data(data), loggers(loggers), use_unify(true),
  learn_new_modes(true), check_after(NEW_MODE_THRESH), clsfr(data, loggers)
{
	noise_var = 1e-15;
	add_mode(false); // noise mode
}

EM::EM(const model_train_data &data, logger_set *loggers,
       bool use_unify, bool learn_new_modes)
: data(data), loggers(loggers), use_unify(use_unify),
  learn_new_modes(learn_new_modes), check_after(NEW_MODE_THRESH),
  clsfr(data, loggers)
{
	noise_var = 1e-15;
	add_mode(false); // noise mode
}

EM::~EM() {
	clear_and_dealloc(insts);
	clear_and_dealloc(modes);
	clear_and_dealloc(sigs);
}


void EM::add_data(int t) {
	function_timer tm(timers.get_or_add("learn"));
	
	const model_train_inst &d = data.get_inst(t);
	inst_info *inst = new inst_info;
	sig_info *s = NULL;
	if (has(sigs, d.sig)) {
		s = sigs[d.sig];
	} else {
		s = new sig_info;
		sigs[d.sig] = s;
	}
	s->members.push_back(t);
	s->noise.insert(t);
	
	inst->mode = 0;
	inst->minfo.resize(modes.size());
	insts.push_back(inst);
	
	modes[0]->add_example(t, vector<int>(), noise_var);
	clsfr.update_class(t, -1, 0);
}

void EM::estep() {
	
	/*
	 For data i and mode j, if:
	 
	  * P(i, j) increases and j was not the MAP mode, or
	  * P(i, j) decreases and j was the MAP mode
	 
	 then we mark i as a point we have to recalculate the MAP mode for.
	*/
	for (int i = 0, iend = insts.size(); i < iend; ++i) {
		const model_train_inst &d = data.get_inst(i);
		inst_info &inst = *insts[i];
		bool best_stale = false;
		for (int j = 1, jend = modes.size(); j < jend; ++j) {
			inst_info::mode_info &m = inst.minfo[j];
			if (!m.error_stale && !modes[j]->is_new_fit()) {
				continue;
			}
			/*
			 Set best_stale to true if
			   - Error(m, j) increases, and m is the current best mode, or
			   - Error(m, j) decreases, and m is not the current best mode
			*/
			double new_error;
			new_error = modes[j]->calc_error(d.target, *d.sig, d.x, d.y(0), noise_var, m.role_map);
			assert(is_inf(new_error) || m.role_map.size() == modes[j]->num_roles());
			if ((inst.mode == j && new_error > m.error) || (inst.mode != j && new_error < m.error)) {
				best_stale = true;
			}
			m.error = new_error;
			m.error_stale = false;
		}
		if (best_stale) {
			int prev = inst.mode, best = 0;
			for (int j = 1, jend = modes.size(); j < jend; ++j) {
				double j_error = inst.minfo[j].error;
				double best_error = inst.minfo[best].error;
				/*
				 These conditions look awkward, but have justification. If I tested the >
				 condition before the approx_equal condition, the test would succeed even
				 if the error of j was only slightly better than the best error
				*/
				if (approx_equal(j_error, best_error, 0.001 * min(j_error, best_error))) {
					if (modes[j]->get_num_nonzero_coefs() < modes[best]->get_num_nonzero_coefs()) {
						best = j;
					}
				} else if (j_error < best_error) {
					best = j;
				}
			}
			if (inst.minfo[best].error > sqrt(noise_var) * NUM_STDEVS_THRESH) {
				best = 0;
			}
			if (best != prev) {
				inst.mode = best;
				modes[prev]->del_example(i);
				if (prev == 0) {
					sigs[d.sig]->noise.erase(i);
				}
				assert(modes[best]->num_roles() == inst.minfo[best].role_map.size());
				modes[best]->add_example(i, inst.minfo[best].role_map, noise_var);
				if (best == 0) {
					sigs[d.sig]->noise.insert(i);
				}
				clsfr.update_class(i, prev, best);
			}
		}
	}
	
	for (int i = 1; i < modes.size(); ++i) {
		modes[i]->reset_new_fit();
	}
}

bool EM::mstep() {
	function_timer t(timers.get_or_add("m-step"));
	
	bool changed = false;
	for (int i = 1, iend = modes.size(); i < iend; ++i) {
		changed = changed || modes[i]->update_fits(noise_var);
	}
	return changed;
}

void EM::fill_xy(const interval_set &rows, mat &X, mat &Y) const {
	if (rows.empty()) {
		X.resize(0, 0);
		Y.resize(0, 0);
		return;
	}

	X.resize(rows.size(), data.get_inst(rows.ith(0)).x.size());
	Y.resize(rows.size(), 1);

	interval_set::const_iterator i, iend;
	int j;
	for (i = rows.begin(), iend = rows.end(), j = 0; i != iend; ++i, ++j) {
		X.row(j) = data.get_inst(*i).x;
		Y.row(j) = data.get_inst(*i).y;
	}
}

em_mode *EM::add_mode(bool manual) {
	em_mode *new_mode = new em_mode(modes.size() == 0, manual, data, loggers);
	modes.push_back(new_mode);
	for (int i = 0, iend = insts.size(); i < iend; ++i) {
		grow_vec(insts[i]->minfo);
	}
	clsfr.add_class();
	return new_mode;
}

void EM::unify(const em_mode *m, const vector<int> &new_ex, int sig, int target, unify_result &result) const {
	result.success = false;
	result.num_ex = 0;
	result.num_uncovered = m->size();
	result.num_new_covered = 0;
	result.num_coefs = 0;

	if (!m->unifiable(sig, target)) {
		return;
	}
	const interval_set &curr_ex = m->get_members();
	int nrows = curr_ex.size() + new_ex.size();
	int ncols = data.get_inst(curr_ex.ith(0)).x.size();
	int thresh = curr_ex.size() + .9 * new_ex.size();

	mat X(nrows, ncols), Y(nrows, 1);

	interval_set::const_iterator i, iend;
	int j = 0;
	for (i = curr_ex.begin(), iend = curr_ex.end(); i != iend; ++i, ++j) {
		X.row(j) = data.get_inst(*i).x;
		Y.row(j) = data.get_inst(*i).y;
	}
	for (int k = 0, kend = new_ex.size(); k < kend; ++k, ++j) {
		X.row(j) = data.get_inst(new_ex[k]).x;
		Y.row(j) = data.get_inst(new_ex[k]).y;
	}

	vector<int> unified_ex;
	mat coefs;
	rvec intercept;
	ransac(X, Y, noise_var, thresh, curr_ex.size(), false, unified_ex, coefs, intercept);
	if (unified_ex.size() < thresh) {
		return;
	}
	
	result.success = true;
	result.coefs = coefs.col(0);
	result.intercept = intercept(0);
	result.num_ex = unified_ex.size();
	result.num_uncovered = curr_ex.size();
	result.num_new_covered = 0;
	for (int k = 0, kend = unified_ex.size(); k < kend; ++k) {
		if (unified_ex[k] < curr_ex.size()) {
			--result.num_uncovered;
		} else {
			++result.num_new_covered;
		}
	}
	result.num_coefs = 0;
	for (int k = 0, kend = result.coefs.size(); k < kend; ++k) {
		if (result.coefs(k) != 0.0) {
			++result.num_coefs;
		}
	}
}

bool EM::unify_or_add_mode() {
	function_timer t(timers.get_or_add("new"));

	assert(check_after >= NEW_MODE_THRESH);
	if (!learn_new_modes || modes[0]->size() < check_after) {
		return false;
	}
	
	vector<int> largest;
	mat coefs(0,1);
	rvec inter;
	modes[0]->largest_const_subset(largest);
	int potential = largest.size();
	inter = data.get_inst(largest[0]).y; // constant model
	if (largest.size() < NEW_MODE_THRESH) {
		mat X, Y;
		sig_table::const_iterator i, iend;
		for (i = sigs.begin(), iend = sigs.end(); i != iend; ++i) {
			const interval_set &ns = i->second->noise;
			if (ns.size() < check_after) {
				if (ns.size() > potential) {
					potential = ns.size();
				}
				continue;
			}
			interval_set inds(ns);
			vector<int> subset;
			fill_xy(inds, X, Y);
			ransac(X, Y, noise_var, NEW_MODE_THRESH, 0, true, subset, coefs, inter);
			if (subset.size() > potential) {
				potential = subset.size();
			}
			if (subset.size() > largest.size()) {
				largest.clear();
				for (int i = 0; i < subset.size(); ++i) {
					largest.push_back(inds.ith(subset[i]));
				}
				if (largest.size() >= NEW_MODE_THRESH) {
					break;
				}
			}
		}
	}
	
	loggers->get(LOG_EM) << "unify_or_add_mode: found " << largest.size() << " colinear examples in noise." << endl;

	if (largest.size() < NEW_MODE_THRESH) {
		check_after = modes[0]->size() + NEW_MODE_THRESH;
		return false;
	}
	
	/*
	 From here I know the noise data is going to either become a new mode or unify
	 with an existing mode, so reset check_after assuming the current noise is
	 gone.
	*/
	check_after = NEW_MODE_THRESH;
	
	int seed_sig = data.get_inst(largest[0]).sig_index;
	int seed_target = data.get_inst(largest[0]).target;

	if (use_unify) {
		/*
		 Try to add noise data to each current model and refit. If the resulting
		 model is just as accurate as the original, then just add the noise to that
		 model instead of creating a new one.
		*/
		unify_result best_result;
		int best_mode = 0;
		for (int i = 1, iend = modes.size(); i < iend; ++i) {
			unify_result r;
			unify(modes[i], largest, seed_sig, seed_target, r);
			if (r.success) {
				loggers->get(LOG_EM) << "Successfully unified with mode " << i << endl;
				if (best_mode == 0 || r.num_coefs < best_result.num_coefs) {
					best_mode = i;
					best_result = r;
				}
			}
		}
		if (best_mode > 0) {
			loggers->get(LOG_EM) << "Unifying with mode " << best_mode << endl;
			modes[best_mode]->set_params(*data.get_sig(seed_sig), seed_target, best_result.coefs, best_result.intercept);
			return true;
		}
		loggers->get(LOG_EM) << "Failed to unify with any modes" << endl;
	}
	
	em_mode *new_mode = add_mode(false);
	const model_train_inst &d0 = data.get_inst(largest[0]);
	new_mode->set_params(*d0.sig, d0.target, coefs.col(0), inter(0));

	loggers->get(LOG_EM) << "Adding new mode " << modes.size() - 1 << endl << "coefs =";
	for (int i = 0; i < coefs.rows(); ++i) {
		loggers->get(LOG_EM) << " " << coefs(i, 0);
	}
	loggers->get(LOG_EM) << endl;
	return true;
}

/*
 If parameter mode > 0, then use that mode for prediction. Otherwise, determine
 mode using classifier. Return the mode used, or 0 if prediction failed.
*/
int EM::predict(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, int mode, double &y, rvec &vote_trace) const {
	vector<int> role_map;
	if (mode > 0) {
		if (!modes[mode]->map_roles(target, sig, rels, role_map)) {
			mode = 0;
		}
	} else {
		if (insts.empty()) {
			mode = 0;
		} else {
			mode = classify(target, sig, rels, x, role_map, vote_trace);
		}
	}
	if (mode > 0) {
		y = modes[mode]->predict(sig, x, role_map);
	} else {
		y = NAN;
	}
	return mode;
}

/*
 Remove all modes that cover fewer than NEW_MODE_THRESH data points.
*/
bool EM::remove_modes() {
	if (modes.size() == 1) {
		return false;
	}
	
	/*
	 i is the first free model index. If model j should be kept, all
	 information pertaining to model j will be copied to row/element i
	 in the respective matrix/vector, and i will be incremented. Most
	 efficient way I can think of to remove elements from the middle
	 of vectors. index_map associates old j's to new i's.
	*/
	vector<int> index_map(modes.size()), removed;
	int i = 1;  // start with 1, noise mode (0) should never be removed
	for (int j = 1, jend = modes.size(); j < jend; ++j) {
		if (modes[j]->size() >= NEW_MODE_THRESH || modes[j]->is_manual()) {
			index_map[j] = i;
			if (j > i) {
				modes[i] = modes[j];
			}
			i++;
		} else {
			loggers->get(LOG_EM) << "Mode " << j << " only has " << modes[j]->size() << " examples, removing" << endl;
			index_map[j] = 0;
			delete modes[j];
			removed.push_back(j);
		}
	}
	if (removed.empty()) {
		return false;
	}
	assert(i == modes.size() - removed.size());
	modes.resize(i);
	for (int j = 0, jend = insts.size(); j < jend; ++j) {
		if (insts[j]->mode >= 0) {
			int old_mode = insts[j]->mode;
			insts[j]->mode = index_map[old_mode];
			if (old_mode != 0 && insts[j]->mode == 0) {
				clsfr.update_class(j, old_mode, 0);
			}
		}
		remove_from_vector(removed, insts[j]->minfo);
	}
	clsfr.del_classes(removed);
	return true;
}

bool EM::run(int maxiters) {
	for (int i = 0; i < maxiters; ++i) {
		estep();
		bool changed = mstep();
		changed |= remove_modes();
		changed |= unify_or_add_mode();
		if (!changed) {
			// reached quiescence
			return true;
		}
	}
	loggers->get(LOG_EM) << "Reached max iterations without quiescence" << endl;
	return false;
}

/*
 Returns the prediction generated by each mode in preds.
*/
void EM::all_predictions(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, rvec &preds) const {
	vector<int> assign;
	rvec votes;

	preds.resize(modes.size());
	for (int i = 1, iend = modes.size(); i < iend; ++i) {
		predict(target, sig, rels, x, i, preds(i), votes);
	}
}

void EM::proxy_use_sub(const vector<string> &args, ostream &os) {
	print_modes(os);
}

void EM::proxy_get_children(map<string, cliproxy*> &c) {
	proxy_group *mode_group = new proxy_group;
	
	for (int i = 0, iend = modes.size(); i < iend; ++i) {
		mode_group->add(tostring(i), modes[i]);
	}
	
	c["mode"] =        mode_group;
	c["classifier"] =  &clsfr;
	c["timers"] =      &timers;
	c["unify_modes"] = new bool_proxy(&use_unify, "Try to unify new modes with old ones.");
	c["learn_modes"] = new bool_proxy(&learn_new_modes, "Learn new modes.");
	c["noise_var"] =   new float_proxy(&noise_var, "Expected variance of environment noise.");
	
	c["error_table"] = new memfunc_proxy<EM>(this, &EM::cli_error_table);
	c["add_mode"] =    new memfunc_proxy<EM>(this, &EM::cli_add_mode);
	c["update_classifiers"] = new memfunc_proxy<EM>(this, &EM::cli_update_classifiers);
}

void EM::cli_error_table(ostream &os) const {
	print_error_table(os);
}

/*
 The format will be [coef] [dim] [coef] [dim] ... [intercept]
*/
void EM::cli_add_mode(const vector<string> &args, ostream &os) {
	if (insts.empty()) {
		os << "need at least one training example to get the signature from" << endl;
		return;
	}
	
	const model_train_inst &inst = data.get_last_inst();
	rvec coefs(inst.sig->dim());
	double intercept;
	coefs.setConstant(0.0);
	
	for (int i = 0, iend = args.size(); i < iend; i += 2) {
		double c;
		if (!parse_double(args[i], c)) {
			os << "expecting a number, got " << args[i] << endl;
			return;
		}
		
		if (i + 1 >= args.size()) {
			intercept = c;
			break;
		}
		
		vector<string> parts;
		split(args[i+1], ":", parts);
		if (parts.size() != 2) {
			os << "expecting object:property, got " << args[i+1] << endl;
			return;
		}
		
		int obj_ind, prop_ind;
		if (!inst.sig->get_dim(parts[0], parts[1], obj_ind, prop_ind)) {
			os << args[i+1] << " not found" << endl;
			return;
		}
		assert(prop_ind >= 0 && prop_ind < coefs.size());
		coefs(prop_ind) = c;
	}
	
	em_mode *new_mode = add_mode(true);
	new_mode->set_params(*inst.sig, inst.target, coefs, intercept);
}

void EM::cli_update_classifiers(ostream &os) {
	clsfr.update();
	for (int i = 1, iend = modes.size(); i < iend; ++i) {
		modes[i]->update_role_classifiers();
	}
}

void EM::serialize(ostream &os) const {
	serializer sr(os);
	sr << insts << clsfr << nc_type << modes.size() << '\n';
	vector<const scene_sig*> s = data.get_sigs();
	for (int i = 0, iend = s.size(); i < iend; ++i) {
		sr << *map_get(sigs, s[i]) << '\n';
	}
	for (int i = 0, iend = modes.size(); i < iend; ++i) {
		sr << *modes[i] << '\n';
	}
}

void EM::unserialize(istream &is) {
	unserializer unsr(is);
	int nmodes;
	
	clear_and_dealloc(insts);
	unsr >> insts >> clsfr >> nc_type >> nmodes;
	assert(insts.size() == data.size());
	
	clear_and_dealloc(sigs);
	vector<const scene_sig*> s = data.get_sigs();
	for (int i = 0, iend = s.size(); i < iend; ++i) {
		sig_info *si = new sig_info;
		unsr >> *si;
		sigs[s[i]] = si;
	}
	
	clear_and_dealloc(modes);
	for (int i = 0, iend = nmodes; i < iend; ++i) {
		em_mode *m = new em_mode(i == 0, false, data, loggers);
		m->unserialize(is);
		modes.push_back(m);
	}
}

void EM::print_error_table(ostream &os) const {
	table_printer t;
	for (int i = 0, iend = insts.size(); i < iend; ++i) {
		inst_info *inst = insts[i];
		t.add_row() << i << inst->mode;
		for (int j = 1, jend = inst->minfo.size(); j < jend; ++j) {
			t << inst->minfo[j].error;
		}
	}
	t.print(os);
}

void EM::print_modes(ostream &os) const {
	table_printer t;
	t.add_row() << 0 << modes[0]->size() << "noise";
	for (int i = 1, iend = modes.size(); i < iend; ++i) {
		string func;
		modes[i]->get_function_string(func);
		t.add_row() << i << modes[i]->size() << func;
	}
	t.print(os);
}

void EM::get_mode_function_string(int m, string &s) const {
	modes[m]->get_function_string(s);
}

void inst_info::serialize(ostream &os) const {
	serializer(os) << mode << minfo;
}

void inst_info::unserialize(istream &is) {
	unserializer(is) >> mode >> minfo;
}

void inst_info::mode_info::serialize(ostream &os) const {
	serializer(os) << error << error_stale << role_map;
}

void inst_info::mode_info::unserialize(istream &is) {
	unserializer(is) >> error >> error_stale >> role_map;
}

int EM::classify(int target, const scene_sig &sig, const relation_table &rels, const rvec &x, vector<int> &role_map, rvec &vote_trace) const {
	vector<int> votes, order;
	clsfr.classify(target, sig, rels, x, votes, vote_trace);
	
	loggers->get(LOG_EM) << "votes:" << endl;
	for (int i = 0, iend = votes.size(); i < iend; ++i) {
		loggers->get(LOG_EM) << i << " = " << votes[i] << endl;
	}
	
	get_ordering(votes, order);
		
	/*
	 The scene has to contain the objects used by the linear model of
	 a mode for it to possibly qualify for that mode.
	*/
	for (int i = order.size() - 1; i >= 0; --i) {
		if (order[i] == 0) {
			// don't need mapping for noise mode
			return 0;
		}
		em_mode &m = *modes[order[i]];
		if (m.num_roles() > sig.size()) {
			continue;
		}
		role_map.clear();
		if (!m.map_roles(target, sig, rels, role_map)) {
			loggers->get(LOG_EM) << "mapping failed for " << i << endl;
			continue;
		}
		
		// mapping worked, classify as this mode;
		loggers->get(LOG_EM) << "best mode = " << order[i] << endl;
		return order[i];
	}
	
	FATAL("Reached unreachable line in EM::classify");
	return -1;
}

sig_info::sig_info() {}

void sig_info::serialize(ostream &os) const {
	serializer(os) << members << noise;
}

void sig_info::unserialize(istream &is) {
	unserializer(is) >> members >> noise;
}

