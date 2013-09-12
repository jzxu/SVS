#ifndef LINEAR_H
#define LINEAR_H

#include <vector>
#include <string>
#include "common.h"
#include "timer.h"
#include "serializable.h"
#include "mat.h"
#include "scene_sig.h"

enum regression_type { OLS, RIDGE, LASSO, FORWARD };

void clean_lr_data(mat &X, std::vector<int> &used_cols);

bool linreg(regression_type t, mat &X, mat &Y, const cvec &w, double var, bool cleaned, mat &coefs, rvec &inter);

bool nfoldcv(const_mat_view X, const_mat_view Y, double var, int n, regression_type t, rvec &avg_error);

#endif
