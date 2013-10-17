#ifndef PARAMS_H
#define PARAMS_H

/* Initial lambda value for ridge regression */
#define RIDGE_LAMBDA 1e-8

/*
 A local model does not need to be refit to data if its training error
 increases by less than this factor with a new data point, assuming the
 error is less than MODEL_ERROR_THRESH
*/
#define REFIT_MUL_THRESH 1.0001

/*
 Any two numbers within this amount of each other can be considered
 identical. This is also used to "zero-out" very small values.
*/
#define SAME_THRESH 1e-15

/* Probability that any instance is just noise */
#define NUM_STDEVS_THRESH 3

/*
 Number of noise instances required before EM tries to create a new
 mode out of them
*/
#define NEW_MODE_THRESH 40

/*
 Number of times EM tries to create a model from the same set of noise
 instances
*/
#define SEL_NOISE_MAX_TRIES 10

/*
 When trying to unify a new model with an existing one, the training
 error of the resulting model must be within this factor of the training
 error of the original
*/
#define UNIFY_MUL_THRESH 1.00001

/*
 In the controller, if the new prediction for the final state of a cached
 trajectory differs at least this much from the original prediction,
 then discard the cached trajectory, because it was computed using an
 outdated model.
*/
#define STATE_DIFF_THRESH 0.001


#define LASSO_LAMBDA 1e-8
#define LASSO_TOL 1e-10
#define LASSO_MAX_ITERS 50

#define FOIL_GROW_RATIO 0.75
#define FOIL_MIN_SUCCESS_RATE 0.5
#define FOIL_MAX_CLAUSE_LEN 5

#define EM_LDA_TRAIN_RATIO 0.75
#define LINEAR_SUBSET_MAX_ITERS 10
#define MINI_EM_MAX_ITERS 10

#define MAX_CLAUSE_FP_RATE .1

// px py pz rx ry rz sx sy sz
#define NUM_NATIVE_PROPS 9

#define INTERSECT_THRESH 1.0e-9

#define RANSAC_MAX_ITERS 100
#define RANSAC_LOG_ALARM_RATE -2.995732273553991  // log(0.05)

#define DEF_VIEWER_PORT "20195"

#endif
