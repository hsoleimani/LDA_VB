/*
 * vblda.h
 *
 *  Created on: Jan 3, 2014
 *      Author: Hossein Soleimani
 */

#ifndef VBLDA_H_
#define VBLDA_H_

#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>


#define NUM_INIT 20
#define EPS 1e-30
#define max(a,b) ({ __typeof__ (a) _a = (a);  __typeof__ (b) _b = (b); _a > _b ? _a : _b; })


typedef struct
{
    int* words;
    int* counts;
    int length;
    int total;
} document;


typedef struct
{
    document* docs;
    int nterms;
    int ndocs;
} vblda_corpus;


typedef struct vblda_model
{
    int m;
    int D;
    int n;
    double** expElogbeta;
    double** Elogbeta;
    double alpha;
    double nu;
} vblda_model;

typedef struct vblda_var
{
	double** mu;
	double* summu;
    double* gamma;
    double sumgamma;
    double** phi;
    double* oldphi;
} vblda_var;


typedef struct vblda_ss
{
    double** t;
    double* sumt;
} vblda_ss;


#endif /* VBLDA_H_ */
