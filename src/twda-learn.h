/*============================================================================= 
#  Author:          Shuangyin Li
#  Email:           shuangyinli@cse.ust.hk
#  School:          Sun Yat-sen University
=============================================================================*/ 

#ifndef SSLDA_LEARN_H
#define SSLDA_LEARN_H

#include "twda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

#define MAX_RECURSION_LIMIT  40
#define MAX_NEWTON_ITERATION 40

void learn_pi(Document** corpus, sslda_model* model, Config* config);
void learn_theta_phi(Document** corpus, sslda_model* model);
void learn_mu(sslda_model* model, Document** corpus, int level);
int  converged (double *u, double *v, int n, double threshold);

#endif
