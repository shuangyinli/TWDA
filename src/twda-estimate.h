/*============================================================================= 
#  Author:          Shuangyin Li
#  Email:           shuangyinli@cse.ust.hk
#  School:          Sun Yat-sen University
=============================================================================*/ 

#ifndef TWDA_ESTIMATE_H
#define TWDA_ESTIMATE_H

#include "utils.h"
#include "twda.h"
#include <algorithm>

double likehood(Document** corpus, sslda_model* model);
double compute_doc_likehood(Document* doc, sslda_model* model);
double compute_doc_likehood2(Document* doc, sslda_model* model);

//double perplexity(Document** corpus, sslda_model* model);

double cal_precision(Document** corpus, Document** unlabel_corpus, sslda_model* model,char* tmp_dir=NULL,int num_round=0);
double cal_map(Document** corpus, Document** unlabel_corpus, sslda_model* model, int at_num=10, char* tmp_dir=NULL, int num_round=0);

#endif
