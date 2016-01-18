/*============================================================================= 
#  Author:          Shuangyin Li
#  Email:           shuangyinli@cse.ust.hk
#  School:          Sun Yat-sen University
=============================================================================*/ 

#ifndef SSLDA_INFERENCE_H
#define SSLDA_INFERENCE_H

#include "utils.h"
#include "twda.h"
#include "pthread.h"
#include "unistd.h"
#include "stdlib.h"
#include "twda-estimate.h"

struct Thread_Data {
    Document** corpus;
    int start;
    int end;
    Config* config;
    sslda_model* model;
    //double* rous;
    Thread_Data(Document** corpus_, int start_, int end_, Config* config_, sslda_model* model_) : corpus(corpus_), start(start_), end(end_), config(config_), model(model_) {
    }
};

void inference_gamma(Document* doc, sslda_model* model);
void inference_xi(Document* doc, sslda_model* model,Config* config);
void run_thread_inference(Document** corpus, sslda_model* model, Config* config);



#endif
