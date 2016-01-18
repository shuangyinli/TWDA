/*============================================================================= 
#  Author:          Shuangyin Li
#  Email:           lishyin@mail2.sysu.edu.cn
#  School:          Sun Yat-sen University
=============================================================================*/ 

#include "twda-estimate.h"


double likehood(Document** corpus, sslda_model* model) {
    int num_docs = model->num_docs;
    double lik = 0.0;
    for (int d = 0; d < num_docs; d++) {
        double temp_lik = compute_doc_likehood2(corpus[d],model);
        lik += temp_lik;
        corpus[d]->lik = temp_lik;
    }
    return lik;
}

double compute_doc_likehood2(Document* doc, sslda_model* model) {
    double* log_topic = doc->topic;
    double* log_theta = model->log_theta;
    double* log_phi = model->log_phi;
    double* rou = doc->rou;
    double sigma_rou = 0;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    memset(log_topic, 0, sizeof(double) * num_topics);
    bool* reset_log_topic = new bool[num_topics];
    memset(reset_log_topic, false, sizeof(bool) * num_topics);
    double sigma_xi = 0;
    double* xi = doc->xi;
    int doc_num_lables = doc->num_labels;
    double lik = 0.0;
    for (int i = 0; i < doc_num_lables; i++) {
        sigma_xi += xi[i];
    }
    for(int i=0; i<num_topics; i++){
        sigma_rou += rou[i];
    }


    for (int i = 0; i < doc_num_lables-1; i++) {
        int labelid = doc->labels_ptr[i];
        
        for (int k = 0; k < num_topics; k++) {
            if (!reset_log_topic[k]) {
                log_topic[k] = log_theta[labelid * num_topics + k] + log(xi[i]) - log(sigma_xi);
                reset_log_topic[k] = true;
            }
            else log_topic[k] = util::log_sum(log_topic[k], log_theta[labelid * num_topics + k] + log(xi[i]) - log(sigma_xi));
        }
    }

    //int last_tagid = doc->labels_ptr[doc_num_lables];

        for(int k = 0; k < num_topics; k++){

            double tem_rou = log(rou[k] / sigma_rou);
            double tem_xi = log(xi[doc_num_lables-1]) - log(sigma_xi);
            double tem_tk = log_topic[k];

            log_topic[k] = util::log_sum(tem_tk, tem_rou + tem_xi);
        }

    int doc_num_words = doc->num_words;
    for (int i = 0; i < doc_num_words; i++) {
        double temp = 0;
        int wordid = doc->words_ptr[i];
        temp = log_topic[0] + log_phi[wordid];
        for (int k = 1; k < num_topics; k++) temp = util::log_sum(temp, log_topic[k] + log_phi[k * num_words + wordid]);
        lik += temp * doc->words_cnt_ptr[i];
    }
    delete[] reset_log_topic;
    return lik;
}

double compute_doc_likehood(Document* doc, sslda_model* model) {
    double* log_topic = doc->topic;
    double* log_theta = model->log_theta;
    double* log_phi = model->log_phi;
    int num_topics = model->num_topics;
    int num_words = model->num_words;
    memset(log_topic, 0, sizeof(double) * num_topics);
    bool* reset_log_topic = new bool[num_topics];
    memset(reset_log_topic, false, sizeof(bool) * num_topics);
    double sigma_xi = 0;
    double* xi = doc->xi;
    int doc_num_lables = doc->num_labels;
    double lik = 0.0;
    for (int i = 0; i < doc_num_lables; i++) {
        sigma_xi += xi[i];
    }

    for (int i = 0; i < doc_num_lables; i++) {
        int labelid = doc->labels_ptr[i];
        //double temp = log_theta[labelid * num_topics] + log(xi[i]) - log(sigma_xi);
        for (int k = 0; k < num_topics; k++) {
            if (!reset_log_topic[k]) {
                log_topic[k] = log_theta[labelid * num_topics + k] + log(xi[i]) - log(sigma_xi);
                reset_log_topic[k] = true;
            }
            else log_topic[k] = util::log_sum(log_topic[k], log_theta[labelid * num_topics + k] + log(xi[i]) - log(sigma_xi));
        }
    }
    int doc_num_words = doc->num_words;
    for (int i = 0; i < doc_num_words; i++) {
        double temp = 0;
        int wordid = doc->words_ptr[i];
        temp = log_topic[0] + log_phi[wordid];
        for (int k = 1; k < num_topics; k++) temp = util::log_sum(temp, log_topic[k] + log_phi[k * num_words + wordid]);
        lik += temp * doc->words_cnt_ptr[i];
    }
    delete[] reset_log_topic;
    return lik;
}

double cal_precision(Document** corpus, Document** unlabel_corpus, sslda_model* model,char* tmp_dir,int num_round) {
    int num_correct = 0;
    int num_labels = model->num_labels;
    int num_docs = model->num_docs;
    char filename[1000];
    FILE* fp=NULL;
    if (tmp_dir) {
        sprintf(filename,"%s/%d.xi",tmp_dir, num_round);
        fp = fopen(filename,"w");
    }
    for (int d = 0; d < num_docs; d++) {
        Document* doc = unlabel_corpus[d];
        int max_xi_labelid = 0;
        int xi_value = doc->xi[0];
        double sigma_xi = doc->xi[0];
        for (int l = 1; l < num_labels; l++) {
            sigma_xi += doc->xi[l];
            if (xi_value < doc->xi[l]) {
                xi_value = doc->xi[l];
                max_xi_labelid = l;
            }
        }
        if (max_xi_labelid == corpus[d]->labels_ptr[0]) num_correct += 1;
        for (int l = 0; fp && l < num_labels; l++) {
            if(fp) fprintf(fp,"%lf ",doc->xi[l]/sigma_xi);
        }
        if (fp) fprintf(fp,"\n");
    }
    if (fp) fclose(fp);
    return double(num_correct)/num_docs;
}

static double* sorted_array_ptr = NULL;
bool cmp(int i, int j) {
    return sorted_array_ptr[i] > sorted_array_ptr[j];
}

double cal_map(Document** corpus, Document** unlabel_corpus, sslda_model* model, int at_num, char* tmp_dir, int num_round) {
    int num_labels = model->num_labels;
    int num_docs = model->num_docs;
    char filename[1000];
    FILE* fp = NULL;
    if (tmp_dir) {
        sprintf(filename, "%s/%d.xi", tmp_dir, num_round);
        fp = fopen(filename, "w");
    }
    int* idx = new int[num_labels];
    for (int i = 0; i < num_labels; i++) idx[i] = i;
    double map = 0.0;
    for (int d = 0; d < num_docs; d++) {
        Document* doc = unlabel_corpus[d];
        sorted_array_ptr = doc->xi;
        std::sort(idx, idx + num_labels, cmp);
        int correct_cnt = 0;
        double ap = 0;
        for (int j = 0; j < at_num; j++) {
            bool correct = false;
            for (int k = 0; !correct && k < corpus[d]->num_labels; k++) {
                if (corpus[d]->labels_ptr[k] == idx[j]) correct = true;
            }
            if (correct) {
                correct_cnt ++;
                ap += double(correct_cnt)/(j+1);
            }
        }
        ap /= std::min(corpus[d]->num_labels, at_num); 
        map+=ap;
        double sigma_xi = 0.0;
        for (int l = 0; fp && l < num_labels; l++) sigma_xi += doc->xi[l];
        for (int l = 0; fp && l < num_labels; l++) {
            if(fp) fprintf(fp,"%lf ",doc->xi[l]/sigma_xi);
        }
        if (fp) fprintf(fp,"\n");
    }
    if(fp) fclose(fp);
    return map/num_docs;
}
