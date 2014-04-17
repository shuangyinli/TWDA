/*============================================================================= 
#  Author:          Shuangyin Li, JiefeiLi
#  Email:           lishyin@mail2.sysu.edu.cn, lijiefei@mail2.sysu.edu.cn 
#  School:          Sun Yat-sen University
=============================================================================*/ 

#include "twda-learn.h"


int converged (double *u, double *v, int n, double threshold)
{
    /* return 1 if |a - b|/|a| < threshold */
    double us = 0;
    double ds = 0;
    double d;
    int i;
    
    for (i = 0; i < n; i++)
        us += u[i] * u[i];

    for (i = 0; i < n; i++) {
        d = u[i] - v[i];
        ds += d * d;
    }

    if (sqrt(ds / us) < threshold)
        return 1;
    else
        return 0;

}

//Newton methods for mu
void learn_mu(sslda_model* model, Document** corpus, int level){

    int M = model->num_docs;
    int K = model->num_topics;

    int i, j, t;
    double *g, *h, *pg, *palpha;
    double z, sh, hgz;
    double psg, spg, gs;
    double alpha0, palpha0;

    double* rous = new double[M * K];

    for(i=0; i<M; i++){
        for(j=0; j<K; j++){
            rous[i*K + j] = corpus[i]->rou[j];
        }
    }

/* allocate arrays */
    if ((g = (double *)calloc(K, sizeof(double))) == NULL) {
        fprintf(stderr, "newton:: cannot allocate g.\n");
        return;
    }
    if ((h = (double *)calloc(K, sizeof(double))) == NULL) {
        fprintf(stderr, "newton:: cannot allocate h.\n");
        return;
    }
    if ((pg = (double *)calloc(K, sizeof(double))) == NULL) {
        fprintf(stderr, "newton:: cannot allocate pg.\n");
        return;
    }
    if ((palpha = (double *)calloc(K, sizeof(double))) == NULL) {
        fprintf(stderr, "newton:: cannot allocate palpha.\n");
        return;
    }

    /* initialize */
    if (level == 0)
    {
        for (i = 0; i < K; i++) {
            for (j = 0, z = 0; j < M; j++)
                z += rous[j*K+i];
            model->mu[i] = z / (M * K);
        }
    } else {
        for (i = 0; i < K; i++) {
            for (j = 0, z = 0; j < M; j++)
                z += rous[j*K+i];
            model->mu[i] = z / (M * K * pow(10, level));
        }
    }

    psg = 0;
    for (i = 0; i < M; i++) {
        for (j = 0, gs = 0; j < K; j++)
            gs += rous[i*K+j];
        psg += util::digamma(gs);
    }
    for (i = 0; i < K; i++) {
        for (j = 0, spg = 0; j < M; j++)
            spg += util::digamma(rous[j*K+i]);
        pg[i] = spg - psg;
    }

    /* main iteration */
    for (t = 0; t < MAX_NEWTON_ITERATION; t++)
    {
        for (i = 0, alpha0 = 0; i < K; i++)
            alpha0 += model->mu[i];
        palpha0 = util::digamma(alpha0);
        
        for (i = 0; i < K; i++)
            g[i] = M * (palpha0 - util::digamma(model->mu[i])) + pg[i];
        for (i = 0; i < K; i++)
            h[i] = - 1 / util::trigamma(model->mu[i]);
        for (i = 0, sh = 0; i < K; i++)
            sh += h[i];
        
        for (i = 0, hgz = 0; i < K; i++)
            hgz += g[i] * h[i];
        hgz /= (1 / util::trigamma(alpha0) + sh);

        for (i = 0; i < K; i++)
            model->mu[i] = model->mu[i] - h[i] * (g[i] - hgz) / M;
        
        for (i = 0; i < K; i++)
            if (model->mu[i] < 0) {
                if (level >= MAX_RECURSION_LIMIT) {
                    fprintf(stderr, "newton:: maximum recursion limit reached.\n");
                    exit(1);
                } else {
                    free(g);
                    free(h);
                    free(pg);
                    free(palpha);
                    return learn_mu(model, corpus, 1 + level);
                }
            }

        if ((t > 0) && converged(model->mu, palpha, K, 1.0e-4)) {
            free(g);
            free(h);
            free(pg);
            free(palpha);
            return;
        } else
            for (i = 0; i < K; i++)
                palpha[i] = model->mu[i];
        
    }
    fprintf(stderr, "newton:: maximum iteration reached. t = %d\n", t);
    
    delete[] rous;
    return;

}

void get_descent_pi(Document** corpus, sslda_model* model,double* descent_pi) {
    int num_labels = model->num_labels;
    int num_docs = model->num_docs;
    memset(descent_pi,0,sizeof(double)* num_labels);
    double* pi = model->pi;
    for (int d = 0; d < num_docs; d++) {
        double sigma_pi = 0.0;
        Document* doc = corpus[d];
        int doc_num_labels = doc->num_labels;
        double sigma_xi = 0.0;
        for (int i = 0; i < doc_num_labels; i++) {
            sigma_pi += pi[doc->labels_ptr[i]];
            sigma_xi += doc->xi[i];
        }
        for (int i = 0; i < doc_num_labels; i++) {
            int label_id = doc->labels_ptr[i];
            double pis = pi[label_id];
            descent_pi[label_id] += util::digamma(sigma_pi) - util::digamma(pis) + util::digamma(doc->xi[i]) - util::digamma(sigma_xi);
        }
    }
}

void init_pi(double* pi, int num_labels) {
    for (int i = 0; i < num_labels; i++) {
        pi[i] = util::random() * 2;
    }
}

double get_pi_function(Document** corpus, sslda_model* model) {
    double pi_function_value = 0.0;
    int num_docs = model->num_docs;
    double* pi = model->pi;
    for (int d = 0; d < num_docs; d++) {
        double sigma_pi = 0.0;
        double sigma_xi = 0.0;
        Document* doc = corpus[d];
        for (int i = 0; i < doc->num_labels; i++) {
            sigma_pi += pi[doc->labels_ptr[i]];
            sigma_xi += doc->xi[i];
        }
        pi_function_value += util::log_gamma(sigma_pi);
        for (int i = 0; i < doc->num_labels; i++) {
            int label_id = doc->labels_ptr[i];
            pi_function_value -= util::log_gamma(pi[label_id]);
            pi_function_value += (pi[label_id] - 1) * (util::digamma(doc->xi[i]) - util::digamma(sigma_xi));
        }
    }
    return pi_function_value;
}

void learn_pi(Document** corpus, sslda_model* model, Config* config) {
    int num_round = 0;
    int num_labels = model->num_labels;
    double* last_pi = new double [model->num_labels];
    //double* pi = new double[num_labels];
    double* descent_pi = new double[num_labels];
    double z;
    do {
        init_pi(model->pi,num_labels);
        z = get_pi_function(corpus,model);
    }
    while ( z < 0);
    double last_z;
    double learn_rate = config->pi_learn_rate;
    double eps = 1000;
    int max_pi_iter = config->max_pi_iter;
    double pi_min_eps = config->pi_min_eps;
    bool has_neg_value_flag = false;
    do {
        last_z = z;
        //for (int i = 0; i < num_labels; i++) last_pi[i] = pi[i]; //memcpy ?
        memcpy(last_pi,model->pi,sizeof(double) * num_labels);
        get_descent_pi(corpus,model,descent_pi); 
        for (int i = 0; !has_neg_value_flag && i < num_labels; i++) {
            model->pi[i] += learn_rate * descent_pi[i];
            if (model->pi[i] < 0) has_neg_value_flag = true;
        }
        if (has_neg_value_flag || last_z > (z=get_pi_function(corpus,model))) {
            learn_rate *= 0.1;
            z = last_z;
            //for ( int i = 0; i < num_labels; i++) pi[i] = last_pi[i];
            memcpy(model->pi,last_pi,sizeof(double) * num_labels);
            eps = 1000.0;
        }
        else eps = util::norm2(last_pi, model->pi, num_labels);
        num_round += 1;
    } 
    while (num_round < max_pi_iter && eps > pi_min_eps);
//    memcpy(model->pi,pi,sizeof(double)*num_labels);
    //for (int i = 0; i < num_labels; i++) model->pi[i] = pi[i];//memcpy(model->pi,pi,sizeof(double)*num_labels);
    delete[] last_pi;
    delete[] descent_pi;
}

void normalize_matrix_rows(double* mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double temp = 0;
        for (int j = 0; j < cols; j++) temp += mat[ i * cols + j];
        for (int j = 0; j < cols; j++) {
            mat[i*cols +j] /= temp;
            if (mat[i*cols + j] == 0)mat[i*cols + j] = 1e-300; ///????
        }
    }
}

void normalize_log_matrix_rows(double* log_mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        double temp = log_mat[ i * cols];
        if (isnan(temp) || isnan(-temp)) {
            printf("normalize nan\n");
        }
        for (int j = 1; j < cols; j++) temp = util::log_sum(temp, log_mat[i * cols + j]);
        if (isnan(-temp) || isnan(temp)) {
            printf("normalize nan\n");
        }
        for (int j = 0; j < cols; j++) log_mat[i*cols + j] -= temp;
    }
}

void learn_theta_phi(Document** corpus, sslda_model* model) {
    int num_docs = model->num_docs;
    int num_topics = model->num_topics;
        int num_words = model->num_words;
    //memset(model->theta, 0, sizeof(double) * model->num_labels * num_topics);
    //memset(model->phi,0,sizeof(double) * num_topics * model->num_words);
    bool* reset_theta_flag = new bool[model->num_labels * num_topics];
    memset(reset_theta_flag, 0, sizeof(bool) * model->num_labels * num_topics);
    bool* reset_phi_flag = new bool[num_topics * model->num_words];
    memset(reset_phi_flag, 0, sizeof(bool) * num_topics * model->num_words);
    for (int d = 0; d < num_docs; d++) {
        Document* doc = corpus[d];
        int doc_num_labels = doc->num_labels;
        int doc_num_words = doc->num_words;
        double sigma_xi = 0;
        for (int i = 0; i < doc_num_labels; i++)sigma_xi += doc->xi[i];
        for (int i = 0; i < doc_num_labels; i++) {
            int label_id = doc->labels_ptr[i];
            for (int k = 0; k < num_topics; k++) {
                for (int j = 0; j < doc_num_words; j++) {
                    if (!reset_theta_flag[label_id * num_topics + k]) {
                        reset_theta_flag[label_id * num_topics + k] = true;
                        model->log_theta[label_id * num_topics + k] = log(doc->words_cnt_ptr[j]) + doc->log_gamma[j * num_topics + k] + log(doc->xi[i]) - log(sigma_xi);
                    }
                    else {
                        //model->theta[label_id * num_topics + k] += doc->words_cnt_ptr[j] * doc->gamma[j * num_topics + k] * doc->xi[i]/sigma_xi;
                        model->log_theta[label_id * num_topics + k] = util::log_sum(model->log_theta[label_id * num_topics + k], log(doc->words_cnt_ptr[j]) +doc->log_gamma[j * num_topics + k] + log(doc->xi[i]) - log(sigma_xi));
                    }
                    /*if (isnan(model->theta[label_id * num_topics + k]) || isnan(-model->theta[label_id * num_topics + k])) {
                        printf("gamma:%lf xi:%lf\n", doc->gamma[j * num_topics + k], doc->xi[i]/sigma_xi);
                    }*/
                }
            }
        }
        for (int k = 0; k < num_topics; k++) {
            for (int i = 0; i < doc_num_words; i++) {
                int wordid = doc->words_ptr[i];
                if (!reset_phi_flag[k * num_words + wordid]) {
                    reset_phi_flag[k * num_words + wordid] = true;
                    model->log_phi[k * num_words + wordid] = log(doc->words_cnt_ptr[i]) + doc->log_gamma[i*num_topics + k];
                }
                else {
                    model->log_phi[k * num_words + wordid] = util::log_sum(model->log_phi[k * num_words + wordid], doc->log_gamma[i*num_topics + k] + log(doc->words_cnt_ptr[i]));
                }
                if (isnan(doc->log_gamma[i*num_topics +k]) || isnan(model->log_phi[k * num_words + wordid])) {
                    printf("%lf %lf\n",doc->log_gamma[i*num_topics +k], model->log_phi[k * num_words + wordid]);
                }
            }
        }
    }
    normalize_log_matrix_rows(model->log_theta, model->num_labels, num_topics);
    normalize_log_matrix_rows(model->log_phi, num_topics, model->num_words);
    delete[] reset_theta_flag;
    delete[] reset_phi_flag;
}

