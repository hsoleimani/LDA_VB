#include "main.h"



int main(int argc, char* argv[])
{

	char task[40];
	char dir[400];
	char corpus_file[400];
	char testcorpus[400];
	char model_name[400];
	char init[400];
	int ntopics;
	double alpha;
	double nu;
	long int seed;

	KAPPA = 0.5;
	TAU = 1.0;
	BATCHSIZE = 50;

	seed = atoi(argv[1]);

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

	printf("SEED = %ld\n", seed);

	MAXITER = 1000;
	CONVERGED = 1e-4;

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);

    if (argc > 1)
    {
        if (strcmp(task, "est_stoch")==0)
        {
        	strcpy(testcorpus,argv[4]);
        	ntopics = atoi(argv[5]);
        	strcpy(init,argv[6]);
			strcpy(dir,argv[7]);
			alpha = atof(argv[8]);
			nu = atof(argv[9]);
			TAU = atof(argv[10]);
			KAPPA = atof(argv[11]);
			BATCHSIZE = atoi(argv[12]);
			train_stochastic(corpus_file, testcorpus, ntopics, init, dir, alpha, nu);

			gsl_rng_free (r);
            return(0);
        }
        if (strcmp(task, "est")==0)
        {
        	ntopics = atoi(argv[4]);
			strcpy(init,argv[5]);
			strcpy(dir,argv[6]);
			alpha = atof(argv[7]);
			nu = atof(argv[8]);
			if ((strcmp(init,"loadbeta")==0))
				strcpy(model_name,argv[9]);
			train(corpus_file, ntopics, init, dir, alpha, nu, model_name);

			gsl_rng_free (r);
            return(0);
        }
        if (strcmp(task, "inf")==0)
        {
			strcpy(model_name,argv[4]);
			strcpy(dir,argv[5]);
			test(corpus_file, model_name, dir);

			gsl_rng_free (r);
            return(0);
        }
    }
    return(0);
}



void train_stochastic(char* dataset, char* test_dataset, int ntopics, char* start,
		char* dir, double alpha, double nu)
{
    FILE* lhood_fptr;
    FILE* fp;
    char string[100];
    char filename[100];
    int iteration;
	double lhood, prev_lhood, conv, doclkh;
	double y, rho, temp;
	int d, n, j, s;
	int maxlen, tmaxlen, trmaxlen;
    vblda_corpus* corpus;
    vblda_corpus* test_corpus;
    vblda_model *model = NULL;
    vblda_ss* ss = NULL;
    vblda_var* var = NULL;
    time_t t1,t2;

    double** theta;
    double** test_theta;

    corpus = read_data(dataset, &trmaxlen);
    test_corpus = read_data(test_dataset, &tmaxlen);

    if (trmaxlen > tmaxlen)
    	maxlen = trmaxlen;
    else
    	maxlen = tmaxlen;


    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);


    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

	model = new_vblda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
	var = new_vblda_var(model, maxlen);
	ss = new_vblda_ss(model);

    if (strcmp(start, "seeded")==0){
    	printf("seeded\n");
		corpus_initialize_model(model, corpus, ss, var);
    }
    else{// (strcmp(start, "random")==0){
    	printf("random\n");
		random_initialize_model(model, corpus, ss, var);
    }


    //theta
    theta = malloc(sizeof(double*)*model->m);
    for (j = 0; j < model->m; j++){
    	theta[j] = malloc(sizeof(double)*corpus->ndocs);
    	for(d = 0; d < corpus->ndocs; d++){
    		theta[j][d] = 0.0;
    	}
    }
    //test theta
    test_theta = malloc(sizeof(double*)*model->m);
    for (j = 0; j < model->m; j++){
    	test_theta[j] = malloc(sizeof(double)*test_corpus->ndocs);
    	for(d = 0; d < test_corpus->ndocs; d++){
    		test_theta[j][d] = 0.0;
    	}
    }

	//init ss
	for (j = 0; j < model->m; j++){
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			var->summu[j] += var->mu[j][n];
		}
		for (n = 0; n < model->n; n++){
			y = var->mu[j][n]/var->summu[j];
			if (y == 0) y = 1e-50;
			model->expElogbeta[j][n] = y;
			model->Elogbeta[j][n] = log(y);
		}
	}

	//zero init tss
	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			ss->t[j][n] = 0.0;;
		}
	}
    iteration = 0;
    sprintf(filename, "%s/%03d", dir,iteration);
    printf("%s\n",filename);
	write_vblda_model(filename, model, var);

    time(&t1);
	prev_lhood = -1e100;
	conv = 1e5;
	do{
		rho = pow(1.0/(iteration+TAU), KAPPA);

		for (s = 0; s < BATCHSIZE; s++){
			// choose a document;
			d = floor(gsl_rng_uniform(r) * (corpus->ndocs-EPS));

			doclkh = doc_inference(&(corpus->docs[d]), model, ss, var, d, 0);
		}
		// update lambda
		for (j = 0; j < model->m; j++){
			var->summu[j] = (1-rho)*var->summu[j] + rho*((double)model->n*model->nu)
							+ rho*((double)corpus->ndocs*ss->sumt[j])/((double)BATCHSIZE);

			temp = gsl_sf_psi(var->summu[j]);
			for (n = 0; n < model->n; n++){

				var->mu[j][n] = (1-rho)*var->mu[j][n] + rho*(model->nu)
						+ rho*((double)corpus->ndocs*ss->t[j][n])/((double)BATCHSIZE);

				ss->t[j][n] = 0.0;

				model->Elogbeta[j][n] = (gsl_sf_psi(var->mu[j][n])-temp);
				model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);
			}
			ss->sumt[j] = 0.0;
		}


		if ((iteration%50) == 0){
			printf("***** VB ITERATION %d *****\n", iteration);
			// compute lkh on test set
			lhood = 0.0;
			for (d = 0; d < test_corpus->ndocs; d++){

				doclkh = doc_inference(&(test_corpus->docs[d]), model, ss, var, d, 1);

				for (j = 0; j < model->m; j++){
					test_theta[j][d] = var->gamma[j]/var->sumgamma;
				}
				doclkh = 0.0;
				for (n = 0; n < test_corpus->docs[d].length; n++){
					temp = 0.0;
					for (j = 0; j < model->m; j++){
						temp += test_theta[j][d]*model->expElogbeta[j][test_corpus->docs[d].words[n]];
					}
					doclkh += (double) test_corpus->docs[d].counts[n]*log(temp);
				}
				lhood += doclkh;
			}

			conv = fabs(prev_lhood - lhood)/fabs(prev_lhood);
			prev_lhood = lhood;

			time(&t2);

		    // write theta
			sprintf(filename, "%s/%03d.theta", dir,1);
			fp = fopen(filename, "w");
			for (d = 0; d < test_corpus->ndocs; d++){
				for (j = 0; j < model->m; j++){
					fprintf(fp, "%5.10lf ", test_theta[j][d]);
				}
				fprintf(fp, "\n");
			}
			fclose(fp);
			sprintf(filename, "%s/%03d", dir,1);
			write_vblda_model(filename, model, var);

			fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, conv, (t2-t1));
			fflush(lhood_fptr);
		}


		iteration ++;

	}while((iteration < 1e5) || (conv > CONVERGED));

	lhood = 0.0;
	for (d = 0; d < test_corpus->ndocs; d++){

		doclkh = doc_inference(&(test_corpus->docs[d]), model, ss, var, d, 1);

		for (j = 0; j < model->m; j++){
			test_theta[j][d] = var->gamma[j]/var->sumgamma;
		}
		doclkh = 0.0;
		for (n = 0; n < test_corpus->docs[d].length; n++){
			temp = 0.0;
			for (j = 0; j < model->m; j++){
				temp += test_theta[j][d]*model->expElogbeta[j][test_corpus->docs[d].words[n]];
			}
			doclkh += (double) test_corpus->docs[d].counts[n]*log(temp);
		}
		lhood += doclkh;

	}
	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, conv, (t2-t1));
	fflush(lhood_fptr);
	fclose(lhood_fptr);

    // write test theta
	sprintf(filename, "%s/testfinal.theta", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < test_corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			fprintf(fp, "%5.10lf ", test_theta[j][d]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	// compute training theta

	for (d = 0; d < corpus->ndocs; d++){

		doclkh = doc_inference(&(corpus->docs[d]), model, ss, var, d, 1);

		for (j = 0; j < model->m; j++){
			theta[j][d] = var->gamma[j]/var->sumgamma;
		}

	}


    // write training theta
	sprintf(filename, "%s/testfinal.theta", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			fprintf(fp, "%5.10lf ", theta[j][d]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(filename, "%s/final", dir);
	write_vblda_model(filename, model, var);
}



// batch training
void train(char* dataset, int ntopics, char* start, char* dir, double alpha, double nu, char* model_name)
{

    FILE* lhood_fptr;
    FILE* fp;
    char string[100];
    char filename[100];
    int iteration;
	double lhood, prev_lhood, conv, doclkh;
	double y, temp;
	int d, n, j;
	int maxlen;
    vblda_corpus* corpus;
    vblda_model *model = NULL;
    vblda_ss* ss = NULL;
    vblda_var* var = NULL;
    time_t t1,t2;

    double** theta;

    corpus = read_data(dataset, &maxlen);


    mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);


    sprintf(string, "%s/likelihood.dat", dir);
    lhood_fptr = fopen(string, "w");

    if (strcmp(start, "random")==0){
    	printf("random\n");
    	model = new_vblda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
    	var = new_vblda_var(model, maxlen);
		ss = new_vblda_ss(model);

		random_initialize_model(model, corpus, ss, var);

    }
    if (strcmp(start, "seeded")==0){
    	printf("random\n");
    	model = new_vblda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
    	var = new_vblda_var(model, maxlen);
		ss = new_vblda_ss(model);

		corpus_initialize_model(model, corpus, ss, var);

    }
    else if (strcmp(start, "loadbeta")==0){ //not updated
    	printf("load beta\n");
    	printf("This init method does not work now!\n");
    	return;
    	model = new_vblda_model(ntopics, corpus->nterms, corpus->ndocs, alpha, nu);
		ss = new_vblda_ss(model);
	    sprintf(filename, "%s.beta", model_name);
	    printf("%s\n",filename);
	    fp = fopen(filename, "r");
	    for (n = 0; n < model->n; n++){
	    	for (j = 0; j < model->m; j++){
				fscanf(fp, " %lf", &y);
				var->mu[j][n] = y;
	    	}
	    }
	    fclose(fp);
    }

    //theta
    theta = malloc(sizeof(double*)*model->m);
    for (j = 0; j < model->m; j++){
    	theta[j] = malloc(sizeof(double)*corpus->ndocs);
    	for(d = 0; d < corpus->ndocs; d++){
    		theta[j][d] = 0.0;
    	}
    }

	//init ss

	for (j = 0; j < model->m; j++){
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			var->summu[j] += var->mu[j][n];
		}
		for (n = 0; n < model->n; n++){

			y = var->mu[j][n]/var->summu[j];
			if (y == 0) y = 1e-50;
			model->expElogbeta[j][n] = y;
			model->Elogbeta[j][n] = log(y);
		}
	}

	//zero init tss
	for (j = 0; j < model->m; j++){
		ss->sumt[j] = 0.0;
		for (n = 0; n < model->n; n++){
			ss->t[j][n] = 0.0;;
		}
	}
    iteration = 0;
    sprintf(filename, "%s/%03d", dir,iteration);
    printf("%s\n",filename);
	write_vblda_model(filename, model, var);

    time(&t1);
	prev_lhood = -1e100;

	do{

		printf("***** VB ITERATION %d *****\n", iteration);

		lhood = 0.0;
		for (d = 0; d < corpus->ndocs; d++){

			doclkh = doc_inference(&(corpus->docs[d]), model, ss, var, d, 0);

			lhood += doclkh;

			//save theta
			for (j = 0; j < model->m; j++){
				theta[j][d] = var->gamma[j];
			}
		}

		//update mu and lhood
		// For 10 initial iterations, we treat mu as word probabilities in a maximum likelihood setting.
		// Obviously, this does not maximize the ELBO. However, experimentally, it leads to better solutions.
		if ((iteration < 10)){
			for (j = 0; j < model->m; j++){
				var->summu[j] = ss->sumt[j];
				for (n = 0; n < model->n; n++){
					var->mu[j][n] = ss->t[j][n];

					y = var->mu[j][n]/var->summu[j];
					if (y == 0) y = 1e-50;
					model->expElogbeta[j][n] = (y);
					model->Elogbeta[j][n] = log(y);

					lhood += ss->t[j][n]*model->Elogbeta[j][n];
					ss->t[j][n] = 0.0;

				}
				ss->sumt[j] = 0.0;
				prev_lhood = -1e100;
			}
		}else{
			for (j = 0; j < model->m; j++){
				var->summu[j] = (double)model->n*model->nu + ss->sumt[j];
				temp = gsl_sf_psi(var->summu[j]);
				for (n = 0; n < model->n; n++){
					var->mu[j][n] = model->nu + ss->t[j][n];

					ss->t[j][n] = 0.0;

					lhood += lgamma(var->mu[j][n]);
					model->Elogbeta[j][n] = (gsl_sf_psi(var->mu[j][n])-temp);
					model->expElogbeta[j][n] = exp(model->Elogbeta[j][n]);
				}
				lhood -= lgamma(var->summu[j]);
				ss->sumt[j] = 0.0;
			}
		}

		if ((prev_lhood > lhood) && (iteration != 11)){
			printf("Warning: Likelihood is decreasing \n");
		}

		conv = fabs(prev_lhood - lhood)/fabs(prev_lhood);
		prev_lhood = lhood;

		time(&t2);

		sprintf(filename, "%s/%03d", dir,1);
		write_vblda_model(filename, model, var);

		fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, conv, (t2-t1));
		fflush(lhood_fptr);
		iteration ++;

	}while((iteration < MAXITER) && (conv > CONVERGED));
	fclose(lhood_fptr);



    // write posterior theta
	sprintf(filename, "%s/final.theta", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			fprintf(fp, "%5.10lf ", theta[j][d]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(filename, "%s/final", dir);
	write_vblda_model(filename, model, var);
}

double doc_inference(document* doc, vblda_model* model, vblda_ss* ss,
		vblda_var* var, int d, int test){

	int n, j, variter, w;
	double c, phisum, temp, cphi, maxval;
	double varlkh, prev_varlkh, conv;

	prev_varlkh = -1e100;
	conv = 0.0;
	variter = 0;

	//init vars
	var->sumgamma = 0.0;
	for (j = 0; j < model->m; j++){
		var->gamma[j] = model->alpha;
		var->sumgamma += var->gamma[j];
	}
	for (n = 0; n < doc->length; n++){
		w = doc->words[n];
		c = (double) doc->counts[n];
		phisum = 0.0;
		maxval = -1e100l;
		for (j = 0; j < model->m; j++){
			//var->phi[n][j] = model->expElogbeta[j][w];
			//phisum +=  var->phi[n][j];
			var->phi[n][j] = model->Elogbeta[j][w];
			if (var->phi[n][j] > maxval)	maxval = var->phi[n][j];
		}
		phisum = 0.0;
		for (j = 0; j < model->m; j++){
			var->phi[n][j] = exp(var->phi[n][j] - maxval);
			phisum += var->phi[n][j];
		}
		for (j = 0; j < model->m; j++){
			var->phi[n][j] /= phisum;
			//var->phi[n][j] = 1.0/((double)model->m);
			cphi = c*var->phi[n][j];
			var->gamma[j] += cphi;
			var->sumgamma += cphi;
		}
	}

	do{
		varlkh = 0.0;
		for (n = 0; n < doc->length; n++){
			w = doc->words[n];
			c = (double) doc->counts[n];

			phisum = 0.0;
			maxval = -1e100;
			for (j = 0; j < model->m; j++){
				var->oldphi[j] = var->phi[n][j];

				//var->phi[n][j] = exp(gsl_sf_psi(var->gamma[j]))*model->expElogbeta[j][w];
				//phisum += var->phi[n][j];
				var->phi[n][j] = gsl_sf_psi(var->gamma[j]) + model->Elogbeta[j][w];
				if (var->phi[n][j] > maxval)	maxval = var->phi[n][j];

			}
			phisum = 0.0;
			for (j = 0; j < model->m; j++){
				var->phi[n][j] = exp(var->phi[n][j] - maxval);
				phisum += var->phi[n][j];
			}
			for (j = 0; j < model->m; j++){

				var->phi[n][j] /= phisum;

				temp = c*(var->phi[n][j] - var->oldphi[j]);
				var->gamma[j] += temp;
				var->sumgamma += temp;

				if (var->phi[n][j] > 0){
					cphi = c*var->phi[n][j];
					varlkh += cphi*(model->Elogbeta[j][w]-log(var->phi[n][j]));
				}
			}
		}
		// The rest of the terms in lkh cancel each other when gamma is optimized
		varlkh -= lgamma(var->sumgamma);
		for (j = 0; j < model->m; j++){
			varlkh += lgamma(var->gamma[j]);
		}

		conv = fabs(prev_varlkh - varlkh)/fabs(prev_varlkh);
		if ((prev_varlkh > varlkh) && (conv > 1e-8)){
			printf("lkh is decreasing in doc %d, %lf %lf, %5.10e\n", d, varlkh, prev_varlkh, conv);
		}
		prev_varlkh = varlkh;
		variter ++;

	}while((variter < MAXITER) && (conv > CONVERGED));

	if (test == 0){
		for (n = 0; n < doc->length; n++){
			w = doc->words[n];
			c = (double) doc->counts[n];
			for (j = 0; j < model->m; j++){
				cphi = c*var->phi[n][j];
				varlkh -= cphi*model->Elogbeta[j][w];
				ss->t[j][w] += cphi;
				ss->sumt[j] += cphi;
			}
		}
	}
	return(varlkh);

}



void test(char* dataset, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	int iteration;
	int d, n, j, doclkh, maxlen;
	double lhood;
	//double sumt;

	vblda_corpus* corpus;
	vblda_model *model = NULL;
	vblda_ss* ss = NULL;
	vblda_var* var = NULL;
	time_t t1,t2;
	double** theta;
	//float x;
	//double y;

	corpus = read_data(dataset, &maxlen);

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	model = load_model(model_name, corpus->ndocs);
	var = new_vblda_var(model, maxlen);
	ss = new_vblda_ss(model);

    //theta
    theta = malloc(sizeof(double*)*model->m);
    for (j = 0; j < model->m; j++){
    	theta[j] = malloc(sizeof(double)*corpus->ndocs);
    	for(d = 0; d < corpus->ndocs; d++){
    		theta[j][d] = 0.0;
    	}
    }

	//*************************************
	for (j = 0; j < model->m; j++){
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			var->mu[j][n] = model->expElogbeta[j][n];
			var->summu[j] += var->mu[j][n];
		}
		for (n = 0; n < model->n; n++){
			model->expElogbeta[j][n] = (var->mu[j][n]/var->summu[j]);
			model->Elogbeta[j][n] = log(model->expElogbeta[j][n]);
		}
	}

    iteration = 0;
    sprintf(filename, "%s/test%03d", dir,iteration);
    printf("%s\n",filename);
    write_vblda_model(filename, model, var);

    time(&t1);

	lhood = 0.0;
	for (d = 0; d < corpus->ndocs; d++){

		doclkh = doc_inference(&(corpus->docs[d]), model, ss, var, d, 1);
		lhood += doclkh;

		for (j = 0; j < model->m; j++){
			theta[j][d] = var->gamma[j];
		}
	}

	time(&t2);

	fprintf(lhood_fptr, "%d %5.5e %5.5e %5ld \n",iteration, lhood, 0.0, (t2-t1));
	fflush(lhood_fptr);
	fclose(lhood_fptr);
	//*************************************

    // write posterior theta
	sprintf(filename, "%s/testfinal.theta", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < corpus->ndocs; d++){
		for (j = 0; j < model->m; j++){
			fprintf(fp, "%5.10lf ", theta[j][d]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);


	sprintf(filename, "%s/testfinal", dir);
	write_vblda_model(filename, model, var);

}


vblda_model* new_vblda_model(int ntopics, int nterms, int ndocs, double alpha, double nu)
{
	int n, j;

	vblda_model* model = malloc(sizeof(vblda_model));
	model->D = ndocs;
    model->m = ntopics;
    model->n = nterms;
    model->nu = nu;
    model->expElogbeta = malloc(sizeof(double*)*ntopics);
    model->Elogbeta = malloc(sizeof(double*)*ntopics);
    for (j = 0; j < ntopics; j++){
    	model->expElogbeta[j] = malloc(sizeof(double)*nterms);
    	model->Elogbeta[j] = malloc(sizeof(double)*nterms);
		for (n = 0; n < nterms; n++){
			model->expElogbeta[j][n] = 0.0;
			model->Elogbeta[j][n] = 0.0;
		}
	}
    model->nu = nu;
    model->alpha = alpha;

    return(model);
}



vblda_ss * new_vblda_ss(vblda_model* model)
{
	int j, n;
	vblda_ss * ss;
    ss = malloc(sizeof(vblda_ss));
	ss->t = malloc(sizeof(double*)*model->m);
	ss->sumt = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		ss->sumt[j] = 0.0;
		ss->t[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			ss->t[j][n] = 0.0;
		}
	}

    return(ss);
}


vblda_var * new_vblda_var(vblda_model* model, int maxlen)
{
	int j, n;
	vblda_var * var;
    var = malloc(sizeof(vblda_var));

    var->sumgamma = 0.0;
	var->oldphi = malloc(sizeof(double)*model->m);
	var->gamma = malloc(sizeof(double)*model->m);
	var->mu = malloc(sizeof(double*)*model->m);
	var->summu = malloc(sizeof(double)*model->m);
	for (j = 0; j < model->m; j++){
		var->oldphi[j] = 0.0;
		var->gamma[j] = 0.0;
		var->summu[j] = 0.0;
		var->mu[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			var->mu[j][n] = 0.0;
		}
	}
	var->phi = malloc(sizeof(double*)*maxlen);
	for (n = 0; n < maxlen; n++){
		var->phi[n] = malloc(sizeof(double)*model->m);
		for (j = 0; j < model->m; j++){
			var->phi[n][j] = 0.0;
		}
	}

    return(var);
}


vblda_corpus* read_data(const char* data_filename, int* maxlen)
{
	FILE *fileptr;
	int length, count, word, n, nd, nw;
	vblda_corpus* c;

	*maxlen = 0;
	printf("reading data from %s\n", data_filename);
	c = malloc(sizeof(vblda_corpus));
	c->docs = 0;
	c->nterms = 0;
	c->ndocs = 0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "%10d", &length) != EOF)){
		c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
		c->docs[nd].length = length;
		c->docs[nd].total = 0;
		c->docs[nd].words = malloc(sizeof(int)*length);
		c->docs[nd].counts = malloc(sizeof(int)*length);
		if (length > *maxlen)
			*maxlen = length;
		for (n = 0; n < length; n++){
			fscanf(fileptr, "%10d:%10d", &word, &count);
			c->docs[nd].words[n] = word;
			c->docs[nd].counts[n] = count;
			c->docs[nd].total += count;
			if (word >= nw)
				nw = word + 1;
		}
		nd++;
	}
	fclose(fileptr);
	c->ndocs = nd;
	c->nterms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);
	return(c);
}



void write_vblda_model(char * root, vblda_model * model, vblda_var* var)
{
    char filename[200];
    FILE* fileptr;
    int n, j;

    //beta
    sprintf(filename, "%s.beta", root);
    fileptr = fopen(filename, "w");
    for (n = 0; n < model->n; n++){
    	for (j = 0; j < model->m; j++){
    		fprintf(fileptr, "%.10lf ",var->mu[j][n]);
    	}
    	fprintf(fileptr, "\n");
    }
    fclose(fileptr);

	sprintf(filename, "%s.other", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr,"num_topics %d \n",model->m);
	fprintf(fileptr,"num_terms %d \n",model->n);
	fprintf(fileptr,"num_docs %d \n",model->D);
	fprintf(fileptr,"alpha %lf \n",model->alpha);
	fprintf(fileptr,"nu %lf \n",model->nu);
	fclose(fileptr);

}

vblda_model* load_model(char* model_root, int ndocs){

	char filename[100];
	FILE* fileptr;
	int j, n, num_topics, num_terms, num_docs;
	//float x;
	double y, alpha, nu;

	vblda_model* model;
	sprintf(filename, "%s.other", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics %d\n", &num_topics);
	fscanf(fileptr, "num_terms %d\n", &num_terms);
	fscanf(fileptr, "num_docs %d\n", &num_docs);
	fscanf(fileptr, "alpha %lf\n", &alpha);
	fscanf(fileptr, "nu %lf\n", &nu);
	fclose(fileptr);

	model  = new_vblda_model(num_topics, num_terms, ndocs, alpha, nu);
	model->n = num_terms;
	model->m = num_topics;
	model->D = ndocs;
	model->alpha = alpha;
	model->nu = nu;


	sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
    for (n = 0; n < num_terms; n++){
		for (j = 0; j < num_topics; j++){
			fscanf(fileptr, " %lf", &y);
			model->expElogbeta[j][n] = y; //for now. copy it to var->mu later
		}
	}
    fclose(fileptr);

    return(model);
}




void random_initialize_model(vblda_model * model, vblda_corpus* corpus, vblda_ss* ss, vblda_var* var){

	int n, j;
	double exp_par;
	//double* beta = malloc(sizeof(double)*model->n);
	//double* nu = malloc(sizeof(double)*model->n);

	/*for (n = 0; n < model->n; n++){
		beta[n] = 0.0;
		nu[n] = 0.0;
	}*/

	exp_par = (double)corpus->ndocs*100.0/((double)model->m*model->n);
	for (j = 0; j < model->m; j++){
		/*for (n = 0; n < model->n; n++){
			nu[n] = model->nu;
		}*/
		//gsl_ran_dirichlet (r, model->n, nu, beta);
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			//model->mu[j][n] = beta[n];
			var->mu[j][n] = model->nu + gsl_ran_exponential(r, exp_par);
			//var->mu[j][n] = model->nu + 1.0 + gsl_rng_uniform(r);
			var->summu[j] += var->mu[j][n];
		}
	}

  	//free(beta);
  	//free(nu);
}

void corpus_initialize_model(vblda_model * model, vblda_corpus* corpus, vblda_ss* ss, vblda_var* var){

	int n, j, d, i, cnt;
	int* docs = malloc(sizeof(int)*corpus->ndocs);
	for (d = 0; d < corpus->ndocs; d++){
		docs[d] = -1;
	}

	for (j = 0; j < model->m; j++){
		var->summu[j] = 0.0;
		for (n = 0; n < model->n; n++){
			var->mu[j][n] = model->nu;
			var->summu[j] += var->mu[j][n];
		}

		for (i = 0; i < 40; i++){
			//choose a doc
			cnt = 0;
			do{
				d = floor(gsl_rng_uniform(r)*corpus->ndocs);
				if ((docs[d] ==- 1) || (cnt > 100)){
					docs[d] = j;
					break;
				}
				cnt += 1;
			}while(1);

			for (n = 0; n < corpus->docs[d].length; n++){
				var->mu[j][corpus->docs[d].words[n]] += (double) corpus->docs[d].counts[n];
				var->summu[j] += (double) corpus->docs[d].counts[n];
			}
		}
		for (n = 0; n < model->n; n++){
			var->mu[j][n] *= (double)corpus->ndocs/var->summu[j];
		}
		var->summu[j] = (double)corpus->ndocs;

	}

  	free(docs);
}

