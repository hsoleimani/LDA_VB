import os
import numpy as np

codepath = './lda_vb'
dirpath = 'vblda'


seed = 1399677451
np.random.seed(seed)

M =  # number of topics 

alpha = 1.0/float(M)
nu = 0.01;

#*************** main run
trdocs = '' # training docs in lda-c format
tdocs = '' # test docs in lda-c format

os.system('mkdir -p ' + dirpath)
if 1: # stochastic
	tau = 1.0;
	kappa = 0.9;
	BATCHSIZE = 10
	s1 = np.random.randint(seed)
	cmdtxt = codepath + ' ' + str(s1) + ' est_stoch ' + trdocs + ' ' + tdocs + ' ' + str(M) + ' seeded '+ dirpath
	cmdtxt += ' ' + str(alpha) + ' ' + str(nu) + ' ' + str(tau) + ' ' + str(kappa) + ' ' + str(BATCHSIZE)
	print(cmdtxt)
	os.system(cmdtxt)


else: # batch
	# train model
	s1 = np.random.randint(seed)
	cmdtxt = codepath + ' ' + str(s1) + ' est ' + trdocs + ' ' + str(M) + ' seeded '+ dirpath
	cmdtxt += ' ' + str(alpha) + ' ' + str(nu)
	print(cmdtxt)
	os.system(cmdtxt)
	
	# test model
	s1 = np.random.randint(seed)
	cmdtxt = codepath + ' ' + str(s1) + ' inf ' + tdocs + ' ' + dirpath + '/final ' + dirpath
	os.system(cmdtxt)

