CC = gcc -O3 -Wall -g
#CC = g++ -ansi -Wall -pedantic
#CFLAGS = -g -Wall -O3 -ffast-math -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
# CFLAGS = -g -Wall
LDFLAGS = -lgsl -lgslcblas -lm

GSL_INCLUDE = /usr/local/include/gsl
GSL_LIB = /usr/local/lib


LSOURCE = main.c
LHEADER = vblda.h main.h

all: $(LSOURCE) $(HEADER)
	  $(CC) -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) $(LDFLAGS) -o lda_vb

clean:
	-rm -f lda_vb