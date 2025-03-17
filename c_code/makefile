
#-----------------------------------------------------------------------
#
#  simp_exam, makefile
#
#-----------------------------------------------------------------------

# Compiler
CC = g++ -O2 -std=c++11 
CFLAGS = 
LIBS = -lm 

# ADOL-C 
ADPATH = $(HOME)/AD/ADOL-C/adolc_base_coin/include/adolc
ADPATH1 = $(HOME)/AD/ADOL-C/adolc_base_coin/include
ADLIBDIR = $(HOME)/AD/ADOL-C/adolc_base_coin/lib64
ADCFLAGS = -I$(ADPATH) -I$(ADPATH1) 
ADLIBS = -Wl,--rpath -Wl,$(ADLIBDIR) -L$(ADLIBDIR) -ladolc

all: simp_exam nonl_exam nonl_exam1

#**************************************************************************
#                                                            SIMP_EXAM *

simp_exam : simp_exam.o 
	$(CC) -o simp_exam $(CFLAGS) \
               simp_exam.o $(LIBS)  $(ADLIBS) 

simp_exam.o : simp_exam.cpp
	$(CC) -c $(CFLAG) $(ADCFLAGS) simp_exam.cpp

#**************************************************************************
#                                                            NONL_EXAM *

nonl_exam : nonl_exam.o 
	$(CC) -o nonl_exam $(CFLAGS) \
               nonl_exam.o $(LIBS)  $(ADLIBS) 

nonl_exam.o : nonl_exam.cpp
	$(CC) -c $(CFLAG) $(ADCFLAGS) nonl_exam.cpp


#**************************************************************************
#                                                            NONL_EXAM *

nonl_exam1 : nonl_exam1.o 
	$(CC) -o nonl_exam1 $(CFLAGS) \
               nonl_exam1.o $(LIBS)  $(ADLIBS) 

nonl_exam1.o : nonl_exam1.cpp
	$(CC) -c $(CFLAG) $(ADCFLAGS) nonl_exam1.cpp



#**************************************************************************
#                                                                         *
clean:
	rm *.o; rm simp_exam; rm nonl_exam; rm nonl_exam1;





