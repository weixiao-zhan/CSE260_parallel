# This makefile is intended for the GNU C compiler.
# Your code must compile (with GCC) with the given CFLAGS.
# You may experiment with the MY_OPT variable to invoke additional compiler options

#########################################################################
#									#
# Sample makefile header for running with Gnu compilers  		#
# Both MPI and OMP compilation may be independently disabled		#
#									#
#  The makefile targets are appended to  the end of this file		#
#	 Don't change anything that comes before the targets 		#
#									#
#									#
#########################################################################

RM		= rm -f
LN		= ln -s
ECHO		= echo

STDCPP          = --std=c11
STDC            = --std=c11

C++ 		= g++ $(STDCPP)
CC 		= gcc $(STDC)
# C++		= cclang 
# CC		= clang 
ifeq ($(omp),1)
OPTIMIZATION    +=  -fopenmp
endif

#include openblas library
# LIB_BLAS =  -lopenblas -lpthread -lm
INCLUDES += -I/usr/include/openblas
LIB_BLAS = -L/opt/arm/armpl_22.0.2_gcc-11.2/lib -larmpl -lpthread -lm
# INCLUDES += -I/opt/arm/armpl_22.0.2_gcc-11.2/include
LDLIBS 		+= $(LIB_BLAS)

AR		= ar
RANLIB		= ranlib
C++LINK		= $(C++)
CLINK		= $(CC)

# This generates output about how the
# compiler vectorized the code
# We  suggest using level 2 (the integer after "verbose=")
# See the gcc manual for the other levels of output: levels 1-7
# http://gcc.gnu.org/onlinedocs/gcc-4.7.3/gcc/Debugging-Options.html#Debugging-Options
# REPORT          = -ftree-vectorizer-verbose=2
# OPTIMIZATION += -ftree-vectorize


# ARCH_FLAGS      =  -m64
WARNINGS        = 
OPTIMIZATION    =  
ifeq ($(omp),1)
OPTIMIZATION    +=  -fopenmp
endif

C++FLAGS        += $(INCLUDES) $(ARCH_FLAGS) $(WARNINGS) $(OPTIMIZATION) \
                  $(XTRAFLAGS) $(DEBUG) $(REPORT) $(MY_OPT)

CFLAGS		+= $(INCLUDES) $(ARCH_FLAGS) $(WARNINGS) $(OPTIMIZATION) \
                  $(XTRAFLAGS) $(DEBUG) $(REPORT) $(MY_OPT)

FFLAGS		= $(ARCH_FLAGS) -O2 -fno-second-underscore -ff90 -fugly-complex

ARFLAGS		= ru

LDFLAGS		= $(WARNINGS) $(OPTIMIZATION) $(DEBUG) $(ARCH_FLAGS)
LDLIBS		+= -lm -pthread

ARCH_HAS_X	= arch_has_X

#########################################################################
# End of the System dependent prefix
#########################################################################


#########################################################################
#									#
# Suffixes for compiling most normal C++ and  C files		        #
#									#
#########################################################################

.SUFFIXES:
.SUFFIXES: .C .cpp .c .o

.C.o:
		@$(ECHO)
		@$(ECHO) "Compiling Source File --" $<
		@$(ECHO) "---------------------"
		$(C++) $(C++FLAGS) -c $<
		@$(ECHO)

.cpp.o:
		@$(ECHO)
		@$(ECHO) "Compiling Source File --" $<
		@$(ECHO) "---------------------"
		$(C++) $(C++FLAGS) -c $<
		@$(ECHO)



.c.o:
		@$(ECHO)
		@$(ECHO) "Compiling Source File --" $<
		@$(ECHO) "---------------------"
		$(CC) $(CFLAGS) -c $<
		@$(ECHO)


## Original Make file

# CFLAGS += -mfma
CFLAGS += -mcpu=neoverse-v1
CFLAGS += "-g"
CFLAGS += "-DOPENBLAS_SINGLETHREAD"
CFLAGS += -mtune=neoverse-v1
CFLAGS += -march=armv8.4-a+sve+f64mm
CFLAGS += -msve-vector-bits=256
CFLAGS += --param aarch64-autovec-preference=4
ifeq ($(debug), 1)
	MY_OPT = "-O0"
else
	MY_OPT = "-O4"
endif
MY_OPT += "-DOPENBLAS_SINGLETHREAD"
# MY_OPT += "-fPIC"
# MY_OPT = "-O3"
# MY_OPT = "-O4"
# If you want to change your optimization settings, do it here.

# WARNINGS += -Wall -pedantic
WARNINGS += -Wall 

# If you want to copy data blocks to contiguous storage
# This applies to the hand code version
ifeq ($(copy), 1)
    C++FLAGS += -DCOPY
    CFLAGS += -DCOPY
endif


# If you want to use restrict pointers, make restrict=1
# This applies to the hand code version
ifeq ($(restrict), 1)
    C++FLAGS += -D__RESTRICT
    CFLAGS += -D__RESTRICT
# ifneq ($(CARVER), 0)
#    C++FLAGS += -restrict
#     CFLAGS += -restrict
# endif
endif

C++FLAGS += -DNO_BLAS
CFLAGS += -DNO_BLAS

ifeq ($(debug), 1)
	CFLAGS += -g -DDEBUG
endif

OPTIMIZATION = $(MY_OPT)

targets = benchmark-naive benchmark-blas benchmark-blislab
BLISLAB = blislab/bl_dgemm_ukr.c  blislab/my_dgemm.c blislab/bl_dgemm_util.c
#	blislab/bl_dgemm_asm_4x12.c \
#	blislab/bl_dgemm_asm_8x4.c
objects = benchmark.o \
	bl_dgemm_util.o \
	blas/dgemm-blas.o \
	naive/dgemm-naive.o \
	$(BLISLAB:.c=.o)

UTIL   = wall_time.o cmdLine.o debugMat.o blislab/bl_dgemm_util.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o naive/dgemm-naive.o  $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o blas/dgemm-blas.o $(UTIL)
	$(CC) -o $@ $^ $(LDLIBS) -static
benchmark-blislab : $(BLISLAB:.c=.o) benchmark.o $(UTIL) 
	$(CC) -o $@ $^ $(LDLIBS) -pg -static

%.o : %.c
	$(CC) -c $(CFLAGS) -c $< -o $@ $(OPTIMIZATION)


%.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(OPTIMIZATION)



.PHONY : clean
clean:
	rm -f $(targets) $(objects) $(UTIL) core

