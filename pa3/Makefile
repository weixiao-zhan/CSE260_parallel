BANG    =       $(shell expr match `hostname` ccom-bang)
BANG-COMPUTE   =  $(shell expr match `hostname` compute)
BANG-BANG = $(shell echo $(BANG)\&$(BANG-COMPUTE))
STAMPEDE = $(shell hostname | grep stampede | wc -c)
COMET = $(shell hostname | grep comet | wc -c)
SORKEN = $(shell hostname | grep sorken | wc -c)

# If you want to compile with MPI enabled,
# uncomment this line
mpi = 1

ifneq ($(STAMPEDE), 0)
include arch.intel-c++11.generic
else
include arch.gnu-c++11.generic
endif

CMD        = apf
THE_DATE    := $(shell date "+%y.%m.%d-%H.%M.%S-%Z")
THE_ARCHIVE = $(CMD)--$(THE_DATE)

ifneq ($(STAMPEDE), 0)
# https://software.intel.com/en-us/articles/overview-of-vectorization-reports-and-new-vec-report6
 REPORT =  -qopt-report=1
# Most detailed reporting
# REPORT =  -qopt-report=5

# REPORT =  -qopt-report-phase=vec
else
endif

ifneq ($(COMET), 0)
 include $(PUB)/Arch/arch.intel.c++11.generic
#  REPORT =  -qopt-report=1
 REPORT = -vec-report=1
else
endif

ifneq ($(SORKEN), 0)
else
 # assume EXPANSE
 REPORT = -fopt-info-vec-omp
endif


# set vec=1 if you want to vectorize by hand
vec=1
ifeq ($(vec),1)
C++FLAGS += -DSSE_VEC
C++FLAGS += -msse -msse2
C++FLAGS += -ftree-vectorize -ftree-vectorizer-verbose=2 -march=native -DSSE_VEC
endif

app:		apf

OBJECTS = apf.o solve.o Plotting.o cmdLine.o Report.o utils.o helper.o Timer.o

apf:	        $(OBJECTS) 
		$(C++LINK) $(LDFLAGS) -o $@ $(OBJECTS)  $(LDLIBS)

tgz:
	-tar cvfz $(THE_ARCHIVE).tgz Makefile *.cpp *.h
	-chmod 640 $(THE_ARCHIVE).tgz

.PHONY: clean
clean:	
	$(RM) *.o apf;
	$(RM) core;
