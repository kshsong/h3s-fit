# --- Variables ---
SOURCES = basis.f90 pipnn.f90
MODULE_NAME = PIPNN_PES
PYF_FILE = $(MODULE_NAME).pyf
ONLY_SUBROUTINES = only: evvdvdx pes_init :

# --- Auto-detect compiler ---
ifeq ($(shell which ifort 2>/dev/null),)
    FCOMPILER = --fcompiler=gnu95
    F90 = gfortran
else
    FCOMPILER = --fcompiler=intelem
    F90 = ifort
endif

# --- Auto-detect BLAS library (MKL, OpenBLAS, or system BLAS) ---
# Try MKL first
MKLROOT ?= $(shell python -c "import os; print(os.environ.get('MKLROOT', ''))" 2>/dev/null)
ifneq ($(MKLROOT),)
    ifeq ($(wildcard $(MKLROOT)/lib/intel64/libmkl_rt.so),)
        MKLROOT :=
    endif
endif

# If MKL not found, try common OpenBLAS paths
OPENBLAS_ROOT ?= $(shell pkg-config --variable=libdir openblas 2>/dev/null || echo "")
ifneq ($(OPENBLAS_ROOT),)
    BLAS_LIBS = -L$(OPENBLAS_ROOT) -lopenblas
else
    # Fallback: system BLAS/LAPACK
    BLAS_LIBS = -lblas -llapack
endif

# If MKL is available, use it (highest priority)
ifeq ($(MKLROOT),)
    # No MKL → use OpenBLAS or system BLAS
    $(info ⚠️ MKL not found. Using: $(BLAS_LIBS))
else
    # Use MKL
    MKL_LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -lm -ldl
    BLAS_LIBS = $(MKL_LIBS)
    $(info ✅ Using MKL from $(MKLROOT))
endif

# Always add -fPIC for shared library
F90FLAGS = --f90flags="-fPIC -O3"

# --- Targets ---

all: $(PYF_FILE)
	@echo ">>> Compiling Fortran sources using the signature file..."
	f2py -c $(PYF_FILE) $(SOURCES) \
		-m $(MODULE_NAME) \
		$(FCOMPILER) \
		$(F90FLAGS) \
		$(BLAS_LIBS)

signature:
	@echo ">>> Generating signature file ($(PYF_FILE))..."
	f2py -h $(PYF_FILE) -m $(MODULE_NAME) $(SOURCES) $(ONLY_SUBROUTINES)
	@echo "---"

clean:
	@echo ">>> Cleaning up..."
	rm -f $(MODULE_NAME).cpython-*.so $(PYF_FILE) *.mod *.o
	rm -rf __pycache__

.PHONY: all signature clean
