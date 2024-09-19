#--------------------------------------
# CONFIGURATION
#--------------------------------------

PYTHON  := python3
SO_EXT  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
LIBDIR  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
psydac_path := $(shell $(PYTHON) -c "import psydac as _; print(_.__path__[0])")
struphy_path := $(shell $(PYTHON) -c "import struphy as _; print(_.__path__[0])")

# Arguments to this script are: 
STRUPHY_SOURCES := $(sources)
FLAGS := --libdir $(LIBDIR) $(flags) 
FLAGS_openmp_pic := $(flags_openmp_pic)
FLAGS_openmp_mhd := $(flags_openmp_mhd)

#--------------------------------------
# SOURCE FILES 
#--------------------------------------

# PSY0
PSY0  := $(psydac_path)/core/arrays
# Doesn't exist devel psydac
# Copied the file to the struphy-2024 branch

# PSY1
# PSY1  := $(psydac_path)/core/kernels
# Found in psydac/core/field_evaluation_kernels
PSY1  := $(psydac_path)/core/field_evaluation_kernels

# PSY2
# PSY2  := $(psydac_path)/core/bsplines_pyccel
# Found in psydac/core/bsplines_kernels
PSY2  := $(psydac_path)/core/bsplines_kernels

# PSY3
# PSY3  := $(psydac_path)/linalg/kernels
# Found in psydac/linalg/kernels/stencil2coo_kernels.py
PSY3  := $(psydac_path)/linalg/kernels/stencil2coo_kernels

# PSY4
PSY4  := $(psydac_path)/feec/dof_kernels
# Found in psydac/feec/global_projectors.py
# However, this file is not pyccelizable and the kernels are not separated out.
# This is handled in stefan-psydac (https://github.com/pyccel/psydac/commit/2507c3c701596b1a40ea9357baad3fb88b09b636)
# For now, I added the dof_kernels to the struphy-2024 branch


# Old sources
# SOURCES := $(PSY0).py $(PSY1).py $(PSY2).py $(PSY3).py $(PSY4).py $(STRUPHY_SOURCES)

# New sources:
SOURCES := $(PSY0).py $(PSY1).py $(PSY2).py $(PSY3).py $(PSY4).py $(STRUPHY_SOURCES)

OUTPUTS := $(SOURCES:.py=$(SO_EXT))

#--------------------------------------
# PYCCELIZE
#--------------------------------------

.PHONY: all
all: $(OUTPUTS)

.SECONDEXPANSION:
%$(SO_EXT) : %.py $$(shell $$(PYTHON) $$(struphy_path)/dependencies.py $$@)
	@echo "Building $@"
	@echo "from dependencies:"
	@for dep in $^ ; do \
		echo $$dep ; \
    done
	pyccel $(FLAGS)$(FLAGS_openmp_pic)$(FLAGS_openmp_mhd) $<
	@echo ""

#--------------------------------------
# CLEAN UP
#--------------------------------------

.PHONY: clean
clean:
	rm -rf $(OUTPUTS)

	rm -rf $(psydac_path)/__pyccel__ $(psydac_path)/__pycache__
	rm -rf $(psydac_path)/core/__pyccel__ $(psydac_path)/core/__pycache__ $(psydac_path)/core/.lock_acquisition.lock
	rm -rf $(psydac_path)/linalg/__pyccel__ $(psydac_path)/linalg/__pycache__ $(psydac_path)/linalg/.lock_acquisition.lock
	rm -rf $(psydac_path)/feec/__pyccel__ $(psydac_path)/feec/__pycache__ $(psydac_path)/feec/.lock_acquisition.lock
    
	find $(struphy_path)/ -type d -name '__pyccel__' -prune -exec rm -rf {} \;
	find $(struphy_path)/ -type d -name '__pycache__' -prune -exec rm -rf {} \;
	find $(struphy_path)/ -type f -name '*.lock' -delete