#--------------------------------------
# CONFIGURATION
#--------------------------------------

PYTHON  := python3
SO_EXT  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
LIBDIR  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
struphy_path := $(shell $(PYTHON) -c "import struphy; print(struphy.__path__[0])")

# Arguments to this script are: 
STRUPHY_SOURCES := $(sources)
FLAGS := --libdir $(LIBDIR) $(flags) #--debug
FLAGS_openmp_pic := $(flags_openmp_pic)
FLAGS_openmp_mhd := $(flags_openmp_mhd)

#--------------------------------------
# SOURCE FILES 
#--------------------------------------

SOURCES := $(STRUPHY_SOURCES)

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
	pyccel $(FLAGS) $(FLAGS_openmp_pic) $(FLAGS_openmp_mhd) $<
	@echo ""

#--------------------------------------
# CLEAN UP
#--------------------------------------
# find $(struphy_path)/ -type d -name '__pyccel__' -prune -exec rm -rf {} \;
# find $(struphy_path)/ -type d -name '__pycache__' -prune -exec rm -rf {} \;
# find $(struphy_path)/ -type f -name '*.lock' -delete

.PHONY: clean
clean:
	rm -rf $(OUTPUTS)
	find $(struphy_path)/ -type d \( -name '__pyccel__' -o -name '__pycache__' \) -exec rm -rf {} +
	find $(struphy_path)/ -type f \( -name '*.lock' -o -name '*.so' -o -name '*.o' -o -name '*.mod' \) -delete