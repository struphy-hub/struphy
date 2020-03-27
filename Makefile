#--------------------------------------
# CONFIGURATION
#--------------------------------------

PYTHON := python3
SO_EXT := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
FLAGS  :=

#--------------------------------------
# SOURCE FILES
#--------------------------------------

MA := hylife/geometry/mappings_analytical
EQ := hylife/geometry/equilibrium
IC := hylife/simulation/initial_conditions
KP := hylife/utilitis_FEEC/kernels_projectors_local_ini

SOURCES := $(MA).py $(EQ).py $(IC).py $(KP).py
OUTPUTS := $(SOURCES:.py=$(SO_EXT))

#--------------------------------------
# PYCCELIZE
#--------------------------------------

.PHONY: all
all: $(OUTPUTS)

$(MA)$(SO_EXT) : $(MA).py
	pyccel $< $(FLAGS)

$(EQ)$(SO_EXT) : $(EQ).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(IC)$(SO_EXT) : $(IC).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(KP)$(SO_EXT) : $(KP).py $(IC)$(SO_EXT) $(MA)$(SO_EXT) $(EQ)$(SO_EXT)
	pyccel $< $(FLAGS)

#--------------------------------------
# CLEAN UP
#--------------------------------------

.PHONY: clean
clean:
	rm -rf $(OUTPUTS)
	rm -rf hylife/__pycache__
	rm -rf hylife/geometry/__pyccel__ hylife/geometry/__pycache__
	rm -rf hylife/simulation/__pyccel__ hylife/simulation/__pycache__
	rm -rf hylife/utilities_FEEC/__pyccel__ hylife/utilitis_FEEC/__pycache__
