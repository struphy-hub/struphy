#--------------------------------------
# CONFIGURATION
#--------------------------------------

PYTHON  := python3
SO_EXT  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
FLAGS   := 

#--------------------------------------
# SOURCE FILES
#--------------------------------------

MA  := hylife/geometry/mappings_analytical
EQM := hylife/simulation/equilibrium_MHD
EQP := hylife/simulation/equilibrium_PIC
ICM := hylife/simulation/initial_conditions_MHD
KCV := hylife/utilitis_FEEC/kernels_control_variate
KM  := hylife/utilitis_FEEC/kernels_mass
KPL := hylife/utilitis_FEEC/kernels_projectors_local
KPI := hylife/utilitis_FEEC/kernels_projectors_local_ini
KPM := hylife/utilitis_FEEC/kernels_projectors_local_mhd
LA  := hylife/linear_algebra/core
PF  := hylife/utilitis_PIC_NEW/STRUPHY_fields
PP  := hylife/utilitis_PIC_NEW/STRUPHY_pusher
PA  := hylife/utilitis_PIC_NEW/STRUPHY_accumulation_kernels
PS  := hylife/utilitis_PIC_NEW/STRUPHY_sampling

SOURCES := $(MA).py $(EQM).py $(EQP).py $(ICM).py $(KCV).py $(KM).py $(KPL).py $(KPI).py $(KPM).py $(LA).py $(PF).py $(PP).py $(PA).py $(PS).py
OUTPUTS := $(SOURCES:.py=$(SO_EXT))

#--------------------------------------
# PYCCELIZE
#--------------------------------------

.PHONY: all
all: $(OUTPUTS)

$(MA)$(SO_EXT) : $(MA).py
	pyccel $< $(FLAGS)

$(EQM)$(SO_EXT) : $(EQM).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(EQP)$(SO_EXT) : $(EQP).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(ICM)$(SO_EXT) : $(ICM).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(KCV)$(SO_EXT) : $(KCV).py $(MA)$(SO_EXT) $(EQM)$(SO_EXT) $(EQP)$(SO_EXT)
	pyccel $< $(FLAGS)

$(KM)$(SO_EXT)  : $(KM).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(KPL)$(SO_EXT) : $(KPL).py
	pyccel $< $(FLAGS)
    
$(KPI)$(SO_EXT) : $(KPI).py $(MA)$(SO_EXT) $(ICM)$(SO_EXT) $(EQM)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(KPM)$(SO_EXT) : $(KPM).py
	pyccel $< $(FLAGS)
    
$(LA)$(SO_EXT) : $(LA).py
	pyccel $< $(FLAGS)
    
$(PF)$(SO_EXT) : $(PF).py $(EQM)$(SO_EXT)
	pyccel --openmp $< $(FLAGS)

$(PP)$(SO_EXT) : $(PP).py $(MA)$(SO_EXT) $(LA)$(SO_EXT)
	pyccel --openmp $< $(FLAGS)

$(PA)$(SO_EXT) : $(PA).py $(MA)$(SO_EXT) $(LA)$(SO_EXT)
	pyccel --openmp $< $(FLAGS)

$(PS)$(SO_EXT) : $(PS).py
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
	rm -rf hylife/linear_algebra/__pyccel__ hylife/linear_algebra/__pycache__
	rm -rf hylife/utilities_FEEC/__pyccel__ hylife/utilitis_FEEC/__pycache__
	rm -rf hylife/utilities_PIC_NEW/__pyccel__ hylife/utilities_PIC_NEW/__pycache__