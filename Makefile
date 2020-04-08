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
EQM := simulation_06042020_2/equilibrium_MHD
EQP := simulation_06042020_2/equilibrium_PIC
ICM := simulation_06042020_2/initial_conditions_MHD
ICP := simulation_06042020_2/initial_conditions_PIC
INT := hylife/interface
KCV := hylife/utilitis_FEEC/kernels_control_variate
KM  := hylife/utilitis_FEEC/kernels_mass
KPL := hylife/utilitis_FEEC/kernels_projectors_local
KPI := hylife/utilitis_FEEC/kernels_projectors_local_ini
KPM := hylife/utilitis_FEEC/kernels_projectors_local_mhd
LA  := hylife/linear_algebra/core
PF  := hylife/utilitis_PIC_April2020/STRUPHY_fields
PP  := hylife/utilitis_PIC_April2020/STRUPHY_pusher
PA  := hylife/utilitis_PIC_April2020/STRUPHY_accumulation_kernels
PS  := hylife/utilitis_PIC_April2020/STRUPHY_sampling

SOURCES := $(MA).py $(EQM).py $(EQP).py $(ICM).py $(ICP).py $(INT).py $(KCV).py $(KM).py $(KPL).py $(KPI).py $(KPM).py $(LA).py $(PF).py $(PP).py $(PA).py $(PS).py
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
    
$(ICP)$(SO_EXT) : $(ICP).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(INT)$(SO_EXT) : $(INT).py $(EQM)$(SO_EXT) $(EQP)$(SO_EXT) $(ICM)$(SO_EXT) $(ICP)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(KCV)$(SO_EXT) : $(KCV).py $(MA)$(SO_EXT) $(INT)$(SO_EXT)
	pyccel $< $(FLAGS)

$(KM)$(SO_EXT)  : $(KM).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(KPL)$(SO_EXT) : $(KPL).py
	pyccel $< $(FLAGS)
    
$(KPI)$(SO_EXT) : $(KPI).py $(MA)$(SO_EXT) $(INT)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(KPM)$(SO_EXT) : $(KPM).py
	pyccel $< $(FLAGS)
    
$(LA)$(SO_EXT) : $(LA).py
	pyccel $< $(FLAGS)
    
$(PF)$(SO_EXT) : $(PF).py $(INT)$(SO_EXT)
	pyccel --openmp $< $(FLAGS)

$(PP)$(SO_EXT) : $(PP).py $(MA)$(SO_EXT) $(LA)$(SO_EXT)
	pyccel --openmp $< $(FLAGS)

$(PA)$(SO_EXT) : $(PA).py $(MA)$(SO_EXT) $(LA)$(SO_EXT)
	pyccel --openmp $< $(FLAGS)

$(PS)$(SO_EXT) : $(PS).py $(MA)$(SO_EXT) $(INT)$(SO_EXT)
	pyccel $< $(FLAGS)


#--------------------------------------
# CLEAN UP
#--------------------------------------

.PHONY: clean
clean:
	rm -rf $(OUTPUTS)
	rm -rf hylife/__pycache__ hylife/__pyccel__
	rm -rf hylife/geometry/__pyccel__ hylife/geometry/__pycache__
	rm -rf hylife/simulation/__pyccel__ hylife/simulation/__pycache__
	rm -rf hylife/linear_algebra/__pyccel__ hylife/linear_algebra/__pycache__
	rm -rf hylife/utilities_FEEC/__pyccel__ hylife/utilitis_FEEC/__pycache__
	rm -rf hylife/utilities_PIC_April2020/__pyccel__ hylife/utilities_PIC_April2020/__pycache__