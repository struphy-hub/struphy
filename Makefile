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
LA  := hylife/linear_algebra/core
EQM := simulation_06042020_2/equilibrium_MHD
EQP := simulation_06042020_2/equilibrium_PIC
ICM := simulation_06042020_2/initial_conditions_MHD
ICP := simulation_06042020_2/initial_conditions_PIC
INT := hylife/interface
SRC_BASE := $(MA).py $(LA).py $(EQM).py $(EQP).py $(ICM).py $(ICP).py $(INT).py
OBJ_BASE := $(SRC_BASE:.py=$(SO_EXT))

KCV := hylife/utilitis_FEEC/kernels_control_variate
KM  := hylife/utilitis_FEEC/kernels_mass
KPL := hylife/utilitis_FEEC/kernels_projectors_local
KPI := hylife/utilitis_FEEC/kernels_projectors_local_ini
KPM := hylife/utilitis_FEEC/kernels_projectors_local_mhd
SRC_FEEC := $(KCV).py $(KM).py $(KPL).py $(KPI).py $(KPM).py 
OBJ_FEEC := $(SRC_FEEC:.py=$(SO_EXT))

PF  := hylife/utilitis_PIC_April2020/STRUPHY_fields
PP  := hylife/utilitis_PIC_April2020/STRUPHY_pusher
PA  := hylife/utilitis_PIC_April2020/STRUPHY_accumulation_kernels
PS  := hylife/utilitis_PIC_April2020/STRUPHY_sampling
SRC_PIC := $(PF).py $(PP).py $(PA).py $(PS).py
OBJ_PIC  := $(SRC_PIC:.py=$(SO_EXT))

KPG := hylife/utilitis_FEEC/kernels_projectors_global
KPGx := hylife/utilitis_FEEC/kernels_projectors_global_V2
SRC_PROJ := $(KPG).py $(KPGx).py
OBJ_PROJ := $(SRC_PROJ:.py=$(SO_EXT))

SOURCES := $(MA).py $(EQM).py $(EQP).py $(ICM).py $(ICP).py $(INT).py $(KCV).py $(KM).py $(KPL).py $(KPI).py $(KPM).py $(LA).py $(PF).py $(PP).py $(PA).py $(PS).py
OUTPUTS := $(SOURCES:.py=$(SO_EXT))

#--------------------------------------
# Main targets and general rules 
#--------------------------------------

.PHONY: all proj base feec pic
#all: proj base feec pic
all: proj

# only projectors
proj: $(OBJ_PROJ)
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PYCCELIZE PROJ DONE. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

base: $(OBJ_BASE)
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PYCCELIZE BASE DONE. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'


feec: base $(OBJ_FEEC)
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PYCCELIZE FEEC DONE. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

pic: base $(OBJ_PIC)
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PYCCELIZE PIC DONE.  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

#--------------------------------------
# dependencies and pyccelize 
#--------------------------------------

$(KPG)$(SO_EXT) : $(KPG).py
	pyccel $< $(FLAGS)

$(KPGx)$(SO_EXT) : $(KPGx).py
	pyccel $< $(FLAGS)

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

.PHONY: clean cleanproj cleanbase cleanfeec cleanpic
#clean: cleanproj cleanbase cleanfeec cleanpic
clean: cleanproj

veryclean:
	rm -rf hylife/*.so hylife/*/*.so
	rm -rf hylife/__pyc*__ hylife/*/__pyc*__
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VERYCLEAN DONE.  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

cleanproj:
	rm -rf $(OBJ_PROJ)
	rm -rf hylife/utilitis_FEEC/__pyc*__
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CLEAN PROJ DONE. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

cleanbase:
	rm -rf $(OBJ_BASE)
	rm -rf hylife/__pyc*__
	rm -rf hylife/geometry/__pyc*__
	rm -rf hylife/linear_algebra/__pyc*__
	rm -rf hylife/simulation_06042020_2/__pyc*__
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CLEAN BASE DONE. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

cleanfeec:
	rm -rf $(OBJ_FEEC)
	rm -rf hylife/utilitis_FEEC/__pyc*__
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CLEAN FEEC DONE. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

cleanpic:
	rm -rf $(OBJ_PIC)
	rm -rf hylife/utilitis_PIC_April2020/__pyc*__
	@echo '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CLEAN PIC DONE.  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
