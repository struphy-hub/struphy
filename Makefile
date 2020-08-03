#--------------------------------------
# CONFIGURATION
#--------------------------------------

PYTHON  := python3
SO_EXT  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
FLAGS   := 

#--------------------------------------
# SOURCE FILES
#--------------------------------------


BK  := hylife/utilitis_FEEC/bsplines_kernels
BEV := hylife/utilitis_FEEC/basics/spline_evaluation_3d
MD  := hylife/geometry/mappings_discrete
MDF := hylife/geometry/mappings_discrete_fast
MA  := hylife/geometry/mappings_analytical
PBD := hylife/geometry/pull_back_discrete
PBA := hylife/geometry/pull_back_analytical
EQM := $(all_sim)/$(run_dir)/equilibrium_MHD
EQP := $(all_sim)/$(run_dir)/equilibrium_PIC
ICM := $(all_sim)/$(run_dir)/initial_conditions_MHD
ICP := $(all_sim)/$(run_dir)/initial_conditions_PIC
INT := hylife/interface_analytical
KCV := hylife/utilitis_FEEC/kernels_control_variate
KM  := hylife/utilitis_FEEC/basics/kernels_3d
KPL := hylife/utilitis_FEEC/projectors/kernels_projectors_local
KPI := hylife/utilitis_FEEC/projectors/kernels_projectors_local_eva_ana
KPM := hylife/utilitis_FEEC/projectors/kernels_projectors_local_mhd
LA  := hylife/linear_algebra/core
PF  := hylife/utilitis_PIC/fields
PP  := hylife/utilitis_PIC/pusher
PA  := hylife/utilitis_PIC/accumulation_kernels
PS  := hylife/utilitis_PIC/sampling

SOURCES := $(BK).py $(BEV).py $(MD).py $(MDF).py $(MA).py $(PBD).py $(PBA).py $(EQM).py $(EQP).py $(ICM).py $(ICP).py $(INT).py $(KCV).py $(KM).py $(KPL).py $(KPI).py $(KPM).py $(LA).py $(PF).py $(PP).py $(PA).py $(PS).py
OUTPUTS := $(SOURCES:.py=$(SO_EXT))

#--------------------------------------
# PYCCELIZE
#--------------------------------------

.PHONY: all
all: $(OUTPUTS)

$(BK)$(SO_EXT) : $(BK).py
	pyccel $< $(FLAGS)
    
$(BEV)$(SO_EXT) : $(BEV).py $(BK)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(MD)$(SO_EXT) : $(MD).py $(BEV)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(MDF)$(SO_EXT) : $(MDF).py $(BEV)$(SO_EXT) $(BK)$(SO_EXT)
	pyccel $< $(FLAGS)

$(MA)$(SO_EXT) : $(MA).py
	pyccel $< $(FLAGS)
    
$(PBD)$(SO_EXT) : $(PBD).py $(MD)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(PBA)$(SO_EXT) : $(PBA).py $(MA)$(SO_EXT)
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

$(KM)$(SO_EXT) : $(KM).py
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
	rm -rf $(all_sim)/$(run_dir)/__pyccel__ $(all_sim)/$(run_dir)/__pycache__
	rm -rf hylife/__pyccel__ hylife/__pycache__
	rm -rf hylife/geometry/__pyccel__ hylife/geometry/__pycache__
	rm -rf hylife/linear_algebra/__pyccel__ hylife/linear_algebra/__pycache__
	rm -rf hylife/utilitis_FEEC/__pyccel__ hylife/utilitis_FEEC/__pycache__
	rm -rf hylife/utilitis_FEEC/basics/__pyccel__ hylife/utilitis_FEEC/basics/__pycache__
	rm -rf hylife/utilitis_FEEC/projectors/__pyccel__ hylife/utilitis_FEEC/projectors/__pycache__
	rm -rf hylife/utilitis_PIC/__pyccel__ hylife/utilitis_PIC/__pycache__
