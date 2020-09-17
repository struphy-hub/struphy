#--------------------------------------
# CONFIGURATION
#--------------------------------------

PYTHON  := python3
SO_EXT  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

FLAGS            := 
FLAGS_openmp_mhd := $(flags_openmp_mhd)
FLAGS_openmp_pic := $(flags_openmp_pic)

#--------------------------------------
# SOURCE FILES
#--------------------------------------

BK  := hylife/utilitis_FEEC/bsplines_kernels
BEV := hylife/utilitis_FEEC/basics/spline_evaluation_3d
MA  := hylife/geometry/mappings_analytical
MD  := hylife/geometry/mappings_discrete
PBA := hylife/geometry/pull_back_analytical
PBD := hylife/geometry/pull_back_discrete
EQM := $(all_sim)/$(run_dir)/input_run/equilibrium_MHD
EQP := $(all_sim)/$(run_dir)/input_run/equilibrium_PIC
ICM := $(all_sim)/$(run_dir)/input_run/initial_conditions_MHD
ICP := $(all_sim)/$(run_dir)/input_run/initial_conditions_PIC
KCV := $(all_sim)/$(run_dir)/source_run/kernels_control_variate
KM  := hylife/utilitis_FEEC/basics/kernels_3d
KPL := hylife/utilitis_FEEC/projectors/kernels_projectors_local
KPV := $(all_sim)/$(run_dir)/source_run/kernels_projectors_local_eva
KPM := hylife/utilitis_FEEC/projectors/kernels_projectors_local_mhd
LA  := hylife/linear_algebra/core
PP  := $(all_sim)/$(run_dir)/source_run/pusher
PA  := $(all_sim)/$(run_dir)/source_run/accumulation_kernels
PS  := $(all_sim)/$(run_dir)/source_run/sampling

SOURCES := $(BK).py $(BEV).py $(MA).py $(MD).py $(PBA).py $(PBD).py $(EQM).py $(EQP).py $(ICM).py $(ICP).py $(KCV).py $(KM).py $(KPL).py $(KPV).py $(KPM).py $(LA).py $(PP).py $(PA).py $(PS).py

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

$(MA)$(SO_EXT) : $(MA).py
	pyccel $< $(FLAGS)
    
$(MD)$(SO_EXT) : $(MD).py $(BEV)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(PBA)$(SO_EXT) : $(PBA).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(PBD)$(SO_EXT) : $(PBD).py $(MD)$(SO_EXT)
	pyccel $< $(FLAGS)

$(EQM)$(SO_EXT) : $(EQM).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(EQP)$(SO_EXT) : $(EQP).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(ICM)$(SO_EXT) : $(ICM).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(ICP)$(SO_EXT) : $(ICP).py $(MA)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(KCV)$(SO_EXT) : $(KCV).py $(MA)$(SO_EXT) $(MD)$(SO_EXT) $(EQP)$(SO_EXT) $(EQM)$(SO_EXT)
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)

$(KM)$(SO_EXT) : $(KM).py $(MA)$(SO_EXT) $(MD)$(SO_EXT)
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)

$(KPL)$(SO_EXT) : $(KPL).py
	pyccel $< $(FLAGS)
    
$(KPV)$(SO_EXT) : $(KPV).py $(MA)$(SO_EXT) $(MD)$(SO_EXT) $(EQM)$(SO_EXT) $(ICM)$(SO_EXT)
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)
    
$(KPM)$(SO_EXT) : $(KPM).py
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)
    
$(LA)$(SO_EXT) : $(LA).py
	pyccel $< $(FLAGS)

$(PP)$(SO_EXT) : $(PP).py $(MA)$(SO_EXT) $(EQM)$(SO_EXT) $(LA)$(SO_EXT) $(BK)$(SO_EXT) $(BEV)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PA)$(SO_EXT) : $(PA).py $(MA)$(SO_EXT) $(EQM)$(SO_EXT) $(LA)$(SO_EXT) $(BK)$(SO_EXT) $(BEV)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PS)$(SO_EXT) : $(PS).py $(MA)$(SO_EXT) $(EQP)$(SO_EXT) $(ICP)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)


#--------------------------------------
# CLEAN UP
#--------------------------------------

.PHONY: clean
clean:
	rm -rf $(OUTPUTS)
	rm -rf $(all_sim)/$(run_dir)/input_run/__pyccel__ $(all_sim)/$(run_dir)/input_run/__pycache__
	rm -rf hylife/__pyccel__ hylife/__pycache__
	rm -rf hylife/geometry/__pyccel__ hylife/geometry/__pycache__
	rm -rf hylife/linear_algebra/__pyccel__ hylife/linear_algebra/__pycache__
	rm -rf hylife/utilitis_FEEC/__pyccel__ hylife/utilitis_FEEC/__pycache__
	rm -rf hylife/utilitis_FEEC/basics/__pyccel__ hylife/utilitis_FEEC/basics/__pycache__
	rm -rf hylife/utilitis_FEEC/projectors/__pyccel__ hylife/utilitis_FEEC/projectors/__pycache__
	rm -rf hylife/utilitis_FEEC/control_variates/__pyccel__ hylife/utilitis_FEEC/control_variates/__pycache__
	rm -rf hylife/utilitis_PIC/__pyccel__ hylife/utilitis_PIC/__pycache__