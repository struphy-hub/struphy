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

BK   := hylife/utilitis_FEEC/bsplines_kernels
BEV1 := hylife/utilitis_FEEC/basics/spline_evaluation_1d
BEV2 := hylife/utilitis_FEEC/basics/spline_evaluation_2d
BEV3 := hylife/utilitis_FEEC/basics/spline_evaluation_3d

M2   := hylife/geometry/mappings_2d
M3   := hylife/geometry/mappings_3d
MF3  := hylife/geometry/mappings_3d_fast
PB3  := hylife/geometry/pullback_3d
PF3  := hylife/geometry/pushforward_3d

KM2  := hylife/utilitis_FEEC/basics/kernels_2d
KM3  := hylife/utilitis_FEEC/basics/kernels_3d

DER  := hylife/utilitis_FEEC/derivatives/kernels_derivatives

LAC  := hylife/linear_algebra/core
LAT  := hylife/linear_algebra/kernels_tensor_product

EQM  := $(all_sim)/$(run_dir)/input_run/equilibrium_MHD
EQP  := $(all_sim)/$(run_dir)/input_run/equilibrium_PIC
ICM  := $(all_sim)/$(run_dir)/input_run/initial_conditions_MHD
ICP  := $(all_sim)/$(run_dir)/input_run/initial_conditions_PIC

KCV  := hylife/utilitis_FEEC/control_variates/kernels_control_variate

KPV  := hylife/utilitis_FEEC/projectors/kernels_projectors_evaluation
KPL  := hylife/utilitis_FEEC/projectors/kernels_projectors_local
KPG  := hylife/utilitis_FEEC/projectors/kernels_projectors_global
KPLM := hylife/utilitis_FEEC/projectors/kernels_projectors_local_mhd
KPGM := hylife/utilitis_FEEC/projectors/kernels_projectors_global_mhd

PP   := hylife/utilitis_PIC/pusher
PA   := hylife/utilitis_PIC/accumulation_kernels
PS   := hylife/utilitis_PIC/sampling

SOURCES := $(BK).py $(BEV1).py $(BEV2).py $(BEV3).py $(M2).py $(M3).py $(MF3).py $(PB3).py $(PF3).py $(KM2).py $(KM3).py $(DER).py $(LAC).py $(LAT).py $(EQM).py $(EQP).py $(ICM).py $(ICP).py $(KCV).py $(KPV).py $(KPL).py $(KPG).py $(KPLM).py $(KPGM).py $(PP).py $(PA).py $(PS).py


OUTPUTS := $(SOURCES:.py=$(SO_EXT))

#--------------------------------------
# PYCCELIZE
#--------------------------------------

.PHONY: all
all: $(OUTPUTS)

$(BK)$(SO_EXT) : $(BK).py
	pyccel $< $(FLAGS)
    
$(BEV1)$(SO_EXT) : $(BEV1).py $(BK)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(BEV2)$(SO_EXT) : $(BEV2).py $(BK)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(BEV3)$(SO_EXT) : $(BEV3).py $(BK)$(SO_EXT)
	pyccel $< $(FLAGS)

$(M2)$(SO_EXT) : $(M2).py $(BEV2)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(M3)$(SO_EXT) : $(M3).py $(BEV3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(MF3)$(SO_EXT) : $(MF3).py $(BK)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(PB3)$(SO_EXT) : $(PB3).py $(M3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(PF3)$(SO_EXT) : $(PF3).py $(M3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(KM2)$(SO_EXT) : $(KM2).py $(M2)$(SO_EXT)
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)
    
$(KM3)$(SO_EXT) : $(KM3).py $(M3)$(SO_EXT)
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)
    
$(DER)$(SO_EXT) : $(DER).py
	pyccel $< $(FLAGS)
    
$(LAC)$(SO_EXT) : $(LAC).py
	pyccel $< $(FLAGS)
    
$(LAT)$(SO_EXT) : $(LAT).py
	pyccel $< $(FLAGS)
    
$(EQM)$(SO_EXT) : $(EQM).py $(M3)$(SO_EXT) $(PB3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(EQP)$(SO_EXT) : $(EQP).py $(M3)$(SO_EXT) $(PB3)$(SO_EXT)
	pyccel $< $(FLAGS)

$(ICM)$(SO_EXT) : $(ICM).py $(M3)$(SO_EXT) $(PB3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(ICP)$(SO_EXT) : $(ICP).py $(M3)$(SO_EXT) $(PB3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(KCV)$(SO_EXT) : $(KCV).py $(M3)$(SO_EXT) $(EQP)$(SO_EXT)
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)

$(KPV)$(SO_EXT) : $(KPV).py $(M3)$(SO_EXT) $(EQM)$(SO_EXT) $(ICM)$(SO_EXT)
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)

$(KPL)$(SO_EXT) : $(KPL).py
	pyccel $< $(FLAGS)
    
$(KPG)$(SO_EXT) : $(KPG).py
	pyccel $< $(FLAGS)
        
$(KPLM)$(SO_EXT) : $(KPLM).py
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)
    
$(KPGM)$(SO_EXT) : $(KPGM).py
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)

$(PP)$(SO_EXT) : $(PP).py $(MF3)$(SO_EXT) $(LAC)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PA)$(SO_EXT) : $(PA).py $(MF3)$(SO_EXT) $(LAC)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PS)$(SO_EXT) : $(PS).py $(EQP)$(SO_EXT) $(ICP)$(SO_EXT)
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
	rm -rf hylife/utilitis_FEEC/derivatives/__pyccel__ hylife/utilitis_FEEC/derivatives/__pycache__
	rm -rf hylife/utilitis_FEEC/projectors/__pyccel__ hylife/utilitis_FEEC/projectors/__pycache__
	rm -rf hylife/utilitis_FEEC/control_variates/__pyccel__ hylife/utilitis_FEEC/control_variates/__pycache__
	rm -rf hylife/utilitis_PIC/__pyccel__ hylife/utilitis_PIC/__pycache__