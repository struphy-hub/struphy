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

M3   := hylife/geometry/mappings_3d
MF3  := hylife/geometry/mappings_3d_fast
PB3  := hylife/geometry/pullback_3d
PF3  := hylife/geometry/pushforward_3d


KM2  := hylife/utilitis_FEEC/basics/kernels_2d
KM3  := hylife/utilitis_FEEC/basics/kernels_3d

DER  := hylife/utilitis_FEEC/derivatives/kernels_derivatives

LAC  := hylife/linear_algebra/core
LAT  := hylife/linear_algebra/kernels_tensor_product

KPG  := hylife/utilitis_FEEC/projectors/kernels_projectors_global
KPGM := hylife/utilitis_FEEC/projectors/kernels_projectors_global_mhd

PP   := hylife/utilitis_PIC/pusher
PA   := hylife/utilitis_PIC/accumulation_kernels
PS   := hylife/utilitis_PIC/sampling

IBK   := hylife/gvec_to_python/hylife/utilities_FEEC/bsplines_kernels
IBEV1 := hylife/gvec_to_python/hylife/utilities_FEEC/basics/spline_evaluation_1d

SOURCES := $(BK).py $(BEV1).py $(BEV2).py $(BEV3).py $(M3).py $(MF3).py $(PB3).py $(PF3).py $(KM2).py $(KM3).py $(DER).py $(LAC).py $(LAT).py $(KPG).py $(KPGM).py $(PP).py $(PA).py $(PS).py $(IBK).py $(IBEV1).py


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
    
$(M3)$(SO_EXT) : $(M3).py $(BEV2)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(MF3)$(SO_EXT) : $(MF3).py $(BK)$(SO_EXT) $(BEV2)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(PB3)$(SO_EXT) : $(PB3).py $(M3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(PF3)$(SO_EXT) : $(PF3).py $(M3)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(KM2)$(SO_EXT) : $(KM2).py
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)
    
$(KM3)$(SO_EXT) : $(KM3).py
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)
    
$(DER)$(SO_EXT) : $(DER).py
	pyccel $< $(FLAGS)
    
$(LAC)$(SO_EXT) : $(LAC).py
	pyccel $< $(FLAGS)
    
$(LAT)$(SO_EXT) : $(LAT).py
	pyccel $< $(FLAGS)
    
$(KPG)$(SO_EXT) : $(KPG).py
	pyccel $< $(FLAGS)
    
$(KPGM)$(SO_EXT) : $(KPGM).py
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)

$(PP)$(SO_EXT) : $(PP).py $(MF3)$(SO_EXT) $(LAC)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PA)$(SO_EXT) : $(PA).py $(MF3)$(SO_EXT) $(LAC)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PS)$(SO_EXT) : $(PS).py $(LAC)$(SO_EXT) $(MF3)$(SO_EXT) $(BK)$(SO_EXT) $(BEV2)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $< $(FLAGS)

$(IBK)$(SO_EXT) : $(IBK).py
	pyccel $< $(FLAGS)

$(IBEV1)$(SO_EXT) : $(IBEV1).py $(IBK)$(SO_EXT)
	pyccel $< $(FLAGS)

#--------------------------------------
# CLEAN UP
#--------------------------------------

.PHONY: clean
clean:
	rm -rf $(OUTPUTS)
	rm -rf $(all_sim)/$(run_dir)/input_run/__pyccel__ $(all_sim)/$(run_dir)/input_run/__pycache__
	rm -rf hylife/__pyccel__ hylife/__pycache__
	rm -rf hylife/diagnostics/__pyccel__ hylife/diagnostics/__pycache__
	rm -rf hylife/dispersion_relations/__pyccel__ hylife/dispersion_relations/__pycache__
	rm -rf hylife/geometry/__pyccel__ hylife/geometry/__pycache__
	rm -rf hylife/linear_algebra/__pyccel__ hylife/linear_algebra/__pycache__
	rm -rf hylife/utilitis_FEEC/__pyccel__ hylife/utilitis_FEEC/__pycache__
	rm -rf hylife/utilitis_FEEC/basics/__pyccel__ hylife/utilitis_FEEC/basics/__pycache__
	rm -rf hylife/utilitis_FEEC/derivatives/__pyccel__ hylife/utilitis_FEEC/derivatives/__pycache__
	rm -rf hylife/utilitis_FEEC/projectors/__pyccel__ hylife/utilitis_FEEC/projectors/__pycache__
	rm -rf hylife/utilitis_FEEC/control_variates/__pyccel__ hylife/utilitis_FEEC/control_variates/__pycache__
	rm -rf hylife/utilitis_PIC/__pyccel__ hylife/utilitis_PIC/__pycache__
	rm -rf hylife/gvec_to_python/hylife/utilities_FEEC/__pyccel__ hylife/gvec_to_python/hylife/utilities_FEEC/__pycache__
	rm -rf hylife/gvec_to_python/hylife/utilities_FEEC/basics/__pyccel__ hylife/gvec_to_python/hylife/utilities_FEEC/basics/__pycache__
