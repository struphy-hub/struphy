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
path_lib=$(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("platlib"))')/struphy/

a = $(shell echo ${path_lib})
$(info $(a))


BK   := ${path_lib}feec/bsplines_kernels
BEV1 := ${path_lib}feec/basics/spline_evaluation_1d
BEV2 := ${path_lib}feec/basics/spline_evaluation_2d
BEV3 := ${path_lib}feec/basics/spline_evaluation_3d

M3   := ${path_lib}geometry/mappings_3d
MF3  := ${path_lib}geometry/mappings_3d_fast
PB3  := ${path_lib}geometry/pullback_3d
PF3  := ${path_lib}geometry/pushforward_3d
TR3  := ${path_lib}geometry/transform_3d

KM2  := ${path_lib}feec/basics/kernels_2d
KM3  := ${path_lib}feec/basics/kernels_3d

DER  := ${path_lib}feec/derivatives/kernels_derivatives

LAC  := ${path_lib}linear_algebra/core
LAT  := ${path_lib}linear_algebra/kernels_tensor_product

KPG  := ${path_lib}feec/projectors/pro_global/kernels_projectors_global
KPGM := ${path_lib}feec/projectors/pro_global/kernels_projectors_global_mhd

PPP  := ${path_lib}pic/pusher_pos
PPV  := ${path_lib}pic/pusher_vel
PA2  := ${path_lib}pic/accumulation_kernels_2d
PA3  := ${path_lib}pic/accumulation_kernels_3d
PS   := ${path_lib}pic/sampling

SOURCES := $(BK).py $(BEV1).py $(BEV2).py $(BEV3).py $(M3).py $(MF3).py $(PB3).py $(PF3).py $(TR3).py $(KM2).py $(KM3).py $(DER).py $(LAC).py $(LAT).py $(KPG).py $(KPGM).py $(PPP).py $(PPV).py $(PA2).py $(PA3).py $(PS).py


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

$(TR3)$(SO_EXT) : $(TR3).py $(M3)$(SO_EXT)
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

$(PPP)$(SO_EXT) : $(PPP).py $(LAC)$(SO_EXT) $(M3)$(SO_EXT) $(MF3)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)
    
$(PPV)$(SO_EXT) : $(PPV).py $(LAC)$(SO_EXT) $(MF3)$(SO_EXT) $(BK)$(SO_EXT) $(BEV2)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PA2)$(SO_EXT) : $(PA2).py $(MF3)$(SO_EXT) $(LAC)$(SO_EXT) $(BK)$(SO_EXT) $(BEV2)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PA3)$(SO_EXT) : $(PA3).py $(MF3)$(SO_EXT) $(LAC)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PS)$(SO_EXT) : $(PS).py $(LAC)$(SO_EXT) $(MF3)$(SO_EXT) $(BK)$(SO_EXT) $(BEV2)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $< $(FLAGS)

#--------------------------------------
# CLEAN UP
#--------------------------------------

.PHONY: clean
clean:
	rm -rf $(OUTPUTS)
	rm -rf ${path_lib}__pyccel__ ${path_lib}__pycache__
	rm -rf ${path_lib}diagnostics/__pyccel__ ${path_lib}diagnostics/__pycache__
	rm -rf ${path_lib}dispersion_relations/__pyccel__ ${path_lib}dispersion_relations/__pycache__
	rm -rf ${path_lib}geometry/__pyccel__ ${path_lib}geometry/__pycache__
	rm -rf ${path_lib}linear_algebra/__pyccel__ ${path_lib}linear_algebra/__pycache__
	rm -rf ${path_lib}feec/__pyccel__ ${path_lib}feec/__pycache__
	rm -rf ${path_lib}feec/basics/__pyccel__ ${path_lib}feec/basics/__pycache__
	rm -rf ${path_lib}feec/derivatives/__pyccel__ ${path_lib}feec/derivatives/__pycache__
	rm -rf ${path_lib}feec/projectors/__pyccel__ ${path_lib}feec/projectors/__pycache__
	rm -rf ${path_lib}feec/projectors/pro_global/__pyccel__ ${path_lib}feec/projectors/pro_global/__pycache__
	rm -rf ${path_lib}feec/projectors/pro_local/__pyccel__ ${path_lib}feec/projectors/pro_local/__pycache__
	rm -rf ${path_lib}pic/__pyccel__ ${path_lib}pic/__pycache__
