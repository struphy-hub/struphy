#--------------------------------------
# CONFIGURATION
#--------------------------------------

PYTHON  := python3
SO_EXT  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

flags_openmp_pic=--openmp

FLAG_PATH        := $(flag_path)
FLAGS            := 
FLAGS_openmp_mhd := $(flags_openmp_mhd)
FLAGS_openmp_pic := $(flags_openmp_pic)

#--------------------------------------
# SOURCE FILES
#--------------------------------------
path_lib=$(FLAG_PATH)/

# Linear algebra
LAC  := ${path_lib}linear_algebra/core

# Splines
BK   := ${path_lib}feec/bsplines_kernels
BEV1 := ${path_lib}feec/basics/spline_evaluation_1d
BEV2 := ${path_lib}feec/basics/spline_evaluation_2d
BEV3 := ${path_lib}feec/basics/spline_evaluation_3d

# Mapping
MAFA := ${path_lib}geometry/mappings_fast
MEVA := ${path_lib}geometry/map_eval
PB3  := ${path_lib}geometry/pullback
PF3  := ${path_lib}geometry/pushforward
TR3  := ${path_lib}geometry/transform

# Kinetic background
MOMK := ${path_lib}kinetic_background/moments_kernels
F0K  := ${path_lib}kinetic_background/f0_kernels
BEVA := ${path_lib}kinetic_background/background_eval

# FEM kernels
PLP  := ${path_lib}psydac_api/mhd_ops_kernels_pure_psydac
PLM  := ${path_lib}psydac_api/mass_kernels_psydac
BTS  := ${path_lib}psydac_api/banded_to_stencil_kernels

# PIC
UTL	 := ${path_lib}pic/utilities_kernels

FK   := ${path_lib}pic/filler_kernels
MVF  := ${path_lib}pic/mat_vec_filler
ACC  := ${path_lib}pic/accum_kernels

PUTL := ${path_lib}pic/pusher_utilities
PUSH := ${path_lib}pic/pusher_kernels
PS   := ${path_lib}pic/sampling

# Legacy
KM2  := ${path_lib}feec/basics/kernels_2d
KM3  := ${path_lib}feec/basics/kernels_3d

KPG  := ${path_lib}feec/projectors/pro_global/kernels_projectors_global
KPGM := ${path_lib}feec/projectors/pro_global/kernels_projectors_global_mhd

SOURCES := $(LAC).py $(BK).py $(BEV1).py $(BEV2).py $(BEV3).py $(MAFA).py $(MEVA).py $(PB3).py $(PF3).py $(TR3).py $(MOMK).py $(F0K).py $(BEVA).py $(PLP).py $(PLM).py $(BTS).py $(UTL).py $(FK).py $(MVF).py $(ACC).py $(PUTL).py $(PUSH).py $(PS).py $(KM2).py $(KM3).py $(KPG).py $(KPGM).py 

OUTPUTS := $(SOURCES:.py=$(SO_EXT))


#--------------------------------------
# PYCCELIZE
#--------------------------------------

.PHONY: all
all: $(OUTPUTS)

$(LAC)$(SO_EXT) : $(LAC).py
	pyccel $< $(FLAGS)

$(BK)$(SO_EXT) : $(BK).py
	pyccel $< $(FLAGS)

$(BEV1)$(SO_EXT) : $(BEV1).py $(BK)$(SO_EXT)
	pyccel $< $(FLAGS)

$(BEV2)$(SO_EXT) : $(BEV2).py $(BK)$(SO_EXT)
	pyccel $< $(FLAGS)

$(BEV3)$(SO_EXT) : $(BEV3).py $(BK)$(SO_EXT)
	pyccel $< $(FLAGS)

$(MAFA)$(SO_EXT) : $(MAFA).py $(BK)$(SO_EXT) $(BEV2)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $< $(FLAGS)

$(MEVA)$(SO_EXT) : $(MEVA).py $(MAFA)$(SO_EXT) $(LAC)$(SO_EXT)
	pyccel $< $(FLAGS)

$(PB3)$(SO_EXT) : $(PB3).py $(LAC)$(SO_EXT) $(MEVA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(PF3)$(SO_EXT) : $(PF3).py $(LAC)$(SO_EXT) $(MEVA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(TR3)$(SO_EXT) : $(TR3).py $(LAC)$(SO_EXT) $(MEVA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(MOMK)$(SO_EXT) : $(MOMK).py
	pyccel $< $(FLAGS)

$(F0K)$(SO_EXT) : $(F0K).py $(MOMK)$(SO_EXT) 
	pyccel $< $(FLAGS)

$(BEVA)$(SO_EXT) : $(BEVA).py $(F0K)$(SO_EXT) 
	pyccel $< $(FLAGS)

$(PLP)$(SO_EXT) : $(PLP).py
	pyccel $< $(FLAGS)

$(PLM)$(SO_EXT) : $(PLM).py
	pyccel $< $(FLAGS)

$(BTS)$(SO_EXT) : $(BTS).py
	pyccel $< $(FLAGS)

$(UTL)$(SO_EXT) : $(UTL).py $(BEV3).py $(BK).py
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(FK)$(SO_EXT) : $(FK).py
	pyccel $< $(FLAGS)

$(MVF)$(SO_EXT) : $(MVF).py $(FK)$(SO_EXT)
	pyccel $< $(FLAGS)

$(ACC)$(SO_EXT) : $(ACC).py $(MEVA)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT) $(BEVA)$(SO_EXT) $(MVF)$(SO_EXT) $(LAC)$(SO_EXT)  
	pyccel $< $(FLAGS)

$(PUTL)$(SO_EXT) : $(PUTL).py $(LAC)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PUSH)$(SO_EXT) : $(PUSH).py $(PUTL).py $(LAC)$(SO_EXT) $(MEVA)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PS)$(SO_EXT) : $(PS).py $(LAC)$(SO_EXT) $(BK)$(SO_EXT) $(BEV2)$(SO_EXT) $(BEV3)$(SO_EXT) $(MEVA)$(SO_EXT)
	pyccel $< $(FLAGS)

$(KM2)$(SO_EXT) : $(KM2).py
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)

$(KM3)$(SO_EXT) : $(KM3).py
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)

$(KPG)$(SO_EXT) : $(KPG).py
	pyccel $< $(FLAGS)

$(KPGM)$(SO_EXT) : $(KPGM).py
	pyccel $(FLAGS_openmp_mhd) $< $(FLAGS)



#--------------------------------------
# CLEAN UP
#--------------------------------------

.PHONY: clean
clean:
	rm -rf $(OUTPUTS)
	rm -rf ${path_lib}__pyccel__ ${path_lib}__pycache__
	rm -rf ${path_lib}linear_algebra/__pyccel__ ${path_lib}linear_algebra/__pycache__ ${path_lib}linear_algebra/.lock_acquisition.lock
	rm -rf ${path_lib}feec/__pyccel__ ${path_lib}feec/__pycache__ ${path_lib}feec/.lock_acquisition.lock
	rm -rf ${path_lib}feec/basics/__pyccel__ ${path_lib}feec/basics/__pycache__ ${path_lib}feec/basics/.lock_acquisition.lock
	rm -rf ${path_lib}feec/projectors/__pyccel__ ${path_lib}feec/projectors/__pycache__ ${path_lib}feec/projectors/.lock_acquisition.lock
	rm -rf ${path_lib}feec/projectors/pro_global/__pyccel__ ${path_lib}feec/projectors/pro_global/__pycache__ ${path_lib}feec/projectors/pro_global/.lock_acquisition.lock
	rm -rf ${path_lib}feec/projectors/pro_local/__pyccel__ ${path_lib}feec/projectors/pro_local/__pycache__ ${path_lib}feec/projectors/pro_local/.lock_acquisition.lock
	rm -rf ${path_lib}geometry/__pyccel__ ${path_lib}geometry/__pycache__ ${path_lib}geometry/.lock_acquisition.lock
	rm -rf ${path_lib}kinetic_background/__pyccel__ ${path_lib}kinetic_background/__pycache__ ${path_lib}kinetic_background/.lock_acquisition.lock
	rm -rf ${path_lib}diagnostics/__pyccel__ ${path_lib}diagnostics/__pycache__ ${path_lib}diagnostics/.lock_acquisition.lock
	rm -rf ${path_lib}dispersion_relations/__pyccel__ ${path_lib}dispersion_relations/__pycache__ ${path_lib}dispersion_relations/.lock_acquisition.lock
	rm -rf ${path_lib}pic/__pyccel__ ${path_lib}pic/__pycache__     ${path_lib}pic/.lock_acquisition.lock
	rm -rf ${path_lib}psydac_api/__pyccel__ ${path_lib}psydac_api/__pycache__ ${path_lib}psydac_api/.lock_acquisition.lock