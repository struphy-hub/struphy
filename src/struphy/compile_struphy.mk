#--------------------------------------
# CONFIGURATION
#--------------------------------------

PYTHON  := python3
SO_EXT  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
LIBDIR  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
psydac_path := $(shell $(PYTHON) -c "import psydac as _; print(_.__path__[0])")
struphy_path := $(shell $(PYTHON) -c "import struphy as _; print(_.__path__[0])")

# Arguments to this script are: 
# flags
# flags_openmp_pic
# flags_openmp_mhd
FLAGS            := --libdir $(LIBDIR) $(flags) 
FLAGS_openmp_pic := $(flags_openmp_pic)
FLAGS_openmp_mhd := $(flags_openmp_mhd)

#--------------------------------------
# SOURCE FILES PSYDAC
#--------------------------------------

PSY1  := $(psydac_path)/core/kernels
PSY2  := $(psydac_path)/core/bsplines_pyccel
PSY3  := $(psydac_path)/linalg/kernels
PSY4  := $(psydac_path)/feec/dof_kernels

#--------------------------------------
# SOURCE FILES STRUPHY
#--------------------------------------

# Linear algebra
LAC  := $(struphy_path)/linear_algebra/core
LAMV  := $(struphy_path)/linear_algebra/stencil_dot_kernels
LATR  := $(struphy_path)/linear_algebra/stencil_transpose_kernels

# Splines
BK   := $(struphy_path)/b_splines/bsplines_kernels
BKP  := $(struphy_path)/b_splines/bsplines_kernels_particles
BEV1 := $(struphy_path)/b_splines/bspline_evaluation_1d
BEV2 := $(struphy_path)/b_splines/bspline_evaluation_2d
BEV3 := $(struphy_path)/b_splines/bspline_evaluation_3d

# Mapping
MAFA := $(struphy_path)/geometry/mappings_fast
MEVA := $(struphy_path)/geometry/map_eval
TR3  := $(struphy_path)/geometry/transform
MK   := $(struphy_path)/geometry/kernels

# Kinetic background
MOMK := $(struphy_path)/kinetic_background/moments_kernels
F0K  := $(struphy_path)/kinetic_background/f0_kernels
BEVA := $(struphy_path)/kinetic_background/background_eval

# FEM kernels
PLP  := $(struphy_path)/feec/basis_projection_kernels
PLM  := $(struphy_path)/feec/mass_kernels
PLMH := $(struphy_path)/feec/utilities_kernels
BTS  := $(struphy_path)/feec/banded_to_stencil_kernels

# PIC
UTL	 := $(struphy_path)/pic/utilities_kernels

FK   := $(struphy_path)/pic/accumulation/filler_kernels
MVF  := $(struphy_path)/pic/accumulation/mat_vec_filler
ACC  := $(struphy_path)/pic/accumulation/accum_kernels
ACC_GC  := $(struphy_path)/pic/accumulation/accum_kernels_gc

PUTL := $(struphy_path)/pic/pushing/pusher_utilities
PUSH := $(struphy_path)/pic/pushing/pusher_kernels
PUSH_GC := $(struphy_path)/pic/pushing/pusher_kernels_gc
EVAL_GC := $(struphy_path)/pic/pushing/eval_kernels_gc
PS   := $(struphy_path)/pic/sampling

# Eigenvalue solver
KM2  := $(struphy_path)/eigenvalue_solvers/kernels_2d
KM3  := $(struphy_path)/eigenvalue_solvers/kernels_3d

KPG  := $(struphy_path)/eigenvalue_solvers/kernels_projectors_global
KPGM := $(struphy_path)/eigenvalue_solvers/kernels_projectors_global_mhd

SOURCES := $(LAC).py $(LAMV).py $(LATR).py $(BK).py $(BKP).py $(BEV1).py $(BEV2).py $(BEV3).py $(MAFA).py $(MEVA).py $(TR3).py $(MK).py $(MOMK).py $(F0K).py $(BEVA).py $(PLP).py $(PLM).py $(PLMH).py $(BTS).py $(UTL).py $(FK).py $(MVF).py $(ACC).py $(ACC_GC).py $(PUTL).py $(PUSH).py $(PUSH_GC).py $(EVAL_GC).py $(PS).py $(KM2).py $(KM3).py $(KPG).py $(KPGM).py $(PSY1).py $(PSY2).py $(PSY3).py $(PSY4).py

OUTPUTS := $(SOURCES:.py=$(SO_EXT))


#--------------------------------------
# PYCCELIZE
#--------------------------------------

.PHONY: all
all: $(OUTPUTS)

#Psydac:
$(PSY1)$(SO_EXT) : $(PSY1).py
	pyccel $< $(FLAGS)

$(PSY2)$(SO_EXT) : $(PSY2).py
	pyccel $< $(FLAGS)
    
$(PSY3)$(SO_EXT) : $(PSY3).py
	pyccel $< $(FLAGS)
    
$(PSY4)$(SO_EXT) : $(PSY4).py
	pyccel $< $(FLAGS)

# Struphy:
$(LAC)$(SO_EXT) : $(LAC).py
	pyccel $< $(FLAGS)

$(LAMV)$(SO_EXT) : $(LAMV).py
	pyccel $< $(FLAGS)

$(LATR)$(SO_EXT) : $(LATR).py
	pyccel $< $(FLAGS)

$(BK)$(SO_EXT) : $(BK).py
	pyccel $< $(FLAGS)

$(BKP)$(SO_EXT) : $(BKP).py
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

$(TR3)$(SO_EXT) : $(TR3).py $(LAC)$(SO_EXT) $(MEVA)$(SO_EXT)
	pyccel $< $(FLAGS)
    
$(MK)$(SO_EXT) : $(MK).py
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

$(PLMH)$(SO_EXT) : $(PLMH).py $(MEVA)$(SO_EXT) $(BK)$(SO_EXT) $(FK)$(SO_EXT)
	pyccel $< $(FLAGS)

$(BTS)$(SO_EXT) : $(BTS).py
	pyccel $< $(FLAGS)

$(UTL)$(SO_EXT) : $(UTL).py $(BEV3)$(SO_EXT) $(BK)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(FK)$(SO_EXT) : $(FK).py $(BKP)$(SO_EXT)
	pyccel $< $(FLAGS)

$(MVF)$(SO_EXT) : $(MVF).py $(FK)$(SO_EXT)
	pyccel $< $(FLAGS)

$(ACC)$(SO_EXT) : $(ACC).py $(MEVA)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT) $(BEVA)$(SO_EXT) $(MVF)$(SO_EXT) $(LAC)$(SO_EXT)  
	pyccel $< $(FLAGS)

$(ACC_GC)$(SO_EXT) : $(ACC_GC).py $(MEVA)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT) $(BEVA)$(SO_EXT) $(MVF)$(SO_EXT) $(LAC)$(SO_EXT)  
	pyccel $< $(FLAGS)

$(PUTL)$(SO_EXT) : $(PUTL).py $(LAC)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PUSH)$(SO_EXT) : $(PUSH).py $(PUTL)$(SO_EXT) $(LAC)$(SO_EXT) $(MEVA)$(SO_EXT) $(BK)$(SO_EXT) $(BKP)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PUSH_GC)$(SO_EXT) : $(PUSH_GC).py $(PUTL)$(SO_EXT) $(LAC)$(SO_EXT) $(MEVA)$(SO_EXT) $(BK)$(SO_EXT) $(BKP)$(SO_EXT) $(BEV3)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(EVAL_GC)$(SO_EXT) : $(EVAL_GC).py $(PUTL)$(SO_EXT) $(LAC)$(SO_EXT) $(MEVA)$(SO_EXT) $(BK)$(SO_EXT) $(BKP)$(SO_EXT) $(BEV3)$(SO_EXT)
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

	rm -rf $(psydac_path)/__pyccel__ $(psydac_path)/__pycache__
	rm -rf $(psydac_path)/core/__pyccel__ $(psydac_path)/core/__pycache__ $(psydac_path)/core/.lock_acquisition.lock
	rm -rf $(psydac_path)/linalg/__pyccel__ $(psydac_path)/linalg/__pycache__ $(psydac_path)/linalg/.lock_acquisition.lock
	rm -rf $(psydac_path)/feec/__pyccel__ $(psydac_path)/feec/__pycache__ $(psydac_path)/feec/.lock_acquisition.lock
    
	find $(struphy_path)/ -type d -name '__pyccel__' -prune -exec rm -rf {} \;
	find $(struphy_path)/ -type d -name '__pycache__' -prune -exec rm -rf {} \;
	find $(struphy_path)/ -type f -name '*.lock' -delete