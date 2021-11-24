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

BK   := struphy/feec/bsplines_kernels
BEV1 := struphy/feec/basics/spline_evaluation_1d
BEV2 := struphy/feec/basics/spline_evaluation_2d
BEV3 := struphy/feec/basics/spline_evaluation_3d

M3   := struphy/geometry/mappings_3d
MF3  := struphy/geometry/mappings_3d_fast
PB3  := struphy/geometry/pullback_3d
PF3  := struphy/geometry/pushforward_3d
TR3  := struphy/geometry/transform_3d

KM2  := struphy/feec/basics/kernels_2d
KM3  := struphy/feec/basics/kernels_3d

DER  := struphy/feec/derivatives/kernels_derivatives

LAC  := struphy/linear_algebra/core
LAT  := struphy/linear_algebra/kernels_tensor_product

KPG  := struphy/feec/projectors/pro_global/kernels_projectors_global
KPGM := struphy/feec/projectors/pro_global/kernels_projectors_global_mhd

PPP  := struphy/pic/pusher_pos
PV2  := struphy/pic/pusher_vel_2d
PV3  := struphy/pic/pusher_vel_3d
PA2  := struphy/pic/cc_lin_mhd_6d/accumulation_kernels_2d
PA3  := struphy/pic/cc_lin_mhd_6d/accumulation_kernels_3d
PS   := struphy/pic/sampling

SOURCES := $(BK).py $(BEV1).py $(BEV2).py $(BEV3).py $(M3).py $(MF3).py $(PB3).py $(PF3).py $(TR3).py $(KM2).py $(KM3).py $(DER).py $(LAC).py $(LAT).py $(KPG).py $(KPGM).py $(PPP).py $(PV2).py $(PV3).py $(PA2).py $(PA3).py $(PS).py


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
    
$(PV2)$(SO_EXT) : $(PV2).py $(LAC)$(SO_EXT) $(MF3)$(SO_EXT) $(BK)$(SO_EXT) $(BEV2)$(SO_EXT)
	pyccel $(FLAGS_openmp_pic) $< $(FLAGS)

$(PV3)$(SO_EXT) : $(PV3).py $(LAC)$(SO_EXT) $(MF3)$(SO_EXT) $(BK)$(SO_EXT) $(BEV3)$(SO_EXT)
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
	rm -rf struphy/__pyccel__ struphy/__pycache__
	rm -rf struphy/diagnostics/__pyccel__ struphy/diagnostics/__pycache__
	rm -rf struphy/dispersion_relations/__pyccel__ struphy/dispersion_relations/__pycache__
	rm -rf struphy/geometry/__pyccel__ struphy/geometry/__pycache__
	rm -rf struphy/linear_algebra/__pyccel__ struphy/linear_algebra/__pycache__
	rm -rf struphy/feec/__pyccel__ struphy/feec/__pycache__
	rm -rf struphy/feec/basics/__pyccel__ struphy/feec/basics/__pycache__
	rm -rf struphy/feec/derivatives/__pyccel__ struphy/feec/derivatives/__pycache__
	rm -rf struphy/feec/projectors/__pyccel__ struphy/feec/projectors/__pycache__
	rm -rf struphy/feec/projectors/pro_global/__pyccel__ struphy/feec/projectors/pro_global/__pycache__
	rm -rf struphy/feec/projectors/pro_local/__pyccel__ struphy/feec/projectors/pro_local/__pycache__
	rm -rf struphy/pic/__pyccel__ struphy/pic/__pycache__
	rm -rf struphy/pic/cc_lin_mhd_6d/__pyccel__ struphy/pic/cc_lin_mhd_6d/__pycache__