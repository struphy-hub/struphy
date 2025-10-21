import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from mpi4py import MPI
import warnings

from struphy.io.options import Units
from struphy.models.variables import Variable
from struphy.pic.utilities import (LoadingParameters, 
                                   WeightsParameters, 
                                   BoundaryParameters,
                                   BinningPlot,
                                   )


class Species(metaclass=ABCMeta):
    """Single species of a StruphyModel."""

    @abstractmethod
    def __init__(self):
        self.init_variables()

    # set species attribute for each variable
    def init_variables(self):
        self._variables = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Variable):
                v._name = k
                v._species = self
                self._variables[k] = v

    @property
    def variables(self) -> dict:
        return self._variables

    @property
    def charge_number(self) -> int:
        """Charge number in units of elementary charge."""
        return self._charge_number

    @property
    def mass_number(self) -> int:
        """Mass number in units of proton mass."""
        return self._mass_number

    def set_phys_params(self, 
                        charge_number: int = 1, 
                        mass_number: int = 1,
                        alpha: float = None,
                        epsilon: float = None,
                        kappa: float = None,
                        ):
        """Set charge- and mass number. Set equation parameters (alpha, epsilon, ...) to override units."""
        self._charge_number = charge_number
        self._mass_number = mass_number
        self.alpha = alpha
        self.epsilon = epsilon
        self.kappa = kappa 
    
    class EquationParameters:
        """Normalization parameters of one species, appearing in scaled equations."""
    
        def __init__(self, 
                     species, 
                     units: Units = None, 
                     alpha: float = None,
                     epsilon: float = None,
                     kappa: float = None,
                     verbose: bool = False,
                     ):
            if units is None:
                units = Units()
                
            Z = species.charge_number
            A = species.mass_number
            
            con = ConstantsOfNature()

            # relevant frequencies
            om_p = xp.sqrt(units.n * (Z * con.e) ** 2 / (con.eps0 * A * con.mH))
            om_c = Z * con.e * units.B / (A * con.mH)
            
            # compute equation parameters
            if alpha is None:
                self.alpha = om_p / om_c
            else:
                self.alpha = alpha
                warnings.warn(f"Override equation parameter {self.alpha = }")
                
            if epsilon is None:
                self.epsilon = 1.0 / (om_c * units.t)
            else:
                self.epsilon = epsilon
                warnings.warn(f"Override equation parameter {self.epsilon = }")
                
            if kappa is None:
                self.kappa = om_p * units.t
            else:
                self.kappa = kappa
                if MPI.COMM_WORLD.Get_rank() == 0:
                    warnings.warn(f"Override equation parameter {self.kappa = }")

            if verbose and MPI.COMM_WORLD.Get_rank() == 0:
                print(f'\nSet normalization parameters for species {species.__class__.__name__}:')
                for key, val in self.__dict__.items():
                    print((key + ":").ljust(25), "{:4.3e}".format(val))
    
    @property
    def equation_params(self) -> EquationParameters:
        return self._equation_params

    def setup_equation_params(self, units: Units, verbose=False):
        """Set the following equation parameters:

        * alpha = plasma-frequenca / cyclotron frequency
        * epsilon = 1 / (cyclotron frequency * time unit)
        * kappa = plasma frequency * time unit
        """
        self._equation_params = self.EquationParameters(species=self, 
                                                        units=units, 
                                                        alpha=self.alpha,
                                                        epsilon=self.epsilon,
                                                        kappa=self.kappa,
                                                        verbose=verbose,)
        
    


class FieldSpecies(Species):
    """Species without mass and charge (so-called 'fields')."""


class FluidSpecies(Species):
    """Single fluid species in 3d configuration space."""


class ParticleSpecies(Species):
    """Single kinetic species in 3d + vdim phase space."""

    def set_markers(
        self,
        loading_params: LoadingParameters = None,
        weights_params: WeightsParameters = None,
        boundary_params: BoundaryParameters = None,
        bufsize: float = 1.0,
    ):
        """Set marker parameters for loading, weight calculation, kernel density reconstruction
        and boundary conditions.

        Parameters
        ----------
        loading_params : LoadingParameters

        weights_params : WeightsParameters

        boundary_params : BoundaryParameters

        bufsize : float
            Size of buffer (as multiple of total size, default=.25) in markers array."""

        # defaults
        if loading_params is None:
            loading_params = LoadingParameters()

        if weights_params is None:
            weights_params = WeightsParameters()
            
        if boundary_params is None:
            boundary_params = BoundaryParameters()
        
        self.loading_params = loading_params
        self.weights_params = weights_params
        self.boundary_params = boundary_params
        self.bufsize = bufsize

    def set_sorting_boxes(
        self,
        do_sort: bool = False,
        sorting_frequency: int = 0,
        boxes_per_dim: tuple = (12, 12, 1),
        box_bufsize: float = 2.0,
        dims_maks: tuple = (True, True, True),
    ):
        """For sorting markers in memory."""
        self.do_sort = do_sort
        self.sorting_fequency = sorting_frequency
        self.boxes_per_dim = boxes_per_dim
        self.box_bufsize = box_bufsize
        self.dims_mask = dims_maks
        
    def set_save_data(self,
                      n_markers: int | float = 3,
                      binning_plots: tuple[BinningPlot] = (),
                      n_sph: dict = None,
                      ):
        """Saving marker orits, binned data and kernel density reconstructions."""
        self.n_markers = n_markers
        self.binning_plots = binning_plots
        self.n_sph = n_sph  


class DiagnosticSpecies(Species):
    """Diagnostic species (fields) without mass and charge."""
