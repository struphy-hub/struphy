from abc import ABCMeta, abstractmethod


class DispersionRelations1D( metaclass=ABCMeta ):
    '''The base class for analytic 1d dispersion relations.
    
    Parameters
    ----------
        branch_names : list[str]
            Branche names of the spectrum.
        
        params : dict
            Physical parameters necessary to compute the dispersion relatio, e.g. c=299792458.

    Note
    ----

        Analytic Struphy dispersion relations are subclasses of ``DispersionRelations1D`` and should be added to 
        ``struphy/models/dispersion_relations/analytic.py``. 
    '''

    def __init__(self, *branch_names, **params):

        self._branches = branch_names
        self._nbranches = len(branch_names)
        self._params = params

    @property
    def branches( self ):
        '''List of branch names in the spectrum.'''
        return self._branches

    @property
    def nbranches( self ):
        '''Integer: number of branches.'''
        return self._nbranches

    @property
    def params( self ):
        '''Dictionary of parameters necessary to compute the dispersion relation.'''
        return self._params

    @abstractmethod
    def spectrum(self, kvec, kperp=None):
        '''The calculation of all branches of a 1d dispersion relation.
        
        Parameters
        ----------
            kvec : np.array
                Wave numbers.
                
            kperp : np.array
                Optional: perpendicular wave numbers (w.r.t to background magnetic field).
                kperp.size=kvec.size
                
        Returns
        -------
            A dictionary with key=name_of_branch and value=np.array(omega_values_of_branch).
            value.size=kvec.size. 
            value can be complex-valued.'''