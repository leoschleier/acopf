import pandas as pd
import numpy as np
import math

from pandas.core.frame import DataFrame

class DataImporter:
    """Facilitates import of data that serves as parameters in the ACOPF problem.
    """
    
    def __init__(self, bus_file_path:str, branch_file_path:str, gen_file_path:str, cost_file_path:str):
        self.bus_file = bus_file_path
        self.branch_file = branch_file_path
        self.gen_file = gen_file_path
        self.cost_file = cost_file_path
    
    def import_bus_data(self) -> pd.DataFrame:
        """Imports the bus data.

        Returns:
            pd.DataFrame: Bus data
        """
        return self.import_data(self.bus_file)
    
    def import_branch_data(self) -> pd.DataFrame:
        """Imports the branch data.

        Returns:
            pd.DataFrame: Branch data
        """
        return self.import_data(self.branch_file)
    
    def import_gen_data(self) -> pd.DataFrame:
        """Imports the generator data.

        Returns:
            pd.DataFrame: Generator data
        """
        return self.import_data(self.gen_file)
    
    def import_cost_data(self) -> pd.DataFrame:
        """Imports the cost data.

        Returns:
            pd.DataFrame: Cost data
        """
        return self.import_data(self.cost_file)
    
    def import_data(self, file_path:str) -> pd.DataFrame:
        """Imports a .csv file as a pandas DataFrame.
            CSV files should use ';' as separator.

        Args:
            file_path (str): Path to the .csv file.

        Returns:
            pd.DataFrame: Data from .csv file
        """
        return pd.read_csv(file_path, sep=';')


class ParameterCalculator:
    """Provides functions to calculate parameters used as input by the ModelBuilder.
    """
    
    def __init__(self):
        """Creates a ParameterCalculator instance.
        """
        self.cost = []
        self.penalty = []
        self.mismatch = []
        self.gamma = []
        self.step_size_limits = []

    def get_pwl_coefficients(self, costs_1:list, costs_2:list, p_minima:list, p_maxima:list, 
            n_segments:int) -> list:
        """Calculates the coefficients for the piecewise linear objective function.

        Args:
            costs_1 (list): Cost coefficient 1.
            costs_2 (list): Cost coefficient 2.
            p_minima (list): Minimum active power generation of each generator.
            p_maxima (list): Maximum active power generation of each generator.
            n_segments (int): Number of segments of the objective function.

        Returns:
            list: Tuple (fix_costs, list(marginal_cost_coefficients)) of cost coefficients for each generator.
        """
        c_generators = []
        for c1,c2,p_min,p_max in zip(costs_1, costs_2, p_minima, p_maxima):
            p_segment_length = (p_max-p_min)/n_segments
            c_gen = []
            for l in range(n_segments):
                c_gen.append(c1 + c2 * (p_min + l*p_segment_length + p_min + (l+1)*p_segment_length))
            c_gen_0 = c2 * p_min**2 + c1 * p_min
            c_generators.append((c_gen_0, c_gen))
        return c_generators
    
    def get_yk_matrices(self, r:list, x:list, g:list, b:list) -> list:
        """Calculates the admittance matrices for each line.

        Args:
            r (list): Resistance of each line.
            x (list): Reactance of each line.
            g (list): Shunt conductance of each line.
            b (list): Shunt supsceptance of each line.

        Returns:
            list: Admittance matrices as numpy arrays for each line.
        """
        y_k = []
        for r_nm, x_nm, g_sh, b_sh in zip(r,x,g,b):
            y_k.append(self.calc_yk_matrix(1/complex(r_nm, x_nm), complex(g_sh, b_sh)/2, complex(g_sh, b_sh)/2)) 
        return y_k
        
    def calc_yk_matrix(self, y_nm:float, y_n_sh:float, y_m_sh:float, trafo_n:int=1, trafo_m:int=1) -> np.array:
        """Calculates the admittance matrix of a single line.

        Args:
            y_nm (float): Admittance of the line from bus n to bus m.
            y_n_sh (float): Shunt admittance at bus n.
            y_m_sh (float): Shunt admittance at bus m.
            trafo_n (int, optional): Transformer weldings on the n side. Defaults to 1.
            trafo_m (int, optional): Transformer weldings on the m side. Defaults to 1.

        Returns:
            np.array: Admittance matrix
        """
        y_11 = abs(trafo_n)**2 * (y_nm + y_n_sh)
        y_12 = -trafo_n.conjugate() * trafo_m * y_nm
        y_21 = -trafo_n * trafo_m.conjugate() * y_nm
        y_22 = abs(trafo_m)**2 * (y_nm + y_m_sh)
        return np.array([[y_11, y_12] , [y_21, y_22]])
    
    def get_step_size_limits(self, p_n:list, p_n_d:list, vn_r_opt:list, vn_j_opt:list, in_r_opt:list, in_j_opt:list, vn_max:list, cost:float, penalty:float, a:float, b:float, 
            n_iteration) -> list:
        """Calculates step size limits.

        Args:
            p_n (list): Active power injection at each bus.
            p_n_d (list): Active power demand at each bus.
            vn_r_opt (list): Optimal real bus voltage at each bus.
            vn_j_opt (list): Optimal reactive bus voltage at each bus.
            in_r_opt (list): Optimal real current injected at each bus.
            in_j_opt (list): Optimal reactive current injected at each bus.
            vn_max (list): Upper limit of voltage magnitude at each bus.
            cost (float): Value of the cost function.
            penalty (float): Value of the objective function penalty.
            a (float): Step size parameter.
            b (float): Step size parameter.
            n_iteration ([type]): Number of the current iteration.

        Returns:
            list: Step size limits.
        """
        f = 0
        for pn, pnd, vnr, vnj, inr, inj in zip(p_n, p_n_d, vn_r_opt, vn_j_opt, in_r_opt, in_j_opt):
            pn_star= pnd+vnr*inr+vnj*inj
            f+= abs(pn_star-pn)
        gamma = (penalty+f)/(cost+f)
        self.gamma.append(gamma)
        self.cost.append(cost)
        self.penalty.append(penalty)
        self.mismatch.append(f)
        beta = -a*math.log(gamma)+b
        alpha = (1-int(10*gamma)/10)/beta
        limits = [alpha*abs(vm)/n_iteration**beta for vm in vn_max]
        self.step_size_limits.append(limits)
        return limits
    
    def update_iv_eva_points(self, xr_eva_dict:dict, xj_eva_dict:dict, x_max:list) -> list:
        """Updates the evaluation points and returns the keys to the updated values.

        Args:
            xr_eva_dict (dict): Real part of evaluation points.
            xj_eva_dict (dict): Imaginary part of evaluation points
            x_max (list): Upper bound.

        Raises:
            ValueError: If the keys of the dictionaries are not identical.

        Returns:
            list: Updated keys.
        """
        updated_keys = []
        for k_xr, k_xj, xmax in zip(xr_eva_dict, xj_eva_dict, x_max):
            if (xr_eva_dict[k_xr]**2 + xj_eva_dict[k_xj]**2) > (xmax**2):
                fac = math.sqrt(xmax**2 / (xr_eva_dict[k_xr]**2 + xj_eva_dict[k_xj]**2))
                xr_eva_dict[k_xr] = xr_eva_dict[k_xr]*fac
                xj_eva_dict[k_xj] = xj_eva_dict[k_xj]*fac
                if k_xr != k_xj:
                    raise ValueError('Keys must be identical.')
                updated_keys.append(k_xr)
        return updated_keys

    def calc_magnitude_n_phase(self, re_list:list, im_list:list) -> tuple:
        """Calculates magnitude and phase for given real and imaginary values.

        Args:
            re_list (list): List of real values.
            im_list (list): List of imaginary values

        Returns:
            tuple: Tuple (list(magnitude), list(phase)) of magnitude and phase.
        """
        magnitude = []
        phase = []
        for re, im in zip(re_list, im_list):
            magnitude.append((math.sqrt(re**2+im**2)))
            if im == 0:
                phase.append(0)
            else:
                phase.append(math.degrees(math.atan(re/im)))
        return magnitude, phase