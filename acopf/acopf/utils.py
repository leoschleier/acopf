import pandas as pd
import numpy as np
import math

class DataImporter:
    
    def __init__(self, bus_file, branch_file, gen_file, cost_file):
        self.bus_file = bus_file
        self.branch_file = branch_file
        self.gen_file = gen_file
        self.cost_file = cost_file
    
    def import_bus_data(self):
        return self.import_data(self.bus_file)
    
    def import_branch_data(self):
        return self.import_data(self.branch_file)
    
    def import_gen_data(self):
        return self.import_data(self.gen_file)
    
    def import_cost_data(self):
        return self.import_data(self.cost_file)
    
    def import_data(self, file):
        return pd.read_csv(file, sep=';')


class ParameterCalculator:
    
    def __init__(self):
        self.cost = []
        self.penalty = []
        self.mismatch = []
        self.gamma = []
        self.step_size_limits = []

    def get_pwl_coefficients(self, costs_1, costs_2, p_minima, p_maxima, base_MVA, n_segments):
        c_generators = []
        for c1,c2,p_min,p_max in zip(costs_1, costs_2, p_minima, p_maxima):
            p_segment_length = (p_max-p_min)/n_segments
            c_gen = []
            for l in range(n_segments):
                c_gen.append(c1 + c2 * (p_min + l*p_segment_length + p_min + (l+1)*p_segment_length))
            c_gen_0 = c2 * p_min**2 + c1 * p_min
            c_generators.append((c_gen_0, c_gen))
        return c_generators
    
    def get_yk_matrices(self, r, x, g, b):
        y_k = []
        for r_nm, x_nm, g_sh, b_sh in zip(r,x,g,b):
            y_k.append(self.calc_yk_matrix(1/complex(r_nm, x_nm), complex(g_sh, b_sh)/2, complex(g_sh, b_sh)/2)) 
        return y_k
        
    def calc_yk_matrix(self, y_nm, y_n_sh, y_m_sh, trafo_n=1, trafo_m=1):
        y_11 = abs(trafo_n)**2 * (y_nm + y_n_sh)
        y_12 = -trafo_n.conjugate() * trafo_m * y_nm
        y_21 = -trafo_n * trafo_m.conjugate() * y_nm
        y_22 = abs(trafo_m)**2 * (y_nm + y_m_sh)
        return np.array([[y_11, y_12] , [y_21, y_22]])
    
    def get_step_size_limits(self, p_n, p_n_d, vn_r_opt, vn_j_opt, in_r_opt, in_j_opt, vn_max, cost, penalty, a, b, n_iteration):
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
    
    def update_iv_eva_points(self, xr_eva_dict, xj_eva_dict, x_max):
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

    def calc_magnitude_n_phase(self, re_list, im_list):
        magnitude = []
        phase = []
        for re, im in zip(re_list, im_list):
            magnitude.append((math.sqrt(re**2+im**2)))
            if im == 0:
                phase.append(0)
            else:
                phase.append(math.degrees(math.atan(re/im)))
        return magnitude, phase