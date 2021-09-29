import gurobipy as gp

class ModelBuilder:
    """Facilitates creation, adjustment, and execution of a LP approximation of the ACOPF problem.
        The model is built using the gurobipy module.
    """

    def __init__(self, n_nodes:int, gen_nodes:list, n_obj_segments:int, lines:dict, lp_name='ACOPF'):
        """Creates a ModelBuilder instance.

        Args:
            n_nodes (int): Number of nodes (buses) of the grid. The nodes are numbered from 1 to n_nodes+1.
            gen_nodes (list): List of node numbers that include a generator.
            n_obj_segments (int): Number of linear segments that will approximate the objective function.
            lines (dict): Dictionary where lines['nm'] and lines['mn'] contain a list of tuples of nodes (x,y) 
                and (y,x) that describe a line.
            lp_name (str, optional): Name of the resulting LP. Defaults to 'ACOPF'.
        """
        self.model = gp.Model('LP: '+str(lp_name))
        self.obj_vars = self.model.addVars(
            gen_nodes, range(n_obj_segments), name='obj_vars')
        self.p_n = self.model.addVars(
            range(1, n_nodes+1), lb=-gp.GRB.INFINITY, name='p_n')
        self.q_n = self.model.addVars(
            range(1, n_nodes+1), lb=-gp.GRB.INFINITY, name='q_n')
        self.v_n_r = self.model.addVars(
            range(1, n_nodes+1), lb=-gp.GRB.INFINITY, name='v_n_r')
        self.v_n_j = self.model.addVars(
            range(1, n_nodes+1), lb=-gp.GRB.INFINITY, name='v_n_j')
        self.v_n_sq = self.model.addVars(
            range(1, n_nodes+1), lb=-gp.GRB.INFINITY, name='v_n_sq')
        self.i_n_r = self.model.addVars(
            range(1, n_nodes+1), lb=-gp.GRB.INFINITY, name='i_n_r')
        self.i_n_j = self.model.addVars(
            range(1, n_nodes+1), lb=-gp.GRB.INFINITY, name='i_n_j')
        self.i_k_r = self.model.addVars(
            lines['nm']+lines['mn'], lb=-gp.GRB.INFINITY, name='i_k_r')
        self.i_k_j = self.model.addVars(
            lines['nm']+lines['mn'], lb=-gp.GRB.INFINITY, name='i_k_j')
        self.i_k_sq = self.model.addVars(
            lines['nm']+lines['mn'], lb=-gp.GRB.INFINITY, name='i_k_sq')

        self.pn_viol_l = self.model.addVars(
            range(1, n_nodes+1), lb=0, name='pn_viol_l')
        self.pn_viol_u = self.model.addVars(
            range(1, n_nodes+1), lb=0, name='pn_viol_u')
        self.qn_viol_l = self.model.addVars(
            range(1, n_nodes+1), lb=0, name='qn_viol_l')
        self.qn_viol_u = self.model.addVars(
            range(1, n_nodes+1), lb=0, name='qn_viol_u')
        self.vn_viol_l = self.model.addVars(
            range(1, n_nodes+1), lb=0, name='vn_viol_l')
        self.vn_viol_u = self.model.addVars(
            range(1, n_nodes+1), lb=0, name='vn_viol_u')
        self.ik_viol_u = self.model.addVars(
            lines['nm']+lines['mn'], lb=0, name='ik_viol_u')

        self.model.update()

        self.n_nodes = n_nodes
        self.gen_nodes = gen_nodes
        self.lines = lines
        self.n_obj_segments = n_obj_segments

        self.cut_constraints = []
        self.step_size_constraints = []
        self.vn_taylor = None
        self.pn_taylor = None
        self.qn_taylor = None
        self.ik_taylor = None

        self.offers = gp.LinExpr()
        self.penalty = gp.LinExpr()

    def run(self, write_solution=False):
        """Starts the optimization process of the Gurobi model.

        Args:
            write_solution (bool, optional): If True, the .sol file of the solution found is written to the 
                current folder. Defaults to False.
        """
        self.model.update()
        self.model.optimize()
        if write_solution:
            self.write_solution()
        return

    def write_model(self, filename="model"):
        """Writes the current Gurobi model to a .lp file in the current folder.

        Args:
            filename (str, optional): Name of the .lp file. Defaults to "model".
        """
        self.model.write(filename+".lp")
        return

    def write_solution(self, filename="solution"):
        """Writes the current Gurobi solution to a .sol file in the current folder.

        Args:
            filename (str, optional): Name of the .sol file. Defaults to "solution".
        """
        self.model.write(filename+".sol")
        return

    def write_iis(self, filename="model_iis"):
        """Computes and writes the Irreducible Inconsistent Subsystem (IIS) of the current model to the current folder

        Args:
            filename (str, optional): Name of the .ilp file. Defaults to "model_iis".
        """
        self.model.computeIIS()
        self.model.write(filename+".ilp")
        return

    def add_objective(self, costs_generators:list, pn_viol_facts:list, qn_viol_facts:list, vn_viol_facts:list, 
            ik_viol_facts:list):
        """Adds the objective (cost) function to the model.

        Args:
            costs_generators (list): List of tuples of cost coefficients (fix_costs, list(marginal_costs)) for each 
                generator.
            pn_viol_facts (list): Violation factors for active power variables.
            qn_viol_facts (list): Violation factors for reactive power variables.
            vn_viol_facts (list): Violation factors for bus voltage variables.
            ik_viol_facts (list): Violation factors for line current variables.
        """
        c_0_sum = sum([x[0] for x in costs_generators])
        c_gen_all = []
        for x in costs_generators:
            c_gen_all += x[1]

        self.offers.addTerms(c_gen_all, self.obj_vars.values())
        self.offers.addConstant(c_0_sum)

        self.penalty.addTerms(
            pn_viol_facts*2, self.pn_viol_l.values()+self.pn_viol_u.values())
        self.penalty.addTerms(
            qn_viol_facts*2, self.qn_viol_l.values()+self.qn_viol_u.values())
        self.penalty.addTerms(
            vn_viol_facts*2, self.vn_viol_l.values()+self.vn_viol_u.values())
        self.penalty.addTerms(ik_viol_facts, self.ik_viol_u.values())

        self.model.setObjective(self.offers+self.penalty, gp.GRB.MINIMIZE)
        return

    def add_obj_constraints(self, p_minima: list, p_maxima:list):
        """Adds constraints to the objective (generator active power) variables

        Args:
            p_minima (list): Lower bounds for active power at each generator.
            p_maxima (list): Upper bounds for active power at each generator.
        """
        for n, p_min, p_max in zip(self.gen_nodes, p_minima, p_maxima):
            p_segment_length = (p_max-p_min)/self.n_obj_segments
            self.model.addConstrs(
                self.obj_vars[n, l] <= p_segment_length for l in range(self.n_obj_segments))
            self.model.addConstr(
                self.p_n[n] == self.obj_vars.sum(n, '*')+p_min)
        return

    def add_flow_constraints(self, yk_matrices:list, gn_shunt:list, bn_shunt:list):
        """Adds current flow constraints for both directions of each line.

        Args:
            yk_matrices (list): Admittance matrices for each line. 
            gn_shunt (list): Shunt conductance for each node.
            bn_shunt (list): Shunt supsceptance for each node.
        """
        for yk, nm, mn in zip(yk_matrices, self.lines['nm'], self.lines['mn']):
            self.model.addConstr(yk[0, 0].real*self.v_n_r[nm[0]] - yk[0, 0].imag*self.v_n_j[nm[0]]
                                 + yk[0, 1].real*self.v_n_r[nm[1]] - yk[0, 1].imag*self.v_n_j[nm[1]] == self.i_k_r[nm])
            self.model.addConstr(yk[0, 0].real*self.v_n_j[nm[0]] + yk[0, 0].imag*self.v_n_r[nm[0]]
                                 + yk[0, 1].real*self.v_n_j[nm[1]] + yk[0, 1].imag*self.v_n_r[nm[1]] == self.i_k_j[nm])
            self.model.addConstr(yk[1, 0].real*self.v_n_r[nm[0]] - yk[1, 0].imag*self.v_n_j[nm[0]]
                                 + yk[1, 1].real*self.v_n_r[nm[1]] - yk[1, 1].imag*self.v_n_j[nm[1]] == self.i_k_r[mn])
            self.model.addConstr(yk[1, 0].real*self.v_n_j[nm[0]] + yk[1, 0].imag*self.v_n_r[nm[0]]
                                 + yk[1, 1].real*self.v_n_j[nm[1]] + yk[1, 1].imag*self.v_n_r[nm[1]] == self.i_k_j[mn])

        for n in range(1, self.n_nodes+1):
            self.model.addConstr(self.i_k_r.sum(n, '*') + gn_shunt[n-1]*self.v_n_r[n] - bn_shunt[n-1]*self.v_n_j[n]
                                 == self.i_n_r[n])
            self.model.addConstr(self.i_k_j.sum(n, '*') + gn_shunt[n-1]*self.v_n_j[n] - bn_shunt[n-1]*self.v_n_r[n]
                                 == self.i_n_j[n])
        return

    def add_step_size_constraints(self, vnr_eva_points:list, vnj_eva_points:list, vn_limits:list):
        """Adds step size constraints.

        Args:
            vnr_eva_points (list): Evaluation points of the real part of the bus voltages.
            vnj_eva_points (list): Evaluation points of the imaginary part of the bus voltages.
            vn_limits (list): Bus voltage limits that serve as step size limits.
        """
        self.step_size_constraints.append(self.model.addConstrs(self.v_n_r[n]-vnr_eva_points[n-1] <= vn_limits[n-1]
                                                                for n in range(1, self.n_nodes+1)))
        self.step_size_constraints.append(self.model.addConstrs(self.v_n_j[n]-vnj_eva_points[n-1] <= vn_limits[n-1]
                                                                for n in range(1, self.n_nodes+1)))
        self.step_size_constraints.append(self.model.addConstrs(-self.v_n_r[n]+vnr_eva_points[n-1] <= vn_limits[n-1]
                                                                for n in range(1, self.n_nodes+1)))
        self.step_size_constraints.append(self.model.addConstrs(-self.v_n_j[n]+vnj_eva_points[n-1] <= vn_limits[n-1]
                                                                for n in range(1, self.n_nodes+1)))
        return

    def add_box_constraints(self, vn_maxima:list, ik_maxima:dict):
        """Adds box constraints to voltage and current variables.

        Args:
            vn_maxima (list): Upper limits of bus voltages.
            ik_maxima (dict): Upper limits of line currents.
        """
        self.model.addConstrs(self.v_n_r[n] >= -vn_maxima[n-1]
                              for n in range(1, self.n_nodes+1))
        self.model.addConstrs(self.v_n_r[n] <= vn_maxima[n-1]
                              for n in range(1, self.n_nodes+1))
        self.model.addConstrs(self.v_n_j[n] >= -vn_maxima[n-1]
                              for n in range(1, self.n_nodes+1))
        self.model.addConstrs(self.v_n_j[n] <= vn_maxima[n-1]
                              for n in range(1, self.n_nodes+1))

        for nm, mn, ik_max in zip(self.lines['nm'], self.lines['mn'], ik_maxima):
            self.model.addConstrs(self.i_k_r[ij] >= -ik_max for ij in [nm, mn])
            self.model.addConstrs(self.i_k_r[ij] <= ik_max for ij in [nm, mn])
            self.model.addConstrs(self.i_k_j[ij] >= -ik_max for ij in [nm, mn])
            self.model.addConstrs(self.i_k_j[ij] <= ik_max for ij in [nm, mn])
        return

    def add_taylor_constraints(self, vnr_eva:list, vnj_eva:list, inr_eva:list, inj_eva:list, ikr_eva:dict, 
            ikj_eva:dict, pn_d:list, qn_d:list):
        """Adds taylor series approximations of the nonlinear ACOPF constraints.

        Args:
            vnr_eva (list): Real bus voltage evaluation points.
            vnj_eva (list): Imaginary bus voltage evaluation points.
            inr_eva (list): Real injected current evaluation points.
            inj_eva (list): Imaginary injected current evaluation points.
            ikr_eva (dict): Real line current evaluation points.
            ikj_eva (dict): Imaginary line current evaluation points.
            pn_d (list): Active power demand at each bus.
            qn_d (list): Reactive power demand at each bus.
        """
        self.vn_taylor = self.model.addConstrs(self.v_n_sq[n] == 2*vnr_eva[n-1]*self.v_n_r[n]
                                               + 2*vnj_eva[n-1]*self.v_n_j[n] -
                                               vnr_eva[n-1]**2 -
                                               vnj_eva[n-1]**2
                                               for n in range(1, self.n_nodes+1))

        self.pn_taylor = self.model.addConstrs(self.p_n[n] == vnr_eva[n-1]*self.i_n_r[n]
                                               + vnj_eva[n-1]*self.i_n_j[n] + self.v_n_r[n] *
                                               inr_eva[n-1] +
                                               self.v_n_j[n]*inj_eva[n-1]
                                               - vnr_eva[n-1]*inr_eva[n-1] - vnj_eva[n-1]*inj_eva[n-1] + pn_d[n-1]
                                               for n in range(1, self.n_nodes+1))

        self.qn_taylor = self.model.addConstrs(self.q_n[n] == vnj_eva[n-1]*self.i_n_r[n] - vnr_eva[n-1]*self.i_n_j[n]
                                               + self.v_n_j[n]*inr_eva[n-1] - self.v_n_r[n]*inj_eva[n-1] - vnj_eva[n-1]
                                               * inr_eva[n-1]
                                               + vnr_eva[n-1]*inj_eva[n-1] + qn_d[n-1] for n in range(1, self.n_nodes+1))

        self.ik_taylor = self.model.addConstrs(self.i_k_sq[ij] == 2*ikr_e*self.i_k_r[ij] + 2*ikj_e*self.i_k_j[ij]
                                               - ikr_e**2 - ikj_e**2 for ij, ikr_e, ikj_e in zip(self.lines['nm']
                                               + self.lines['mn'], ikr_eva, ikj_eva))
        return

    def add_power_constraints(self, p_min:list, p_max:list, q_min:list, q_max:list):
        """Adds upper and lower limits for each bus. 

        Args:
            p_min (list): Lower limits to active power at each bus.
            p_max (list): Upper limits to active power at each bus.
            q_min (list): Lower limits to reactive power at each bus.
            q_max (list): Upper limits to reactive power at each bus.
        """
        self.model.addConstrs(
            self.p_n[n] <= p_max[n-1]+self.pn_viol_u[n] for n in range(1, self.n_nodes+1))
        self.model.addConstrs(
            self.p_n[n] >= p_min[n-1]-self.pn_viol_l[n] for n in range(1, self.n_nodes+1))
        self.model.addConstrs(
            self.q_n[n] <= q_max[n-1]+self.qn_viol_u[n] for n in range(1, self.n_nodes+1))
        self.model.addConstrs(
            self.q_n[n] >= q_min[n-1]-self.qn_viol_l[n] for n in range(1, self.n_nodes+1))
        return

    def add_iv_constraints(self, vn_min:list, vn_max:list, ik_max:dict):
        """Adds constraints for absolute bus voltages and absolute line currents.

        Args:
            vn_min (list): Lower limit for bus voltage.
            vn_max (list): Upper limit for bus voltage.
            ik_max (dict): Upper limit for line currents.
        """
        self.model.addConstrs(
            self.v_n_sq[n] <= vn_max[n-1]**2 + self.vn_viol_u[n] for n in range(1, self.n_nodes+1))
        self.model.addConstrs(
            self.v_n_sq[n] >= vn_min[n-1]**2 + self.vn_viol_l[n] for n in range(1, self.n_nodes+1))

        for nm, mn, imax in zip(self.lines['nm'], self.lines['mn'], ik_max):
            self.model.addConstr(
                self.i_k_sq[nm] <= imax**2 + self.ik_viol_u[nm])
            self.model.addConstr(
                self.i_k_sq[mn] <= imax**2 + self.ik_viol_u[mn])
        return

    def add_voltage_cutting_plane(self, vnr_eva_dict:dict, vnj_eva_dict:dict, v_max:list, vn_updated_keys:list):
        """Adds  bus voltage cutting plane to cut off solutions that are infeasible in the original ACOPF.

        Args:
            vnr_eva_dict (dict): New voltage evaluation points.
            vnj_eva_dict (dict): New voltage evaluation points.
            v_max (list): Upper limit of absolute voltage level.
            vn_updated_keys (list): Keys of bus voltages that require a cutting plane.
        """
        for key, vmax in zip(vn_updated_keys, v_max):
            self.add_cutting_plane(self.v_n_r[key], vnr_eva_dict[key], self.v_n_j[key], vnj_eva_dict[key], vmax,
                                   self.vn_viol_u[key])
        return

    def add_line_current_cutting_plane(self, ikr_eva_dict:dict, ikj_eva_dict:dict, ik_max:list, ik_updated_keys:list):
        """Adds line current cutting plane to cut off solutions that are infeasible in the original ACOPF.

        Args:
            ikr_eva_dict (dict): New line current evaluation points.
            ikj_eva_dict (dict): New line current evaluation points.
            ik_max (list): Upper limit of absolute line current.
            ik_updated_keys (list): Keys of kube currents that require a cutting plane.
        """
        for key, imax in zip(ik_updated_keys, ik_max):
            self.add_cutting_plane(self.i_k_r[key], ikr_eva_dict[key], self.i_k_j[key], ikj_eva_dict[key], imax,
                                   self.ik_viol_u[key])
        return

    def add_cutting_plane(self, xr, xr_eva, xj, xj_eva, x_max, x_viol_u):
        """Adds a cutting plane to the problem.
        """
        self.cut_constraints.append(self.model.addConstr(
            xr_eva*xr + xj_eva*xj <= x_max**2 + x_viol_u))
        return

    def remove_cutting_planes(self):
        """Removes all cutting planes
        """
        for constrs in self.cut_constraints:
            self.model.remove(constrs)
            self.cut_constraints = []
        return

    def remove_step_size_constraints(self):
        """Removes all step size constraints
        """
        self.model.remove(self.step_size_constraints)
        self.step_size_constraints = []
        return

    def remove_taylor_constraints(self):
        """Removes all taylor constraints.
        """
        self.model.remove(self.vn_taylor)
        self.model.remove(self.pn_taylor)
        self.model.remove(self.qn_taylor)
        self.model.remove(self.ik_taylor)
        self.vn_taylor = None
        self.pn_taylor = None
        self.qn_taylor = None
        self.ik_taylor = None
        return