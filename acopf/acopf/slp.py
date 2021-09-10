import gurobipy as gp


class ModelBuilder:

    def __init__(self, n_nodes, gen_nodes, n_obj_segments, lines, lp_name='ACOPF'):
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
        self.model.update()
        self.model.optimize()
        if write_solution:
            self.write_solution()
        return

    def write_model(self, filename="model"):
        self.model.write(filename+".lp")
        return

    def write_solution(self, filename="solution"):
        self.model.write(filename+".sol")
        return

    def write_iis(self, filename="model_iis"):
        self.model.computeIIS()
        self.model.write(filename+".ilp")
        return

    def add_objective(self, costs_generators, pn_viol_facts, qn_viol_facts, vn_viol_facts, ik_viol_facts):
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

    def add_obj_constraints(self, p_minima, p_maxima):
        for n, p_min, p_max in zip(self.gen_nodes, p_minima, p_maxima):
            p_segment_length = (p_max-p_min)/self.n_obj_segments
            self.model.addConstrs(
                self.obj_vars[n, l] <= p_segment_length for l in range(self.n_obj_segments))
            self.model.addConstr(
                self.p_n[n] == self.obj_vars.sum(n, '*')+p_min)
        return

    def add_flow_constraints(self, yk_matrices, gn_shunt, bn_shunt):
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

    def add_step_size_constraints(self, vnr_eva_points, vnj_eva_points, vn_limits):
        self.step_size_constraints.append(self.model.addConstrs(self.v_n_r[n]-vnr_eva_points[n-1] <= vn_limits[n-1]
                                                                for n in range(1, self.n_nodes+1)))
        self.step_size_constraints.append(self.model.addConstrs(self.v_n_j[n]-vnj_eva_points[n-1] <= vn_limits[n-1]
                                                                for n in range(1, self.n_nodes+1)))
        self.step_size_constraints.append(self.model.addConstrs(-self.v_n_r[n]+vnr_eva_points[n-1] <= vn_limits[n-1]
                                                                for n in range(1, self.n_nodes+1)))
        self.step_size_constraints.append(self.model.addConstrs(-self.v_n_j[n]+vnj_eva_points[n-1] <= vn_limits[n-1]
                                                                for n in range(1, self.n_nodes+1)))
        return

    def add_box_constraints(self, vn_maxima, ik_maxima):
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

    def add_taylor_constraints(self, vnr_eva, vnj_eva, inr_eva, inj_eva, ikr_eva, ikj_eva, pn_d, qn_d):
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

    def add_power_constraints(self, p_min, p_max, q_min, q_max):
        self.model.addConstrs(
            self.p_n[n] <= p_max[n-1]+self.pn_viol_u[n] for n in range(1, self.n_nodes+1))
        self.model.addConstrs(
            self.p_n[n] >= p_min[n-1]-self.pn_viol_l[n] for n in range(1, self.n_nodes+1))
        self.model.addConstrs(
            self.q_n[n] <= q_max[n-1]+self.qn_viol_u[n] for n in range(1, self.n_nodes+1))
        self.model.addConstrs(
            self.q_n[n] >= q_min[n-1]-self.qn_viol_l[n] for n in range(1, self.n_nodes+1))
        return

    def add_iv_constraints(self, vn_min, vn_max, ik_max):
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

    def add_voltage_cutting_plane(self, vnr_eva_dict, vnj_eva_dict, v_max, vn_updated_keys):
        for key, vmax in zip(vn_updated_keys, v_max):
            self.add_cutting_plane(self.v_n_r[key], vnr_eva_dict[key], self.v_n_j[key], vnj_eva_dict[key], vmax,
                                   self.vn_viol_u[key])
        return

    def add_line_current_cutting_plane(self, ikr_eva_dict, ikj_eva_dict, ik_max, ik_updated_keys):
        for key, imax in zip(ik_updated_keys, ik_max):
            self.add_cutting_plane(self.i_k_r[key], ikr_eva_dict[key], self.i_k_j[key], ikj_eva_dict[key], imax,
                                   self.ik_viol_u[key])
        return

    def add_cutting_plane(self, xr, xr_eva, xj, xj_eva, x_max, x_viol_u):
        self.cut_constraints.append(self.model.addConstr(
            xr_eva*xr + xj_eva*xj <= x_max**2 + x_viol_u))
        return

    def remove_cutting_planes(self):
        for constrs in self.cut_constraints:
            self.model.remove(constrs)
            self.cut_constraints = []
        return

    def remove_step_size_constraints(self):
        self.model.remove(self.step_size_constraints)
        self.step_size_constraints = []
        return

    def remove_taylor_constraints(self):
        self.model.remove(self.vn_taylor)
        self.model.remove(self.pn_taylor)
        self.model.remove(self.qn_taylor)
        self.model.remove(self.ik_taylor)
        self.vn_taylor = None
        self.pn_taylor = None
        self.qn_taylor = None
        self.ik_taylor = None
        return