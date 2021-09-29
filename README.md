# acopf
Implementation of a successive linear programming (SLP) algorithm to solve the alternating current optimal power flow (ACOPF) problem.

## The Alternating Current Optimal Power Flow Problem
The transformation of the energy sector manifests itself in various shapes. Among them are sector coupling and a more decentralized power generation. At the same time, the different types of power plants are not evenly distributed (e.g., wind turbines in coastal regions and photovoltaic systems in the south). As a consequence, optimal planning and operation of the energy system become increasingly difficult. An instrument to overcome some of those difficulties is the so-called optimal power flow (OPF) problem. Its goal is to find an optimal operating state of the power system subject to system-related or, in particular, grid-based constraints. In this context, the alternating current optimal power flow (ACOPF) problem searches for solutions in the domain of an AC grid.
The ACOPF is NP-hard. Hence, a globally optimal solution can probably not be found efficiently. Approximations of the ACOPF, which are easier to solve, are developed to find an optimal solution within a reasonable amount of time. Moreover, requirements on ACOPF solutions vary. Thus, various approximations and solution-seeking techniques exist.

## A Successive Linear Programming Algorithm
Castillo et. al [1] use a linear program (LP) to approximate a version of the ACOPF. As linear programming itself would not provide an adequate solution to the original problem, an algorithm is set up to modify and solve the LP iteratively. Their successive linear programming (SLP) algorithm adds, adjusts, and removes certain constraints of the LP approximation.

The present implementation of central parts of the SLP algorithm is able to efficiently find a local optimal solutions to the original ACOPF problem.

## Test Cases
Test cases are drawn from Bukhsh et al. [2] who examine ACOPFs for local optimal solutions. Their results can be found in an online archive [3]. Those test cases are chosen, because they enable a verifcation of results while studying the behavior of the algorithm.

## References
[1] Castillo, A., Lipka, P., Watson, J.-P., Oren, S. S., and O'Neill, R. P. A successive linear programming approach to solving the iv-acopf. IEEE Transactions on Power Systems 31, 4 (2016), 2752-2763.

[2] Bukhsh, W. A., Grothey, A., McKinnon, K. I. M., and Trodden, P. A. Local Solutions of the Optimal Power Flow Problem. IEEE Transactions on Power Systems 28, 4 (2013), 4780-4788.

[3] Test Case Archive of Optimal Power Flow (OPF) Problems with Local Optima, 04.03.2015. https://www.maths.ed.ac.uk/optenergy/LocalOpt/.
