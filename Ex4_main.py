from Ex_4_const import *
from Ex_4 import *
import numpy as np

step = 0
for t in np.arange(T_min, T_max + 0.2, 0.2):
    U = U_Levels()
    my = myu_binary_search_Step(N, n_max, t)
    while step < 1:  # for now
        # TODO change the step definition
        N_i = random.randint(1, N) # random particle
        print("N_i", N_i)
        print("U_store", U.U_initial_state())
        print("n_top", U.n_top)
        print("n_bottom", U.n_bottom)
        p_b = max(np.where(U.n_bottom <= N_i, U.n_bottom, U.n_bottom != 0))
        p_b_i = np.where(p_b == U.n_bottom)[0][0]
        p_n_i = np.where(N_i <= U.n_top)[0][0]
        if p_n_i == p_b_i:
            print('yay!!!')
            print("p_b= ", p_b)
            print("p_n= ", U.n_top[p_n_i])
        P_minus = energy_minus_probability(p_n_i, t, my)  # true or false for n-1
        # TODO always P_minus=1?
        if P_minus:
            U.n_bottom[p_b_i] += 1
            U.n_top[p_n_i - 1] += 1
            # TODO what if there was no particle at n-1
            # TODO what if he was the only particle (need to change to zero)
        else:
            U.n_top[p_n_i] -= 1
            U.n_bottom[p_b_i + 1] += 1
            # TODO check if right

        print("P_minus= ", P_minus)
        print("n_top", U.n_top)
        print("n_bottom", U.n_bottom)
        step += 1

