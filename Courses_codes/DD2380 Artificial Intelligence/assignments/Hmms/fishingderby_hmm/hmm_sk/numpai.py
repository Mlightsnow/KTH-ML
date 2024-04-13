import math
import sys
epsilon = sys.float_info.epsilon

class Matrix:

    def __init__(self, input_row, isvector = False):
        if isvector:
            self.n_row = 1
            self.n_col = int(input_row[0])
            self.input_list = [int(x) if type(x) == type('a') else x for x in input_row[1:]] # type of number in the obs seq should be int
        else:
            self.n_row = int(input_row[0])
            self.n_col = int(input_row[1])
            self.input_list = [float(x) if type(x) == type('a') else x for x in input_row[2:]]

        self.trans_matrice = None
        self.is_input_list_changed = True

    def get_n_row(self):
        return self.n_row
    
    def get_n_col(self):
        return self.n_col
    
    def get_input_list(self):
        return self.input_list
    
    def multiply(self, B):
        assert self.n_col == B.n_row, "The col of A is not equal to the row of B!"

        result_rows = self.n_row
        result_cols = B.n_col
        result_elements = [0] * (result_rows * result_cols)

        for i in range(result_rows):
            for j in range(result_cols):
                dot_product = 0
                for k in range(self.n_col):
                    dot_product += self.input_list[i * self.n_col + k] * B.input_list[k * B.n_col + j]
                result_elements[i * result_cols + j] = dot_product

        result = [result_rows, result_cols]
        for x in result_elements:
            result.append(x)

        return Matrix(result)
    
    def trans_to_matrice(self):
        if self.is_input_list_changed == False:
            return self.trans_matrice
        
        self.trans_matrice = []
        for i in range(self.n_row):
            row = []
            for j in range(self.n_col):
                row.append(self.input_list[i*self.n_col + j])
            self.trans_matrice.append(row)

        self.is_input_list_changed = False
        return self.trans_matrice
    
    def list_append(self, append_list):
        '''
        append_list must be the one-dimensional vector like [1,2,3]
        '''
        assert len(append_list) == self.n_col, "the size of list appended is not equal to the n_col of the Matrix"
        self.input_list = self.input_list + append_list
        self.n_row += 1
        self.is_input_list_changed = True
        self.trans_to_matrice()

    def slice(self, i: int, axis = 0):
        '''
        return the list that is self.trans_matrice[i][:] or self.trans_matrice[:][i]
        axis = 0 row vector
        axis = 1 col vector
        '''
        self.trans_to_matrice()
        if axis == 0:
            return self.trans_matrice[i]
        else:
            L = []
            for row in self.trans_matrice:
                L.append(row[i])
            return L

class HMM:
    def __init__(self, A: Matrix, B: Matrix, pi: Matrix):
        self.trans_M = A
        self.obs_M = B
        self.init_state = pi
        self.n_states = B.get_n_row()
        self.n_ob_type = B.get_n_col()
    
    def get_trans_M(self):
        return self.trans_M.trans_to_matrice()
    
    def get_obs_M(self):
        return self.obs_M.trans_to_matrice()
    
    def get_init_state(self):
        return self.init_state.trans_to_matrice()

    def estimate_S_sq(self, Osq: Matrix):
        '''
        input: A, B, pi of HMM model and observation squence (class Matrix)
        output: the estimation squence of states using Viterbi algorithm

        '''
        obs_sq = Osq.get_input_list()
        n_obs_sq = Osq.get_n_col()
        delta_1 = []
        b_o1 = self.obs_M.slice(obs_sq[0], axis = 1)

        for i in range(self.n_states):
            delta_1.append(b_o1[i] * self.init_state.input_list[i])
            
        delta = Matrix([0, self.n_states])
        delta_idx = Matrix([0, self.n_states])
        delta.list_append(delta_1)
        
        for t in range(1, n_obs_sq):
            delta_prev = delta.trans_to_matrice()[-1]
            delta_t = []
            delta_idx_t = []

            for i in range(self.n_states):
                b_ot = self.obs_M.slice(obs_sq[t], axis = 1)
                L = [self.trans_M.trans_to_matrice()[j][i] * delta_prev[j] * b_ot[i] for j in range(self.n_states)]
                delta_t.append(max(L))
                delta_idx_t.append(L.index(max(L)))

            delta.list_append(delta_t)
            delta_idx.list_append(delta_idx_t)

        est_state_sq = []
        LastL = delta.trans_to_matrice()[-1]
        est_state_sq.append(LastL.index(max(LastL)))
        
        #print(delta.input_list)

        for i in range(1, self.n_obs_sq):
            L = delta_idx.trans_to_matrice()[delta_idx.get_n_row() - i]
            est_state_sq.append(L[est_state_sq[i - 1]])
        est_state_sq = est_state_sq[::-1]

        return est_state_sq
    

    def alpha_pass(self, Osq: Matrix):
        obs_sq = Osq.get_input_list()
        n_obs_sq = Osq.get_n_col()

        B_matrix = self.obs_M.trans_to_matrice()
        A_matrix = self.trans_M.trans_to_matrice()
        
        # calculate alpha_0
        N_state = self.n_states
        obs_0 = obs_sq[0]
        alpha = []
        alpha_vector = []
        T = n_obs_sq
        scaler = [0] * T
        temp_s = 0

        for i in range(N_state):
            #print(self.init_state.input_list[i])
            #print(B_matrix)
            alpha.insert(i, self.init_state.input_list[i] * B_matrix[i][obs_0])
            temp_s += alpha[i]

        scaler[0] = 1 / (temp_s + epsilon)
        for i in range(N_state):
            alpha[i] = scaler[0] * alpha[i]

        alpha_vector.insert(0, alpha)

        # calculate t = 1 ~ T-1
        for t in range(1, n_obs_sq):
            alpha = []
            obs_t = obs_sq[t]
            alpha_t_1 = alpha_vector[t - 1]
            temp_s = 0

            for i in range(N_state):
                sum = 0
                for j in range(N_state):
                    sum += alpha_t_1[j] * A_matrix[j][i]
                alpha.insert(i, sum * B_matrix[i][obs_t])
                temp_s += alpha[i]
            
            scaler[t] = 1 / (temp_s + epsilon)
            for i in range(N_state):
                alpha[i] = scaler[t] * alpha[i]

            alpha_vector.insert(t, alpha)
        
        return alpha_vector, scaler
    
    def beta_pass(self, Osq: Matrix, scaler: list):
        obs_sq = Osq.get_input_list()
        n_obs_sq = Osq.get_n_col()
        B_matrix = self.obs_M.trans_to_matrice()
        A_matrix = self.trans_M.trans_to_matrice()

        N_state = self.n_states
        T = n_obs_sq
        beta = []
        beta_vector = [None] * T

        for i in range(N_state):
            beta.insert(i, scaler[T - 1])
        beta_vector[T - 1] = beta

        for t in range(T - 2, -1, -1):
            beta = []
            obs_t1 = obs_sq[t + 1]
            beta_t1 = beta_vector[t + 1]

            for i in range(N_state):
                sum = 0
                for j in range(N_state):
                    sum += A_matrix[i][j] * B_matrix[j][obs_t1] * beta_t1[j]
                beta.insert(i, sum)
                beta[i] = scaler[t] * beta[i]

            beta_vector[t] = beta

        return beta_vector

    def Gamma(self, a: list, b: list, Osq: Matrix):
        obs_sq = Osq.get_input_list()
        n_obs_sq = Osq.get_n_col()
        B_matrix = self.obs_M.trans_to_matrice()
        A_matrix = self.trans_M.trans_to_matrice()
        N_state = self.n_states
        T = n_obs_sq

        di_gamma = [[[0 for _ in range(N_state)] for _ in range(N_state)] for _ in range(T)]
        gamma = [[0 for _ in range(N_state)] for _ in range(T)]

        for t in range(T - 1):
            for i in range(N_state):
                for j in range(N_state):
                    di_gamma[t][i][j] = a[t][i] * A_matrix[i][j] * B_matrix[j][obs_sq[t + 1]] * b[t + 1][j]
                    gamma[t][i] += di_gamma[t][i][j]

        for i in range(N_state):
            gamma[T - 1][i] = a[T - 1][i]

        return gamma, di_gamma
    
    def re_estimate(self, Osq : Matrix):

        alpha, c = self.alpha_pass(Osq)
        beta = self.beta_pass(Osq, c)
        gamma, di_gamma = self.Gamma(alpha, beta, Osq)

        obs_sq = Osq.get_input_list()
        n_obs_sq = Osq.get_n_col()
        new_A = self.trans_M.trans_to_matrice()
        new_B = self.obs_M.trans_to_matrice()
        N_state = self.n_states
        M = self.n_ob_type
        T = n_obs_sq

        for i in range(N_state):
            self.init_state.input_list[i] = gamma[0][i]
            sum_gamma = sum(gamma[t][i] for t in range(T - 1))
            for j in range(N_state):
                new_A[i][j] = sum(di_gamma[t][i][j] for t in range(T - 1)) / (sum_gamma + epsilon)

            sum_gamma += gamma[T - 1][i]
            for k in range(M):
                new_B[i][k] = sum(gamma[t][i] for t in range(T) if obs_sq[t] == k) / (sum_gamma + epsilon)
        
        self.trans_M.input_list = [a for row in new_A for a in row]
        self.trans_M.is_input_list_changed = True
        self.trans_M.trans_to_matrice()

        self.obs_M.input_list = [b for row in new_B for b in row]
        self.obs_M.is_input_list_changed = True
        self.obs_M.trans_to_matrice()

    def model_para_learn(self, obs : Matrix, max_iter = 50):
        iter_cnt = 0
        log_prob = 1
        log2 = 0
        prev_log_prob = -math.inf
        while iter_cnt < max_iter and log_prob > prev_log_prob:
            print("iter_cnt: "+ str(iter_cnt))
            print("log_prob: "+ str(log_prob))
            print("P(obs): " + str(log2))
            iter_cnt += 1
            if iter_cnt != 1:
                prev_log_prob = log_prob
            _, c = self.alpha_pass(obs)
            self.re_estimate(obs)
            log_prob = compute_log(c)
            log2 = compute_log2(c)

    def alpha_pass_no_scale(self, Osq : Matrix):
        # calculate alpha_0
        N_state = self.n_states
        obs_0 = Osq.get_input_list()[0]
        n_obs = Osq.get_n_col()
        pi = self.init_state.trans_to_matrice()
        B = self.obs_M.trans_to_matrice()
        A = self.trans_M.trans_to_matrice()
        alpha = []
        alpha_vector = []

        for i in range(N_state):
            alpha.insert(i, pi[0][i] * B[i][obs_0])
        alpha_vector.insert(0, alpha)

        # calculate t = 1 ~ T-1
        for t in range(1, n_obs):
            alpha = []
            obs_t = Osq.get_input_list()[t]
            alpha_t_1 = alpha_vector[t - 1]

            for i in range(N_state):
                Sum = 0
                for j in range(N_state):
                    Sum += alpha_t_1[j] * A[j][i]
                alpha.insert(i, Sum * B[i][obs_t])
            
            alpha_vector.insert(t, alpha)

        alpha_T = alpha_vector[n_obs - 1]
        Result = sum(alpha_T)
        return Result

def compute_log(c: list):
    return -sum(math.log(c_t) for c_t in c)

def compute_log2(c: list):
    #print(c)
    return 1/c[-1]