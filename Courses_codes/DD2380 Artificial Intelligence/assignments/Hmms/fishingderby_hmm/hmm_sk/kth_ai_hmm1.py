#!/usr/bin/python3

"""
Input:

4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0 
4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9 
1 4 1.0 0.0 0.0 0.0 
8 0 1 2 3 0 1 2 3 

Output:

0.090276 
"""

class Matrix:

    def __init__(self, input_row, isvector = False):
        if isvector:
            self.n_row = 1
            self.n_col = int(input_row[0])
            self.input_list = [float(x) if type(x) == type('a') else x for x in input_row[1:]]
        else:
            self.n_row = int(input_row[0])
            self.n_col = int(input_row[1])
            self.input_list = [float(x) if type(x) == type('a') else x for x in input_row[2:]]
        if self.n_row == 1:
            self.is_vector = True
        else:
            self.is_vector = False
        self.trans_matrice = None

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
        if self.is_vector:
            self.trans_matrice = self.input_list
        else:
            self.trans_matrice = []
            for i in range(self.n_row):
                row = []
                for j in range(self.n_col):
                    row.append(self.input_list[i*self.n_col + j])
                self.trans_matrice.append(row)
        return self.trans_matrice
    
class Observe_seq:

    def __init__(self, input_row):
        self.n_obs = int(input_row[0])
        self.obs_list = [int(x) for x in input_row[1 : ]]

def alpha_pass(A : Matrix, B : Matrix, pi : Matrix, o_seq : Observe_seq):
    
    B_matrix = B.trans_to_matrice()
    A_matrix = A.trans_to_matrice()
    
    # calculate alpha_0
    N_state = B.n_row
    obs_0 = o_seq.obs_list[0]
    alpha = []
    alpha_vector = []

    for i in range(N_state):
        alpha.insert(i, pi.input_list[i] * B_matrix[i][obs_0])
    alpha_vector.insert(0, alpha)

    # calculate t = 1 ~ T-1
    for t in range(1, o_seq.n_obs):
        alpha = []
        obs_t = o_seq.obs_list[t]
        alpha_t_1 = alpha_vector[t - 1]

        for i in range(N_state):
            sum = 0
            for j in range(N_state):
                sum += alpha_t_1[j] * A_matrix[j][i]
            alpha.insert(i, sum * B_matrix[i][obs_t])
        
        alpha_vector.insert(t, alpha)
        
    return alpha_vector

if __name__ == '__main__':
    A = Matrix(input("").split())
    B = Matrix(input("").split())
    pi = Matrix(input("").split())
    obs_seq = Observe_seq(input("").split())

    alpha_vector = alpha_pass(A, B, pi, obs_seq)

    alpha_T = alpha_vector[obs_seq.n_obs - 1]

    print(sum(alpha_T))
    
