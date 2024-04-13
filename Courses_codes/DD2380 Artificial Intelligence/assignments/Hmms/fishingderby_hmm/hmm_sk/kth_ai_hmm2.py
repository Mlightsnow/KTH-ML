from numpai import Matrix, HMM

if __name__ == '__main__':
    # a = "4 4 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.1 0.1 0.1 0.0 0.8 0.8 0.1 0.1 0.0".split(' ')
    # b = "4 4 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.0 0.0 0.0 0.9 0.1 0.1 0.0 0.0 0.9".split(' ')
    # p = "1 4 1.0 0.0 0.0 0.0".split(' ')
    # O_sq = "4 1 1 2 2".split(' ')
    A = Matrix(input().split())
    B = Matrix(input().split())
    pi = Matrix(input().split())
    O_squence = Matrix(input().split(), isvector=True)
    
    Model = HMM(A,B,pi)
    result_list = Model.alpha_pass(O_squence)
    alpha_T = result_list[-1]
    print(sum(alpha_T))
    #print(' '.join(map(str, result_list)))
