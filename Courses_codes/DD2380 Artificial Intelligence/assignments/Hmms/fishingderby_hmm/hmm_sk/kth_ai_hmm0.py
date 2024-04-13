from numpai import Matrix

if __name__ == '__main__':

    A = Matrix([4, 4, 0.2, 0.5, 0.3, 0.0, 0.1, 0.4, 0.4, 0.1, 0.2, 0.0, 0.4, 0.4, 0.2, 0.3, 0.0, 0.5])
    B = Matrix([4, 3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.6, 0.2])
    pi = Matrix([1, 4, 0.0, 0.0, 0.0, 1.0])

    next_state = pi.multiply(A)

    observation = next_state.multiply(B)

    output = [observation.n_row, observation.n_col]
    for x in observation.input_list:
        output.append(x)

    print(' '.join(map(str, output)))

