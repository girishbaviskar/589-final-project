import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x)))

def forward_propagation(x, weights, number_of_layers, debug_mode):
    if debug_mode: print('Forward propagating the input', x)
    L = number_of_layers
    A = {}  # dictionary to store the activations of each layer
    Z = {}  # dictionary to store the linear combinations of each layer
    # Step 1a: Set a1 = x
    x_reshaped = x.reshape(-1, 1)
    A['a1'] = x
    # Step 2: Add bias term to a1
    A['a1'] = np.vstack((np.ones((1, 1)), x_reshaped))
    if debug_mode: print('a1 : ', A['a1'])
    # Step 3: Loop over layers 2 to L-1
    for l in range(2, L):
        # Step 3.1: Compute z(l)
        Z['z' + str(l)] = np.dot(weights['theta' + str(l - 1)], A['a' + str(l - 1)])
        if debug_mode: print('z' + str(l), Z['z' + str(l)])
        # Step 3.2: Compute a(l) = g(z(l))
        A['a' + str(l)] = sigmoid(Z['z' + str(l)])
        # Step 3.3: Add bias term to a(l)
        A['a' + str(l)] = np.vstack((np.ones((1, 1)), A['a' + str(l)]))
        if debug_mode: print('a' + str(l), A['a' + str(l)])
    # Step 4: Compute z(L)
    Z['z' + str(L)] = np.dot(weights['theta' + str(L - 1)], A['a' + str(L - 1)])
    if debug_mode: print('z' + str(L), Z['z' + str(L)])
    # Step 5: Compute a(L) = g(z(L))
    A['a' + str(L)] = sigmoid(Z['z' + str(L)])
    if debug_mode: print('a' + str(L), A['a' + str(L)])
    # Step 6: Return f_theta(x(i)) = a(L)
    aL = A['a' + str(L)]
    return aL, A, Z


def back_propagation(X, Y, A, weights, alpha, reg_lambda, number_of_layers, batch_size, acc_grads, debug_mode):
    if isinstance(Y, np.float64) or isinstance(Y, np.int64):
        m = 1
    else:
        m = len(Y)
    L = number_of_layers
    deltas = {}
    grads = {}

    # Compute output layer delta
    deltas[f"delta{L}"] = A[f"a{L}"] - Y.reshape((m, 1))
    if debug_mode: print('delta', L, deltas[f"delta{L}"])
    # Compute delta for hidden layers
    for l in range(L - 1, 1, -1):
        secTerm = A[f"a{l}"] * (1 - A[f"a{l}"])

        if len(weights[f"theta{l}"].shape) == 1:  # if input is a 1D array
            trans = weights[f"theta{l}"].reshape(-1, 1)  # reshape it to a column array
        else:  # if input is an n-dimensional matrix
            trans = weights[f"theta{l}"].T  # transpose the matrix

        dlt = deltas[f"delta{l + 1}"]
        firstTerm = np.dot(trans, dlt)
        deltas[f"delta{l}"] = firstTerm * secTerm
        deltas[f"delta{l}"] = deltas[f"delta{l}"][1:]
        if debug_mode : print('delta', l, deltas[f"delta{l}"] )
    # Compute gradients for all layers
    for l in range(L - 1, 0, -1):
        grads[f"dW{l}"] = np.dot(deltas[f"delta{l + 1}"], A[f"a{l}"].T)
        if debug_mode: print('Gradients of Theta', l, 'based on training instance : ' , grads[f"dW{l}"])
        if f"dW{l}" in acc_grads:
            acc_grads[f"dW{l}"] += grads[f"dW{l}"]
        else:
            acc_grads[f"dW{l}"] = grads[f"dW{l}"]

    return acc_grads


def calculate_reg_gradients(grads, weights, reg_lambda, L, n, debug_mode):
    P = {}
    D_reg = {}
    for k in range(L - 1, 0, -1):
        # Check if weights[f'theta{k}'] is a 1D array
        if weights[f'theta{k}'].ndim == 1:
            P[f'P{k}'] = np.hstack([0, reg_lambda * weights[f'theta{k}'][1:]])
        else:
            P[f'P{k}'] = np.hstack(
                [np.zeros((weights[f'theta{k}'].shape[0], 1)), reg_lambda * weights[f'theta{k}'][:, 1:]])
        D_reg[f'D{k}'] = (1 / n) * (grads[f'dW{k}'] + P[f'P{k}'])
        grads[f'dW{k}'] = D_reg[f'D{k}']
        if debug_mode: print('Final regularized gradients of theta', k , grads[f'dW{k}'])
    return grads


def update_weights(weights, grads, L, alpha):
    for l in range(L - 1, 0, -1):
        if len(weights[f"theta{l}"].shape) == 1:  # 1D array
            temp = alpha * grads[f"dW{l}"].reshape(-1)
            weights[f"theta{l}"] -= temp
        else:  # 2D array
            weights[f"theta{l}"] -= alpha * grads[f"dW{l}"]
    return weights

def make_predictions(X, weights, n_layers, debug_mode):
    pred_list = []
    list_J_all_x = []
    for i, x in enumerate(X):
        aL, A, Z = forward_propagation(x, weights, n_layers, debug_mode)
        output = np.argmax(aL)
        pred_list.append(output)
    return pred_list

def calculate_accuracy(listOfPredictedLabels, listOfActualLabels):
    correct_predictions = 0
    for index, label in enumerate(listOfPredictedLabels):
        # print(label)
        if label == listOfActualLabels[index]:
            correct_predictions += 1
    return ((correct_predictions / len(listOfPredictedLabels)))

