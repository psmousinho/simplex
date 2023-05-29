import sys
import argparse
import numpy as np
from functools import reduce
BIG_M = 100000
EPSILON = sys.float_info.epsilon

def readInstance(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        
        n_var = int(lines[0].split()[-1]) # read number o variables
        n_res = int(lines[1].split()[-1]) # read number of restrictions
        
        # read objective funtion
        OF_str = lines[2].split()
        min = (OF_str[1] == "MIN")
        obj_function = [float(OF_str[i]) for i in range(2 , len(OF_str))] 
        
        # read restrictions
        restrictions_ls = []
        res_signes = []
        b_vec = []
        for line in lines[4:4 + n_res]:
            split_line = line.split()
            row = []
            for i in range(len(split_line) - 2):
                row.append(float(split_line[i]))
            restrictions_ls.append(row)
            res_signes.append(split_line[-2])
            b_vec.append(float(split_line[-1]))
            
        # read variabels` bounds restrictions
        bounds_res = []
        for line in lines[4 + n_res : ]:
            split_line = line.split()
            if(split_line[1] == "livre"):
                bounds_res.append([split_line[1]])
            else:
                bounds_res.append([split_line[1], float(split_line[2])])
        
        # transform MIN z into MAX -z, if necessary
        if (min): 
            obj_function = [-1 * i for i in obj_function]
        
        # transform restrictions to standard form 
        A_mat = []
        identity = np.identity(n_res, dtype=int).tolist()
        base_var_indexes = []
        artificial_var_indexes = []
        eq_sign_res_indexes= []
        geq_sign_res_indexes = []
        for row in range(n_res):
            match res_signes[row]:
                case "<=": # add slack var
                    A_mat.append(restrictions_ls[row] + identity[row])
                    obj_function.append(0)
                    base_var_indexes.append(n_var + row) # put var in base
                case "=": # add artifical var
                    A_mat.append(restrictions_ls[row] + identity[row])
                    obj_function.append(- BIG_M)
                    artificial_var_indexes.append(row)
                    base_var_indexes.append(n_var + row) #put var in base
                    eq_sign_res_indexes.append(row)
                case ">=": # add surplus var
                    A_mat.append(restrictions_ls[row] + ([-1 * i for i in identity[row]]))
                    obj_function.append(0)
                    artificial_var_indexes.append(row)
                    geq_sign_res_indexes.append(row)
        
        # add artificial var for ">=" restrictions 
        for index in geq_sign_res_indexes:
            new_col = n_res * [0] 
            new_col[index] = 1 
            obj_function.append(- BIG_M)
            base_var_indexes.append(len(A_mat[row])) # put var in base
            for row in range(n_res):
                A_mat[row].append(new_col[row])
                
    printProblem(n_var, n_res, min, obj_function, A_mat, res_signes, b_vec, bounds_res)
    printDualProblem(n_var, n_res, min, obj_function, A_mat, res_signes, b_vec, bounds_res)
    
    return n_var, n_res, min, obj_function, A_mat, b_vec, artificial_var_indexes, base_var_indexes, eq_sign_res_indexes

def printProblem(n_var, n_res, min, obj_function, A_mat, res_signals, b_vec, bounds_res):
    print("Problema Primal:")
    if(min):
        print("\tMin z = ", end="")
        obj_function = [-i for i in obj_function]
    else:
        print("\tMax z = ", end="")
        
    for i in range(n_var):
        print("{} X{} ".format(obj_function[i], i+1), end="")
    print("\n\t st")

    for i in range(n_res):
        print("\t", end="")
        for j in range(n_var):
            print("{} X{} ".format(A_mat[i][j], j+1), end="")
        print(res_signals[i], end=" ")
        print("{}".format(b_vec[i]))
        
    for i in range(n_var):
        print("\t", end="")
        if (bounds_res[i][0] == "livre"):
            print("X{} {}".format(i+1, bounds_res[i][0]))
        else:
            print("X{} {} {}".format(i+1, bounds_res[i][0], bounds_res[i][1]))
    print("\n")
        
def printDualProblem(n_var, n_res, min, obj_function, A_mat, res_signals, b_vec, bounds_res):
    print("Problema Dual:")
    if(not min):
        print("\tMin z = ", end="")
    else:
        print("\tMax z = ", end="")
        
    for i in range(len(b_vec)):
        print("{} Y{} ".format(b_vec[i], i+1), end="")
    print("\n\t st")

    for i in range(n_var):
        print("\t", end="")
        for j in range(n_res):
            print("{} Y{} ".format(A_mat[j][i], j+1), end="")
        match bounds_res[i][0]:
            case ">=":
                print(">=", end=" ")
            case "livre":
                print("=", end=" ")
            case "<=":
                print("<=", end=" ") 
        print("{}".format(obj_function[i]))
        
    for i in range(n_res):
        match res_signals[i]:
            case "=":
                print("\t\tY{}, livre".format(i+1))
            case "<=":
                print("\t\tY{} >= 0".format(i+1, res_signals[i]))    
            case ">=":
                print("\t\tY{} <= 0".format(i+1, res_signals[i]))
    print("\n")   

def isOptimal(obj_function):
    for i in range(len(obj_function) - 1):
        if(obj_function[i] < -EPSILON):
            return False
            
    return True

def selectColumn(obj_function):
    min = 0
    index = 0
    for i in range(len(obj_function) - 1):
        if(obj_function[i] < min):
            min  = obj_function[i]
            index = i
            
    return index

def selectRow(tableau, column):
    b_index = len(tableau[0]) -1
    min = sys.maxsize
    index = 0
    
    for i in range(1,len(tableau)):
        if(tableau[i][column] > EPSILON and tableau[i][b_index]/tableau[i][column] < min):
            min  = tableau[i][b_index]/tableau[i][column]
            index = i
            
    return index

def updateTableau(tableau, column, row):
    tableau[row] = tableau[row]/tableau[row][column]
    for i in range(len(tableau)):
        if (i == row):
            continue
        tableau[i] = (-1 * tableau[i][column] * tableau[row])  + tableau[i]
                
    return tableau

def printTableau(base_var_indexes, tableau):
    n_row = len(tableau)
    n_col = len(tableau[0])
    
    print("\t{:^7} ".format(""), end="")
    for i in range(n_col - 1):
            print("| {:^5} ".format("X" + str(i+1)), end="")
    print("| {:^5} |".format("b"))
    
    for i in range(n_row):
        if(i == 0):
            print("\t| {:^5} ".format("z"), end="")
        else:
            print("\t| {:^5} ".format("X" + str(base_var_indexes[i -1] + 1)), end="")
            
        for j in range(n_col):
            print("| {:^5.2f} ".format(tableau[i][j]), end="")
        print("|")

def printBasicVariables(base_var_index, tableau):
    b_index = len(tableau[0]) - 1
    print("\t", end="")
    for i in range(len(base_var_index)):
        print("| X{} = {:5.2f} ".format(base_var_index[i] + 1, tableau[ i + 1][b_index]), end="")
    print("| ")

def printDualVariables(n_var, n_res, obj_function, eq_sign_res_indexes):
    print("\t", end="")
    for i in range(n_var, n_var + n_res): # for every slack or surplus variable
        aux = 0
        if (i - n_var in eq_sign_res_indexes):
            aux = BIG_M
        print("| Y{}={:5.2f} ".format(i - n_var + 1, obj_function[i] - aux), end="")
    print("|")
    
def printBRange(n_var, n_res, b_vec, tableau):
    b_column = len(tableau[0]) -1

    limits = []
    for col in range(n_var, n_var + n_res): # for every slack or surplus variable
        var_limits = [float('-inf'), float('inf')]
        for row in range(1, len(tableau)): # for every restriction
            if (tableau[row][col] != 0):    
                new_limit = -tableau[row][b_column] / tableau[row][col]
                print("linha " + str(row))
                print("col " + str(col))
                print(new_limit)
                if (tableau[row][col] > EPSILON):
                    if (new_limit > var_limits[0]):
                        var_limits[0] = new_limit
                if (tableau[row][col] < -EPSILON):
                    if (new_limit < var_limits[1]):
                        var_limits[1] = new_limit
        limits.append(var_limits)
            
    for i in range(len(limits)):
        print ("\t{:.2f} <= delta_b{} <= {:.2f}, portanto ".format(limits[i][0], i+1, limits[i][1]), end="")
        print ("{:.2f} <= b{} <= {:.2f}".format(b_vec[i] + limits[i][0], i+1, b_vec[i] + limits[i][1]))

def simplex(n_var, n_res, minimize, obj_function, A_mat, b_vec, artificial_var_indexes, base_var_indexes, eq_sign_res_indexes, interactive=False):
    print("\nSimplex:")

    # set basic variables to 0 in objective funtion 
    obj_function = np.array([-1 * i for i in obj_function] + [0])
    if (artificial_var_indexes):
        for index in artificial_var_indexes:
            obj_function = obj_function - (BIG_M * np.array(A_mat[index] + [b_vec[index]]))

    tableau = []
    tableau.append(obj_function.tolist())
    for row in range(n_res):
        tableau.append(A_mat[row] + [b_vec[row]])

    tableau = np.array(tableau, dtype=float)

    iter = 1
    print("Tableau original:")
    printTableau(base_var_indexes, tableau)
    while(not isOptimal(tableau[0])):
        column = selectColumn(tableau[0])
        row = selectRow(tableau, column)
        tableau = updateTableau(tableau, column, row)
        print("\n\tColuna pivo:{} | Linha pivo:{}".format(column + 1, row + 1))
        base_var_indexes[row - 1] = column
        print("\nIteração {}:".format(iter))
        printTableau(base_var_indexes, tableau)
        iter += 1
        
        if(interactive):
            input("\n Para continuar aperte enter")

    if(minimize):
        tableau[0][len(tableau[0])-1] = - tableau[0][len(tableau[0])-1]
    
    print("\n")
    print("Execução encerrada!")
    print("O valor otimo é: {:5.2f}".format(tableau[0][len(tableau[0])-1]))
    print("Os valores otimos das variaveis basicas são:")
    printBasicVariables(base_var_indexes, tableau) 
    print("Os valores otimos das variaveis duais são: ")
    printDualVariables(n_var, n_res, tableau[0], eq_sign_res_indexes)
    print("Os limites de variações do vetor b são: ")
    printBRange(n_var, n_res, b_vec, tableau)
     
def revisedSimplex(n_var, n_res, minimize, obj_function, A_mat, b_vec, artificial_var_indexes, base_var_indexes, eq_sign_res_indexes, interactive=False):
    print("\nSimplex revisado:")
    
    # transforme para np.array
    c_vec = np.array(obj_function)
    A_mat = np.array(A_mat)
    b_vec = np.array(b_vec)
    B_indexes = np.array(base_var_indexes)
    N_indexes = np.array([ i for i in range(len(A_mat[0])) if i not in B_indexes])

    get_from_c = lambda c, indexes : np.array([c[i] for i in indexes])
    get_from_A = lambda A, indexes : np.array([[A[row][col] for col in indexes] for row in range(len(A))] )

    #1 Dadoes previos
    cB_vec = get_from_c(c_vec,B_indexes)
    cN_vec = get_from_c(c_vec,N_indexes)
    B_mat = get_from_A(A_mat, B_indexes)
    N_mat = get_from_A(A_mat, N_indexes)
    B_inv_mat = np.linalg.inv(B_mat)
    
    iter = 1
    while(True):
        print("\nIteração {}:".format(iter))
        iter += 1

        print("Matriz B:\n {}".format(B_mat))
        
        # Valor atual das variaveis e da funcao objetivo           
        u_vec = np.matmul(cB_vec, B_inv_mat)
        xB_vec = np.matmul(B_inv_mat, b_vec)
        print("Valor atual das variaveis basicas xB: ", end="")
        for i in range(len(B_indexes)):
            print("X{} = {:5.2f} | ".format(B_indexes[i]+1, xB_vec[i]), end="")
        print()
        
        # Calcule custos reduzidos
        z_value = np.matmul(u_vec, b_vec)
        redux_cost_vec =  np.subtract(np.matmul(u_vec, N_mat) , cN_vec)
        print("Valor objetivo atual: {:5.2f}".format(z_value))
        
        # Ache a variavel N que vai entrar em B
        min = 0
        incoming_var_index = -1
        for i in range(len(redux_cost_vec)):
            if (redux_cost_vec[i] < min):
                incoming_var_index = i
                min = redux_cost_vec[i]
        
        # Se achou o otimo, pare
        if (incoming_var_index == -1):

            # Print valor minimo
            if(minimize):
                z_value = -z_value
            print("\nExecução encerrada!")
            print("O valor otimo é: {:5.2f}".format(z_value))
            
            # Print variaveis basicas
            print("Os valores otimos das variaveis basicas são: \n\t", end="")
            for i in range(len(B_indexes)):
                print("| X{} = {:5.2f} ".format(B_indexes[i]+1, xB_vec[i]), end="")
            print("|")
            
            # Print variaveis duais
            print("Os valores otimos das variaveis duais são: \n\t", end="")
            for i in range(len(u_vec)):
                aux = 0
                if (n_var + i  in eq_sign_res_indexes):
                    aux = BIG_M
                print("| Y{}={:5.2f} ".format(i +1, u_vec[i] - aux), end="")
            print("|")
            
            #Print b ranges
            print("Os limites de variações do vetor b são: ")
            limits = []
            for col in range(n_res):
                var_limits = [float('-inf'), float('inf')]
                for row in range(0, len(B_inv_mat[0])):
                    if (B_inv_mat[row][col] != 0):
                        new_limit = -xB_vec[row]/B_inv_mat[row][col]
                        if (B_inv_mat[row][col] > EPSILON):
                            if (new_limit > var_limits[0]):
                                var_limits[0] = new_limit
                        if (B_inv_mat[row][col] < -EPSILON):
                            if (new_limit < var_limits[1]):
                                var_limits[1] = new_limit            
                limits.append(var_limits)
                    
            for i in range(len(limits)):
                print ("\t{:.2f} <= delta_b{} <= {:.2f}, portanto ".format(limits[i][0], i+1, limits[i][1]), end="")
                print ("{:.2f} <= b{} <= {:.2f}".format(b_vec[i] + limits[i][0], i+1, b_vec[i] + limits[i][1]))
                    
            break
        else:
            print("Variavel X{} vai entrar na base".format(N_indexes[incoming_var_index] + 1))
            
            # Se nao achou o valor otimo, ache a variavel que vai sair de B
            y_vec = np.matmul(B_inv_mat, get_from_A(A_mat, [N_indexes[incoming_var_index]]))
            

            if (reduce((lambda x,y : x or y),map((lambda x : x > EPSILON ), y_vec))):
                alpha_min = float("inf")
                outgoing_var_index = -1
                for i in range(len(xB_vec)):
                    if(y_vec[i][0] > EPSILON and xB_vec[i]/y_vec[i][0] < alpha_min):
                        print(xB_vec[i])
                        print(y_vec[i][0])
                        outgoing_var_index = i
                        alpha_min =  xB_vec[i]/y_vec[i][0]
                    
                print("Variavel X{} vai sair da base".format(B_indexes[outgoing_var_index] + 1))
                
                
                # Atualize B-1
                E_mat_inv = np.identity(len(B_mat))
                for row in range(len(E_mat_inv)):
                    if(row == outgoing_var_index):
                        E_mat_inv[row][outgoing_var_index] = 1/y_vec[outgoing_var_index][0]
                    else:    
                        E_mat_inv[row][outgoing_var_index] = - y_vec[row][0]/y_vec[outgoing_var_index][0]          
                B_inv_mat = np.matmul(E_mat_inv, B_inv_mat)
                
                # Atualize a base
                aux = B_indexes[outgoing_var_index]
                B_indexes[outgoing_var_index] = N_indexes[incoming_var_index]
                N_indexes[incoming_var_index] = aux
                
                # Atualize dados
                cB_vec = get_from_c(c_vec,B_indexes)
                cN_vec = get_from_c(c_vec,N_indexes)
                B_mat = get_from_A(A_mat, B_indexes)
                N_mat = get_from_A(A_mat, N_indexes)
                
                if(interactive):
                    input("\n Para continuar aperte enter")
            else:
                print("A soluçao deste problema ́e ilimitada!!")
                break


parser = argparse.ArgumentParser()
parser.add_argument('-r', action='store_true')
parser.add_argument('-i', action='store_true')
parser.add_argument('filepath', type=str)
args = parser.parse_args()

n_var, n_res, min, obj_function, A_mat, b_vec, artificial_var_indexes, base_var_indexes, eq_sign_res_indexes = readInstance(args.filepath)

print("Dados:")
print("\tN_VAR: {}".format(n_var))
print("\tN_RES: {}".format(n_res))
print("\tOBJ_FUN: {}".format(obj_function))
print("\tA:{}:".format(A_mat))
print("\tb: {}:".format(b_vec))

if (args.r):
    revisedSimplex(n_var, n_res, min, obj_function, A_mat, b_vec, artificial_var_indexes, base_var_indexes, eq_sign_res_indexes, args.i)
else:
    simplex(n_var, n_res, min, obj_function, A_mat, b_vec, artificial_var_indexes, base_var_indexes, eq_sign_res_indexes, args.i)