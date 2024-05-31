import numpy as np
import scipy.sparse as sparse

def generate_symmetric_sparse_matrix(size, density, diag_val=None):
    # Генерируем симметричный набор индексов
    i, j = np.triu_indices(size, k=1)
    # Выбираем случайный поднабор индексов в соответствии с заданной плотностью
    num_entries = int(density * size * (size - 1) / 2)
    indices = np.random.choice(len(i), size=num_entries, replace=False)
    i = i[indices]
    j = j[indices]
    # Генерируем случайные данные для выбранных индексов
    data = np.random.randint(1, 10, len(i)) 
    # Создаем разреженную матрицу из данных
    matrix = sparse.csr_matrix((data, (i, j)), shape=(size, size))
    # Если задано значение диагонали, устанавливаем его
    if diag_val is not None:
        matrix.setdiag(diag_val * np.ones(size))
    # Делаем матрицу симметричной
    matrix += matrix.T
    # Генерируем случайное целое число для диагонали и устанавливаем его, если значение не было задано
    if diag_val is None:
        diag_val = np.random.randint(1, 10)
        matrix.setdiag(diag_val * np.ones(size))
    return matrix

def input_symmetric_matrix(size):
    # Создаем пустую симметричную матрицу
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i, size):
            while True:
                try:
                    value = int(input(f"Enter value for matrix[{i}][{j}] (matrix[{j}][{i}] will be the same): "))
                    break
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            matrix[i][j] = value
            matrix[j][i] = value
    return sparse.csr_matrix(matrix)

def main():
    # Спрашиваем у пользователя размер и плотность матриц
    while True:
        try:
            size = int(input("Enter the size of the matrices: "))
            if size > 0:
                break
            else:
                print("Invalid input. Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    while True:
        try:
            density = float(input("Enter the density of the matrices (between 0 and 1): "))
            if 0 <= density <= 1:
                break
            else:
                print("Invalid input. Please enter a value between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Спрашиваем у пользователя, хочет ли он генерировать или вводить матрицы
    while True:
        generate_input = input("Do you want to generate or input the matrices? (g/i): ").lower()
        if generate_input in ["g", "i"]:
            break
        else:
            print("Invalid input. Please enter 'g' or 'i'.")

    if generate_input == "g":
        matrix1 = generate_symmetric_sparse_matrix(size, density)
        matrix2 = generate_symmetric_sparse_matrix(size, density, matrix1.diagonal()[0])
    elif generate_input == "i":
        matrix1 = input_symmetric_matrix(size)
        while True:
            matrix2_input = input("Do you want to input the second matrix or generate it with the same diagonal as the first matrix? (i/g): ").lower()
            if matrix2_input in ["i", "g"]:
                break
            else:
                print("Invalid input. Please enter 'i' or 'g'.")
        if matrix2_input == "i":
            matrix2 = input_symmetric_matrix(size)
        else:
            matrix2 = generate_symmetric_sparse_matrix(size, density, matrix1.diagonal()[0])

    result = matrix1.dot(matrix2)
    # Выводим матрицы в формате SSS без десятичной точки
    print("Matrix 1 in SSS format:")
    print(np.array2string(matrix1.toarray(), formatter={'float_kind':lambda x: "%d" % x}))
    print("Matrix 2 in SSS format:")
    print(np.array2string(matrix2.toarray(), formatter={'float_kind':lambda x: "%d" % x}))
    print("Result in SSS format:")
    print(np.array2string(result.toarray(), formatter={'float_kind':lambda x: "%d" % x}))
    # Выводим матрицы в формате CSR 
    print("Result in CSR format:")
    print(result)

if __name__ == "__main__":
    main()
