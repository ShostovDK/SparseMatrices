import os
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import psutil
import numpy as np

def generate_sss_matrix(n, density):
    """
    Генерирует разреженную симметричную матрицу в формате SSS.

    :param n: размер матрицы
    :param density: плотность матрицы
    :return: сгенерированная матрица в формате SSS
    """
    mat = sparse.random(n, n, density=density, format='coo')
    mat = mat + mat.transpose()  # делаем матрицу симметричной
    mat_coo = mat.tocoo()
    mat_data = mat_coo.data[mat_coo.row != mat_coo.col]
    mat_rows = mat_coo.row[mat_coo.row != mat_coo.col]
    mat_cols = mat_coo.col[mat_coo.row != mat_coo.col]
    mat_sss = sparse.coo_matrix((mat_data, (mat_rows, mat_cols)), shape=(n, n))
    mat_sss.setdiag(mat.diagonal())
    return mat_sss

# Считываем размер и плотность матриц с клавиатуры
n = int(input("Введите размер матриц: "))
density = float(input("Введите плотность матриц: "))

# Генерируем две разреженные симметричные матрицы в формате SSS
mat1 = generate_sss_matrix(n, density)
mat2 = generate_sss_matrix(n, density)

# Перемножаем матрицы с измерением используемой памяти
start_mem = psutil.Process(os.getpid()).memory_info().rss
mat3 = mat1 * mat2
end_mem = psutil.Process(os.getpid()).memory_info().rss
mem_usage = (end_mem - start_mem) / 1024 / 1024

print(f"Использовано памяти: {mem_usage:.2f} МБ")

# Преобразуем результат в формат CSR и выводим его
mat3_csr = mat3.tocsr()
print(mat3_csr)