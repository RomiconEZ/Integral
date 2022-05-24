# from builtins import function

import numpy as np
from scipy import integrate
from scipy.special import gammaln
from math import comb

# Степени и пределы интегрирования
alpha = 1 / 5
betta = 0
a = 0.1
b = 2.3
# alpha = 2 / 3 # 9
# betta = 0 # 9
# a = 0.7 # 9
# b = 3.2 # 9
M_n = 23.0382 # Максимум производной от F(x) на искомом промежутке 3
# M_n = 2.3684 # Максимум производной от F(x) на искомом промежутке 9

S_h_s =np.empty((2,0)) # массив посчитанных значений кдваратурной формы для заданного шага

def Gauss(N_: int = 3, a_: float = a, b_: float = b):
    mU = []
    # 1 Вычисляем моменты весовой функции p(x) на [a,b]
    mU = mU_i_s(a_, b_, s=N_ - 1, alpha_=alpha)[::-1]
    mU_n_plus_s = map(lambda x: -x, mU[N_:2 * N_])
    # 2 Решаем СЛАУ
    mU_j_plus_s = np.zeros((N_, N_))
    for i in range(0, N_):
        for j in range(0, N_):
            mU_j_plus_s[i, j] = mU[i + j]

    a_i_j = np.linalg.solve(mU_j_plus_s, mU_n_plus_s)
    # Находим узлы, как корни узлового многочлена
    # Добавить единичку в a_i_j
    x_j = np.roots(a_i_j.transpose())
    A = np.array([np.power(x_j, i) for i in range(0, N_)])
    return np.linalg.solve(A, mU[0:N_])


# f(x)
def f(x: np.float_) -> np.float_:
    return 2.5 * np.cos(2. * x) * np.exp(2. * x / 3.) + \
           4. * np.sin(3.5 * x) * np.exp(-3.0 * x) + 3 * x


def p(x: np.float_) -> np.float_:
    return 1 / (np.power(x - a, -alpha) * np.power(b - x, -betta))


def F(x: np.float_) -> np.float_:
    return f(x) / (np.power(x - a, -alpha) * np.power(b - x, -betta))


def mU_i_s(a_: float, b_: float, s: int = 0, alpha_: float = alpha):
    """
    Рекурсивное вычисление интеграла по промежутку [a_, b_] от функции (x^2/(x-a)^alpha)
    Parameters
    ----------
    :param a_: нижний предел интегрирования
    :param b_: вернхний предел интегрирования
    :param s: степень x
    :param alpha_: то же, что и в основной функции
    :return: list
    """
    global a
    if s == 0:
        return [(pow((b_ - a), 1 - alpha_) - pow((a_ - a), 1 - alpha_)) / (1 - alpha_)]
    else:
        res = (pow((b_ - a), s + 1 - alpha_) - pow((a_ - a), s + 1 - alpha_)) / (s + 1 - alpha_)
        mUs = mU_i_s(a_, b_, s=s - 1)
        l_ = len(mUs)
        for num, value in enumerate(mUs):
            res += comb(s, num + 1) * pow(-1, num) * pow(a, num + 1) * mUs[num]
        return [res] + mUs


# Интегрирвоание через SciPy
# exact, err = integrate.quad(func=F, a=a, b=b)
# print(exact)

# Task 1.1
# Построить интерполяционную квадратурную формулу с весо-
# вой функцией p(x) = (x − a)^(−α) (b − x)^(−β) на отрезке [a, b] по
# трём равномерно распределённым узлам x_1 = a, x_2 = (a + b)/2,
# x_3 = b. Оценить методическую погрешность построенного пра-
# вила (11), сравнить её с точной погрешностью.

# Осталось сделать
#   1. Оценить методическую погрешность
#   2. Сравнить методическую погрешность с точной

TARGET = 3.578861536040539915439859609644293194417  # Точное значение интеграла 3
# TARGET = 20.73027110955223102601793414048307154080  # Точное значение интеграла 9
print(TARGET)


def newton_cotes(N_: int = 3, h_: int = -1,
                 a_: float = a, b_: float = b):
    """
    :param N_: количество отрезков
    :param h_: шаг. Елси задан, то используется он
    :param a_: нижний предел интегрирования
    :param b_: верхний предел интегрирования
    :return:
    """
    mU = []
    # Задаём узлы квадратурной формулы
    if h_ != -1:
        nodes_x = np.arange(a_, b_ + h_, h_)
    else:
        nodes_x = np.linspace(a_, b_, N_)
    # Вычисляем моменты весовой функции p(x) на [a,b]

    mU = mU_i_s(a_, b_, s=len(nodes_x) - 1, alpha_=alpha)[::-1]
    # Решаем СЛАУ
    mU = np.array(mU)
    A = [np.power(nodes_x, i) for i in range(0, len(nodes_x))]

    An = np.linalg.solve(A, mU)
    quad = np.sum(An * f(nodes_x))

    return quad


quad = newton_cotes()
error = abs(quad - TARGET)
methodic_error = M_n
print('{:10d}  {:10.10f}  {:.10e}'.format(3, quad, error))


# Task 1.2
# На базе построенной малой ИКФ построить составную КФ и,
# уменьшая длину шага h, добиться заданной точности ε = 10 −6 .
# Погрешность оценивать методом Ричардсона. На каждых по-
# следовательных трёх сетках оценивать скорость сходимости по
# правилу Эйткена.
def Aitken_process(method, h__: float = abs(b - a) / 3, L: float = 2, a_: float = a, b_: float = b):

    h3 = h__ / np.power(L, 2)
    if np.size(S_h_s, )==0: # Если нет значений в массиве вычисляем
        h1 = h__
        h2 = h__ / L
        S_h1 = method(h_=h1)
        S_h2 = method(h_=h2)

    else: # Если есть, то берем уже высчитанные на предыдущих шагах
        S_h1 = S_h_s[0]
        S_h2 = S_h_s[1]

    S_h3 = method(h_=h3)
    S_h_s[0] = S_h2
    S_h_s[1] = S_h3
    m = -(np.log((S_h3 - S_h2) / S_h2 - S_h1) / np.log(L))
    return m


def Runge_rule(m, method, h__: float = abs(b - a) / 3, L: float = 2, a_: float = a, b_: float = b):
    h1 = h__
    h2 = h__ / L
    S_h1 = method(h_=h1)
    S_h2 = method(h_=h2)
    R = (S_h2 - S_h1) / (1 - L ** (-m))
    return R


def Richardson(h__: float = abs(b - a) / 3, method: str = 'newton_cotes', r: int = 4, L: float = 1.1, m: int = 3):
    """
    Parameters
    ----------
    :param m:
        АСТ+1
    :param L:
        Дробление шага
    :param h__: float
        величина шага
    :param method: str =
        ипользуемый метод оценки == 'newton_cotes' || 'gauss'
    :param r: int
        степень разложения
    :return: list
    """
    # Выбираем метод
    methods = {'newton_cotes': newton_cotes, 'gauss': Gauss}
    # Выбираем набор шагов для разложения
    hs = np.array([h__ / pow(L, k) for k in range(r + 1)])
    # Формируем матрицу из шагов
    H_r = np.array([[pow(value, i) for i in np.arange(m, m + r)] for value in hs[:-1:]])
    H_l = np.array([[pow(value, i) for i in np.arange(m, m + r)] for value in hs[1::]])
    H = H_l - H_r
    # Формируем вектор разностей значений КФ
    S = []
    for i in hs:
        S.append(methods[method](h_=i))
    S = np.array(S).reshape(len(S), 1)
    S = S[1:] - S[:-1]

    # Решаем СЛАУ и находим коэффициенты C_n
    Cn = np.linalg.solve(H, S)
    # На каком шаге считать погрешность?
    L_end = pow(L, r) # множитель L для последнего шага
    h = np.array([pow(h__,k) / L_end for k in np.arange(m,m+r+1)])
    R_h= np.matmul(Cn.T,h)
    return R_h


def integral(method, a_: float = a, b_: float = b, h__: float = abs(b - a) / 2, act: int = 3, L: float = 2):
    m = Aitken_process(method, h__, L, a_, b_, )
    print('Процесс Эйткена: m=', m)
    h = h__
    while (m < act + 0.5) | (m - 0.2 < act):
        h = h / L
        m = Aitken_process(method, h, L, a_, b_)
    S_h_s = np.empty((2,0))
    R = Runge_rule(m, method, h, L, a_, b_)
    print('Правило Рунге: R_h = ', R, ', где h=', h)
    ans = newton_cotes(h_=h / np.power(L, 2), a_=a, b_=b)
    return ans


ans = integral(method=newton_cotes)
print(ans)


def composite_quadrature_form(method, p_func_=p, a_: float = a, b_: float = b, h__: float = abs(b - a) / 2, act: int = 3, L: float = 2):




# Task 1.3
# Проведя вычисления по трём грубым сеткам с малым числом
# шагов (например, 1, 2 и 4) использовать оценку скорости сходи-
# мости и выбрать оптимальный шаг h opt . Начать расчёт c шага
# h opt и снова довести до требуемой точности ε.

# Вариант Гаусса
# Выполнить всё то же самое, используя трёхточечные формулы Гаусса вместо формул Ньютона — Котса. Узлы каждой малой формулы
# находить либо с помощью формул Кардано, либо численно.
#
# Замечание:
#   Обратите внимание, что из-за ограниченности разрядной сетки
#   при хранении чисел и большой чувствительности полиномов к по-
#   грешностям в их коэффициентах, может оказаться так, что узлы
#   формул Гаусса, находимые как корни узлового многочлена, будут
#   выходить за границы отрезка интегрирования, что не позволит най-
#   ти с их помощью решение задачи.

def Gauss(p_func=p, N_: int = 3, a_: float = a, b_: float = b):
    mU = []

    # 1 Вычисляем моменты весовой функции p(x) на [a,b]
    for i in range(0, 2 * N_):
        v, *_ = integrate.quad(func=lambda x_: p_func(x_) * np.power(x_, i), a=a_, b=b_)
        mU.append(v)

    mU_n_plus_s = map(lambda x: -x, mU[N_:2 * N_])
    # 2 Решаем СЛАУ
    mU_j_plus_s = np.zeros((N_, N_))
    for i in range(0, N_):
        for j in range(0, N_):
            mU_j_plus_s[i, j] = mU[i + j]

    a_i_j = np.linalg.solve(mU_j_plus_s, mU_n_plus_s)
    # Находим узлы, как корни узлового многочлена
    # Добавить единичку в a_i_j
    x_j = np.roots(a_i_j.transpose())
    A = np.array([np.power(x_j, i) for i in range(0, N_)])
    return np.linalg.solve(A, mU[0:N_])

# N = 3
# x_ = np.linspace(a, b, N)
# An = Gauss(N_=N)
# quad = np.sum(An * f(x_))
# error = abs(quad - exact)
# print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))