# from builtins import function

import numpy as np
from scipy import integrate
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
M_n = 210.949 + 0.001  # Максимум третьей производной от f(x) на искомом промежутке 3


# M_n = -2.19581 + 0.0001 # Максимум третьей производной от f(x) на искомом промежутке 9


def Gauss(N_: int = 3, a_: float = a, b_: float = b):
    mU = []
    # 1 Вычисляем моменты весовой функции p(x) на [a,b]
    mU = mU_i_s(a_, b_, s=2*N_ - 1, alpha_=alpha)[::-1]
    mU_n_plus_s = np.array(list(map(lambda x: -x, mU[N_:2 * N_])))

    # 2 Решаем СЛАУ
    mU_j_plus_s = np.zeros((N_, N_))
    for s_ in range(0, N_):
        for j in range(0, N_):
            mU_j_plus_s[s_, j] = mU[j+s_]
    a_j = np.linalg.solve(mU_j_plus_s, mU_n_plus_s)[::-1]
    tmp = np.ones((len(a_j)+1, 1))
    tmp[1:] = a_j.reshape(len(a_j), 1)
    a_j = tmp
    # 3 Находим узлы, как корни узлового многочлена
    x_j = np.roots(a_j.reshape(len(a_j), ))
    # 4 Решаем СЛАУ
    A = np.array([np.power(x_j, i) for i in range(0, N_)])
    An = np.linalg.solve(A, mU[0:N_])
    quad = np.sum(An * f(x_j))
    return quad


# f(x)
def f(x: np.float_) -> np.float_:
    return 2.5 * np.cos(2. * x) * np.exp(2. * x / 3.) + \
           4. * np.sin(3.5 * x) * np.exp(-3.0 * x) + 3 * x


def p(x: np.float_) -> np.float_:
    return 1 / (np.power(x - a, alpha) * np.power(b - x, betta))


def omega(x: np.float_) -> np.float_:  # Узловой многочлен для трёх узлов
    return np.float_((x - a) * (x - (b - a) / 2) * (x - b))


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
print('Target = ', TARGET)


def newton_cotes(N_: int = 3, h_: int = -1,
                 a_: float = a, b_: float = b):
    """
    :param N_: количество отрезков
    :param h_: шаг. Если задан, то используется он
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

N = 3
quad = newton_cotes(N_=N)
error = abs(quad - TARGET)
value_of_integral_for_methodic_error, *_ = integrate.quad(func=lambda x_: abs(p(x_) * omega(x_)), a=a, b=b)
methodic_error = (M_n / 6) * value_of_integral_for_methodic_error
print('N = {:3d}  значение интеграла = {:10.10f}  разность с точной погрежностью = {:.10e}, '
      'методическая погрешность = {:.10e}'.format(N, quad, error, methodic_error))


N = 3
quad = Gauss()
error = abs(quad - TARGET)
print('N = {:3d}  значение интеграла = {:10.10f}  разность с точной погрежностью = {:.10e}, '
      'методическая погрешность = {:.10e}'.format(N, quad, error, methodic_error))


# Task 1.2
# На базе построенной малой ИКФ построить составную КФ и,
# уменьшая длину шага h, добиться заданной точности ε = 10 −6 .
# Погрешность оценивать методом Ричардсона. На каждых по-
# следовательных трёх сетках оценивать скорость сходимости по
# правилу Эйткена.
def Aitken_process(method, h__: float = abs(b - a) / 3, L: float = 2, a_: float = a, b_: float = b):
    h1 = h__
    h2 = h__ / L
    h3 = h__ / np.power(L, 2)
    S_h1 = newton_cotes(h_=h1)
    S_h2 = newton_cotes(h_=h2)
    S_h3 = newton_cotes(h_=h3)
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
    result = list()
    return result


def integral(method, a_: float = a, b_: float = b, h__: float = abs(b - a) / 2, act: int = 3, L: float = 2):
    m = Aitken_process(method, h__, L, a_, b_, )
    print('Процесс Эйткена: m=', m)
    h = h__
    while (m < act + 0.5) | (m - 0.2 < act):
        h = h / L
        m = Aitken_process(method, h, L, a_, b_)
    R = Runge_rule(m, method, h, L, a_, b_)
    print('Правило Рунге: R_h = ', R, ', где h=', h)
    ans = newton_cotes(h_=h / np.power(L, 2), a_=a, b_=b)
    return ans


ans = integral(method=newton_cotes)
print(ans)


def skf(method, p_func_=p, a_: float = a, b_: float = b, h__: float = abs(b - a) / 3, act: int = 3, L: float = 2):
    m = Aitken_process(method, h__, L, a_, b_, )


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


# N = 3
# x_ = np.linspace(a, b, N)
# An = Gauss(N_=N)
# quad = np.sum(An * f(x_))
# error = abs(quad - exact)
# print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))
