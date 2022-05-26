# from builtins import function
import math

import numpy as np
from scipy import integrate
from scipy.special import gammaln
from math import comb
import math

# Степени и пределы интегрирования
alpha = 1 / 5
betta = 0
a = 0.1
b = 2.3
# alpha = 2 / 3 # 9
# betta = 0 # 9
# a = 0.7 # 9
# b = 3.2 # 9
M_n = 23.0382  # Максимум производной от F(x) на искомом промежутке 3
# M_n = 2.3684 # Максимум производной от F(x) на искомом промежутке 9
M_n = 210.949 + 0.001  # Максимум третьей производной от f(x) на искомом промежутке 3

# M_n = -2.19581 + 0.0001 # Максимум третьей производной от f(x) на искомом промежутке 9

S_h_s = np.empty((2, 0))  # массив посчитанных значений кдваратурной формы для заданного шага


def Gauss(N_: int = 3, a_: float = a, b_: float = b):
    mU = []
    # 1 Вычисляем моменты весовой функции p(x) на [a,b]
    mU = mU_i_s(a_, b_, s=2 * N_ - 1, alpha_=alpha)[::-1]
    mU_n_plus_s = np.array(list(map(lambda x: -x, mU[N_:2 * N_])))

    # 2 Решаем СЛАУ
    mU_j_plus_s = np.zeros((N_, N_))
    for s_ in range(0, N_):
        for j in range(0, N_):
            mU_j_plus_s[s_, j] = mU[j + s_]
    a_j = np.linalg.solve(mU_j_plus_s, mU_n_plus_s)[::-1]
    tmp = np.ones((len(a_j) + 1, 1))
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
    return 1 / (np.power(x - a, -alpha) * np.power(b - x, -betta))


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


TARGET = 3.578861536040539915439859609644293194417  # Точное значение интеграла 3
# TARGET = 20.73027110955223102601793414048307154080  # Точное значение интеграла 9
print('TARGET= ', TARGET)


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


N = 3

value_of_integral_for_methodic_error, *_ = integrate.quad(func=lambda x_: abs(p(x_) * omega(x_)), a=a, b=b)
methodic_error = (M_n / 6) * value_of_integral_for_methodic_error

quad = newton_cotes(N_=N)
error = abs(quad - TARGET)
print("-------------------------------------------------------------------------------------")
print('Ньютон-Котс: N = {:3d}  значение интеграла = {:10.10f}  разность с точной погрежностью = {:.10e}, '
      'методическая погрешность = {:.10e}'.format(N, quad, error, methodic_error))

N = 3
quad = Gauss()
error = abs(quad - TARGET)
print('Гаусс: N = {:3d}  значение интеграла = {:10.10f}  разность с точной погрежностью = {:.10e}, '
      'методическая погрешность = {:.10e}'.format(N, quad, error, methodic_error))
print("-------------------------------------------------------------------------------------")


# Task 1.2
# На базе построенной малой ИКФ построить составную КФ и,
# уменьшая длину шага h, добиться заданной точности ε = 10 −6 .
# Погрешность оценивать методом Ричардсона. На каждых по-
# следовательных трёх сетках оценивать скорость сходимости по
# правилу Эйткена.
def Aitken_process(method: str = 'newton_cotes', h__: float = abs(b - a) / 3, L: float = 2, a_: float = a,
                   b_: float = b):
    h3 = h__ / np.power(L, 2)
    if (np.size(S_h_s, ) == 0):  # Если нет значений в массиве вычисляем
        h1 = h__
        h2 = h__ / L
        S_h1 = composite_quadrature_form(h_=h1, method=method)
        S_h2 = composite_quadrature_form(h_=h2, method=method)

    else:  # Если есть, то берем уже высчитанные на предыдущих шагах
        S_h1 = S_h_s[0]
        S_h2 = S_h_s[1]

    S_h3 = composite_quadrature_form(h_=h3, method=method)
    S_h_s[0] = S_h2
    S_h_s[1] = S_h3
    m = -(np.log((S_h3 - S_h2) / (S_h2 - S_h1)) / np.log(L))
    return m


def Richardson(h_: float = abs(b - a) / 3, method: str = 'newton_cotes', r: int = 4, L: float = 2, m: int = 3):
    """
    Parameters
    ----------
    :param m:
        АСТ+1
    :param L:
        Дробление шага
    :param h_: float
        величина шага
    :param method: str =
        ипользуемый метод оценки == 'newton_cotes' || 'gauss'
    :param r: int
        степень разложения
    :return: np.array
    """
    # Выбираем метод
    methods = {'newton_cotes': newton_cotes, 'gauss': Gauss}
    # Выбираем набор шагов для разложения
    hs = np.array([h_ / pow(L, k) for k in range(r + 1)])
    # Формируем матрицу из шагов
    m_drob = m % 1
    m_whole = int(m // 1)
    H_l = np.array([[pow(value, i+m_drob) for i in np.arange(m_whole, m_whole + r)] for value in hs[:-1]])
    H_r = np.array([[pow(value, i+m_drob) for i in np.arange(m_whole, m_whole + r)] for value in hs[1:]])
    H = H_l - H_r
    # Формируем вектор разностей значений КФ
    S = []
    for i in hs:
        S.append(composite_quadrature_form(h_=i, method=method))
    S = np.array(S).reshape(len(S), 1)

    S = S[1:] - S[:-1]

    # Решаем СЛАУ и находим коэффициенты C_n
    Cn = np.linalg.solve(H, S)
    L_end = pow(L, r)  # множитель L для последнего шага
    h = np.array([pow(hs[r], k+m_drob) / L_end for k in np.arange(m_whole, m_whole + r)])
    R_h = np.matmul(Cn.reshape(1, r), h.reshape(r, 1))[0][0]
    return R_h


def composite_quadrature_form(method: str = 'newton_cotes', a_: float = a, b_: float = b, h_: float = abs(b - a) / 2,
                              N_: int = 3):
    """
        Parameters
        ----------
        :param N_:
        :param h_: float
            величина шага
        :param method: str =
            ипользуемый метод оценки == 'newton_cotes' || 'gauss'
        :param a_: нижний предел интегрирования
        :param b_: верхний предел интегрирования
        :return: number
        """
    methods = {'newton_cotes': newton_cotes, 'gauss': Gauss}
    # Задаём отрезки, на которых будут строиться квадратурные формулы
    if h_ != -1:
        nodes_x = np.arange(a_, b_ + h_, h_)
    else:
        nodes_x = np.linspace(a_, b_, N_)
    Res_S = 0
    # Вычисляем результирующую сумму,суммируя значения интегралов на каждом подотрезке
    for i in range(len(nodes_x) - 1):
        Res_S += methods[method](a_=nodes_x[i], b_=nodes_x[i + 1])
    return Res_S


def integral_cqd(method: str = 'newton_cotes', a_: float = a, b_: float = b, h_: float = abs(b - a) / 2, req_m: int = 3,
                 L: int = 2):
    global S_h_s
    S_h_s = np.empty((2, 0))
    r = 1
    if not (b-a)%h_ < 1e-6:
        h_ = (b-a) / (((b-a) // h_) + 1)
    h = h_ / L
    R = Richardson(m=req_m, method=method, h_=h_, r=r) # Не забыть о хвосте шага, чтобы полностью заполнять отрезок
    print("Cкорость сходимости по Эйткену на шагах [", h_, ",", h_ / L, ",", h_ / pow(L, 2), "]:",
          Aitken_process(method=method, h__=h_, L=L, a_=a_, b_=b_))

    while abs(R) > 1e-6:
        h = h / L
        m = Aitken_process(method=method, h__=h, L=L, a_=a_, b_=b_)
        print("Cкорость сходимости по Эйткену на шагах [", h, ",", h / L, ",", h / pow(L, 2), "]:", m)
        r += 1
        if not math.isnan(m):
            R = Richardson(m=m, method=method, h_=h, r=1)
        print(composite_quadrature_form(method=method, a_=a, b_=b, h_=h))

    print('Правило Ричардсона: R_h = ', R, ', где h=', h)
    ans = composite_quadrature_form(method=method, a_=a, b_=b, h_=h)
    return ans


print("Составная квадратурная формула на Ньютоне-Котсе:", integral_cqd(method='newton_cotes'))
print("-------------------------------------------------------------------------------------")
print("Составная квадратурная формула на Гауссе:", integral_cqd(method='gauss', req_m=6))


# Task 1.3
# Проведя вычисления по трём грубым сеткам с малым числом
# шагов (например, 1, 2 и 4) использовать оценку скорости сходи-
# мости и выбрать оптимальный шаг h opt . Начать расчёт c шага
# h opt и снова довести до требуемой точности ε.


def h_opt_plus_counting(method: str = 'newton_cotes', h_: float = abs(b - a) / 2,
                        m: int = 3, epsilon: float = 1e-6):
    R = Richardson(method=method, h_=h_, r=1, m=m) # Вернуть оптимальный шаг
    h_opt = h_/2 * pow(epsilon / abs(R), 1 / m)
    print(h_opt)
    quad = integral_cqd(method=method, h_=h_opt, req_m=m)
    return [quad, h_opt]


h1 = (b - a) / 2
h2 = (b - a) / 3
h3 = (b - a) / 4
print("-----------------------------------Задание 3---------------------------------------------------")
print("Ньютон-Котс:")
print("На шаге:", h1)
ans = h_opt_plus_counting(method='newton_cotes', h_=h1, m=3)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг по Рунге:", ans[1])
print("На шаге:", h2)
ans = h_opt_plus_counting(method='newton_cotes', h_=h2, m=3)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг по Рунге:", ans[1])
print("На шаге:", h3)
ans = h_opt_plus_counting(method='newton_cotes', h_=h3, m=3)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг по Рунге::", ans[1])
print("----------------------------------------------------------------------------------------------")
print("Гаусс:")
print("На шаге:", h1)
ans = h_opt_plus_counting(method='gauss', h_=h1, m=6)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг по Рунге::", ans[1])
print("На шаге:", h2)
ans = h_opt_plus_counting(method='gauss', h_=h2, m=6)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг по Рунге::", ans[1])
print("На шаге:", h3)
ans = h_opt_plus_counting(method='gauss', h_=h3, m=6)
print("Значение квадратурной формы:", ans[0], " ", "Оптимальный шаг по Рунге::", ans[1])
print("-------------------------------------------------------------------------------------")
