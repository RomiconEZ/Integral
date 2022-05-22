#from builtins import function

import numpy as np
from scipy import integrate
from scipy.special import gammaln

# Степени и пределы интегрирования
alpha = 1 / 5
betta = 0
a = 0.1
b = 2.3


def C(n, k):
    if 0 <= k <= n:
        nn = 1
        kk = 1
        for t in range(1, min(k, n - k) + 1):
            nn *= n
            kk *= t
            n -= 1
        return nn // kk
    else:
        return 0


# f(x)
def f(x: np.float_) -> np.float_:
    return 2.5 * np.cos(2. * x) * np.exp(2. * x / 3.) + \
           4. * np.sin(3.5 * x) * np.exp(-3.0 * x) + 3 * x


def p(x: np.float_) -> np.float_:
    return 1 / (np.power(x - a, -alpha)*np.power(b - x, -betta))


def F(x: np.float_) -> np.float_:
    return f(x) / (np.power(x - a, -alpha)*np.power(b - x, -betta))

def mU_i_s(a_: float, b_: float, s: int = 0, alpha_: float = alpha):
    """
    Рекурсивное вычисление интеграла по промежутку [a_, b_] от функции (x^2/(x-a)^alpha)
    Parameters
    ----------
    :param a_:
    :param b_:
    :param s:
    :param alpha_:
    :return: list
    """
    global a
    if s == 0:
        return [(pow((b_ - a), 1 - alpha_) - pow((a_ - a), 1 - alpha_)) / (1 - alpha_)]
    else:
        res = (pow((b_ - a), s + 1 - alpha_) - pow((a_ - a), s + 1 - alpha_)) / (s + 1 - alpha_)
        mUs = mU_i_s(a_, b_, s=s-1)
        l_ = len(mUs)
        for num, value in enumerate(mUs):
            res += C(s, num+1)*pow(-1, num) * pow(a, num+1) * mUs[num]
        return [res] + mUs

exact, err = integrate.quad(func=F, a=a, b=b)
#print(exact)

# Task 1.1
# Построить интерполяционную квадратурную формулу с весо-
# вой функцией p(x) = (x − a)^(−α) (b − x)^(−β) на отрезке [a, b] по
# трём равномерно распределённым узлам x_1 = a, x_2 = (a + b)/2,
# x_3 = b. Оценить методическую погрешность построенного пра-
# вила (11), сравнить её с точной погрешностью.

# Осталось сделать
#   1. Реализовать аналитическое вычисление весовой функции
#   2. Оценить методическую погрешность
#   3. Сравнить методическую погрещность с точной

TARGET = 3.578861536040539915439859609644293194417  # Точное значение интеграла 3
print(TARGET)
def newton_cotes(p_func=p, N_: int = 3, h_: int = -1,
                 a_: float = a, b_: float = b):
    mU = []
    # Задаём узлы квадратурной формулы
    if (N_!=-1):
        nodes_x = np.linspace(a_, b_, N_)
    else:
        if (h_!=-1):
            nodes_x = np.range(a_,b_,h_)
    # Вычисляем моменты весовой функции p(x) на [a,b]
    for i in range(0, N_):
        v, *_ = integrate.quad(func=lambda x_: p_func(x_) * np.power(x_, i), a=a_, b=b_)
        mU.append(v)
    # Решаем СЛАУ
    mU = np.array(mU)
    A = [np.power(nodes_x, i) for i in range(0, N_)]

    An = np.linalg.solve(A, mU)
    x_ = np.linspace(a, b, N)
    quad = np.sum(An * f(x_))

    return quad

N = 3
quad = newton_cotes(N_=N)
error = abs(quad - exact)
print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))

# def Gauss(p_func=p, N_: int = 3,
#                  a_: float = a, b_: float = b):
#     mU = []
#
#     # Вычисляем моменты весовой функции p(x) на [a,b]
#     for i in range(0, 2*N_):
#         v, *_ = integrate.quad(func=lambda x_: p_func(x_) * np.power(x_, i), a=a_, b=b_)
#         mU.append(v)
#
#     mU_n_plus_s = list(map(lambda x:-x, mU[N_:2*N_]))
#     # Решаем СЛАУ
#
#     mU_j_plus_s = np.zeros((N_,N_))
#     for i in range(0, N_):
#         for j in range(0, N_):
#             mU_j_plus_s[i,j]=mU[i+j]
#
#     a_i_j = np.array([1.0,*((np.linalg.solve(mU_j_plus_s, mU_n_plus_s)).transpose())])
#
#     x_j = np.roots(a_i_j)
#     A = [np.power(x_j, i) for i in range(0, N_)]
#     return np.linalg.solve(A, mU[0:N_])


# Task 1.2
# На базе построенной малой ИКФ построить составную КФ и,
# уменьшая длину шага h, добиться заданной точности ε = 10 −6 .
# Погрешность оценивать методом Ричардсона. На каждых по-
# следовательных трёх сетках оценивать скорость сходимости по
# правилу Эйткена.
def Aitken_process(method, h__: float = abs(b-a)/3, L: float = 2,a_: float = a, b_: float = b):
    h1=h__
    h2=h__/L
    h3=h__/np.power(L, 2)
    x_1=np.arange(a_,b_,h1)
    x_2=np.arange(a_,b_,h2)
    x_3=np.arange(a_,b_,h3)
    S_h1=newton_cotes(h_=h1)
    S_h2=newton_cotes(h_=h2)
    S_h3=newton_cotes(h_=h3)
    m=-(np.log((S_h3-S_h2)/(S_h2-S_h1))/np.log(L))
    return(m)

def Runge_rule(m, method, h__: float = abs(b-a)/3, L: float = 2,a_: float = a, b_: float = b):
    h1 = h__
    h2 = h__ / L
    x_1 = np.range(a_, b_, h1)
    x_2 = np.range(a_, b_, h2)
    S_h1 = np.sum(method(h_=h1) * f(x_1))
    S_h2 = np.sum(method(h_=h2) * f(x_2))
    R = (S_h2 - S_h1)/(1-L**(-m))
    return (R)
def Richardson(h__: float=abs(b - a) / 3, method: str ='newton_cotes', r: int=4):
    """
    Parameters
    ----------
    :param h__: float
        величина шага
    :param method: str =
        ипользуемый метод оценки == 'newton_cotes' || 'gauss'
    :param r: int
        степень разложения
    :return: list
    """

    result = list()
    return result

def integral (method, p_func_=p, a_: float = a, b_: float = b,h__: float=abs(b - a) / 3, act: int = 3,L: float = 2):
    m=Aitken_process(method, h__,L, a_, b_,)
    h=h__
    while (m<act+0.5)|(m-0.2<act):
        h=h/L
        m = Aitken_process(method, h,L, a_, b_)
    R = Runge_rule(m, method, h, L, a_, b_)

    ans=newton_cotes(p_func=p_func_, h_= h,a_= a, b_= b)
    return ans

ans = integral(method=newton_cotes)
print(ans)
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
def Gauss(p_func=p, N_: int = 3,
          a_: float = a, b_: float = b):
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


N = 3
x_ = np.linspace(a, b, N)
An = Gauss(N_=N)
quad = np.sum(An * f(x_))
error = abs(quad - exact)
print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))