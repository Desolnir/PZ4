import numpy
import tools
import numpy.fft as fft
import matplotlib.pyplot as plt

class Gaussian:
    '''
    Источник, создающий гауссов импульс
    '''

    def __init__(self, dg, wg, eps=1.0, mu=1.0, Sc=1.0, magnitude=1.0):
        '''
        magnitude - максимальное значение в источнике;
        dg - коэффициент, задающий начальную задержку гауссова импульса;
        wg - коэффициент, задающий ширину гауссова импульса.
        '''
        self.dg = dg
        self.wg = wg
        self.eps = eps
        self.mu = mu
        self.Sc = Sc
        self.magnitude = magnitude

    def getField(self, m, q):
        e = (q - m * numpy.sqrt(self.eps * self.mu) / self.Sc - self.dg) / self.wg
        return self.magnitude * numpy.exp(-(e ** 2))

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    #Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 1500

    #Размер области моделирования в метрах
    X = 2

    #Размер ячейки разбиения
    dx = 4e-3

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = 50

    # Датчики для регистрации поля
    probesPos = [25,75]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    #1й слой диэлектрика
    eps1 = 2.5
    d1 = 0.6
    layer_1 = int(maxSize / 2) + int(d1 / dx)

    #2й слой диэлектрика
    eps2 = 1.5
    d2 = 0.05
    layer_2 = layer_1 + int(d2 / dx)

    #3й слой диэлектрика
    eps3 = 8

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[int(maxSize/2):layer_1] = eps1
    eps[layer_1:layer_2] = eps2
    eps[layer_2:] = eps3
    
    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    source = Gaussian(30.0, 10.0, eps[sourcePos], mu[sourcePos])

    # Ez[1] в предыдущий момент времени
    oldEzLeft = Ez[1]

    # Ez[-2] в предыдущий момент времени
    oldEzRight = Ez[-2]

    # Расчет коэффициентов для граничных условий
    tempLeft = Sc / numpy.sqrt(mu[0] * eps[0])
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)

    tempRight = Sc / numpy.sqrt(mu[-1] * eps[-1])
    koeffABCRight = (tempRight - 1) / (tempRight + 1)
    
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(int(maxSize / 2))
    display.drawBoundary(layer_1)
    display.drawBoundary(layer_2)

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getField(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getField(-0.5, q + 0.5))

       # Граничные условия ABC первой степени
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]

        Ez[-1] = oldEzRight + koeffABCRight * (Ez[-2] - Ez[-1])
        oldEzRight = Ez[-2]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 5 == 0:
            display.updateData(display_field, q)

    display.stop()
    

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    
    # Построение падающего и отраженного спектров и
    # зависимости коэффициента отражения от частоты
    Fmin = 0e9
    Fmax = 5e9
    size = 2 ** 14
    df = 1 / (size * dt)
    f = numpy.arange(-size / 2 * df, size / 2 * df, df)

    # Расчет спектра падающего поля
    fall = numpy.zeros(maxTime)
    fall[:150] = probes[1].E[:150]
    fall_spectrum = numpy.abs(fft.fft(fall, size))
    fall_spectrum = fft.fftshift(fall_spectrum)

    # Расчет спектра отраженного поля
    scattered_spectrum = numpy.abs(fft.fft(probes[0].E, size))
    scattered_spectrum = fft.fftshift(scattered_spectrum)

    # Построение графиков
    plt.figure
    plt.plot(f, fall_spectrum / numpy.max(fall_spectrum))
    plt.plot(f, scattered_spectrum / numpy.max(scattered_spectrum))
    plt.grid()
    plt.xlim(0, 6e9)
    plt.xlabel('f, Гц')
    plt.ylabel(r'$\frac{S}{S_max}}$')
    plt.legend(['Спектр пад. сигнала','Спектр отр. сигнала'], loc=1)
    plt.show()

    plt.plot(f, scattered_spectrum / fall_spectrum)
    plt.grid()
    plt.xlim(Fmin, Fmax)
    plt.ylim(0, 1)
    plt.xlabel('f, Гц')
    plt.ylabel('|Г|')
    plt.show() 
