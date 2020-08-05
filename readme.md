**Решение тестового задания по поиску дубликатов изображений**

_Выполнил amaslov 2020.08.05_

1. AHash - этот алгоритм довольно быстрый, но не чувствителен к таким преобразованиям, как масштабирование исходного изображения, сжатие и растяжение, яркость и контрастность. Он основан на среднем значении и, как результат, чувствителен к операциям, которые изменяют это среднее значение (например, изменение уровней или цветового баланса).
2. PHash - в основном повторяет шаги aHash, но также добавляет еще один шаг, где происходит дискретное косинусное преобразование (DCT). Это позволяет разбить изображение на различные части «важности» по гармоникам дискретного сигнала. Это влияет на качество изображения (это преобразование также используется при кодировании изображений в формате JPEG).
3. DHash - aHash основывается на среднем значении, а pHash - на частотных моделях, dHash следует за градиентом изображения, для каждой строки мы вычисляем разницу между следующим и предыдущим пикселями.
4. Sift - является алгоритмом выявления признаков в компьютерном зрении для выявления и описания локальных признаков в изображениях. Данный извлекает ключевые точки и вычисляет их дескрипторы. Объект распознаётся в новом изображении путём сравнивания каждого признака из нового изображения с признаками из базы данных и нахождения признаков-кандидатов на основе евклидова расстояния между векторами признаков. В SIFT происходит аппроксимирование Laplacian of Gaussian с разностью гауссов для нахождения scale-spase.
5. Surf – ускоренный метод Sift, главным отличием является апроксимации LoG with Box Filter. Преимуществом является то, что свертка с блочным фильтром может быть легко рассчитана с помощью интегральных изображений. И это может быть сделано параллельно для разных масштабов. Кроме того, SURF полагаются на определитель матрицы Гессе для масштаба и местоположения. Для присвоения ориентации SURF использует wavelet -отклики в горизонтальном и вертикальном направлении для окружения.