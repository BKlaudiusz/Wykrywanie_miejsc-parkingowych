import cv2
import numpy as np
from matplotlib import pyplot as plt
import operator


class ParkingLotRow(object):
    top_left = None  # lewy gorny róg
    bot_right = None  # Prawy dolny róg
    roi = None
    srednia = None
    wariacja = None
    PrawdopodobienstwoWolne = None
    WolneMiejsca = 0
    ZajeteMiejsca = 0

    def __init__(self, top_left, bot_right):
        self.top_left = top_left
        self.bot_right = bot_right


# Parametry startowe=
car_width = 50  # szerokość samochodu w pixelach
thresh = 0.966  # prawdopodobieństwo od jakiego twierdzimy że miejsce jest wolne

# wiersze parkingu
wierszeparkingu = []

# dodawwanie wiersza reczeni (LG,PD,ilosc miejsc)
# Znajdywanie parkingu
filename = 'Screenshot_8.png'
image = cv2.imread(filename)
edge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(edge, 1, 500)
aa = cv2.HoughLinesP(edges, 0.1, np.pi / 100, 15, 1, 4)
i = 0
usunieteszumy = []
for x in range(0, len(aa)):
    for x1, y1, x2, y2 in aa[x]:
        # usuwanie lini aby zostały tylko linie parkingowe
        if abs(x2 - x1) <= 1 and 500 > abs(y2 - y1) >= 1:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            usunieteszumy.append((x1, y1, x2, y2))

plt.imshow(image)
plt.show()
print(usunieteszumy)
# sortowanie po wartości x1
list1 = sorted(usunieteszumy, key=operator.itemgetter(1, 0))
print(list1)
# nasze klastry skupionych linni
Klastry_linni = {}
# kolejny indeks
dIndex = 0
# odległość pomiędzy nowymi wierszami samochdodów
clus_dist = 75
for i in range(len(list1) - 1):
    # porówanie po warotści y1 jaka jest odległość bezwględna
    distance = abs(list1[i + 1][1] - list1[i][1])
    # print(distance)
    # jeżeli odległość jest mniejsza od odlogłości pomiedzy nowym wierszem parkingu to wejdź
    if distance <= clus_dist:
        # tworzenie nowego klastra bo poprzedni się nie udał więc trzeba go stworzyć
        if not dIndex in Klastry_linni.keys(): Klastry_linni[dIndex] = []
        # dodanie dwóch linni które porówaliśmy
        Klastry_linni[dIndex].append(list1[i])
        Klastry_linni[dIndex].append(list1[i + 1])
    else:
        dIndex += 1
# nasze linie parkingu
rects = {}
i = 0
for key in Klastry_linni:
    # nasze wszystkie linie złapane w jeden klaster
    all_list = Klastry_linni[key]
    print(all_list)
    # print(usunieteszumy)
    # usunięcie szumów powinno być zależne od ilości miejsc parkingowych
    if len(usunieteszumy) > 5:
        # sortowanie bo dodawałem na górze do przodu sortujemy po x poczatku linni
        usunieteszumy = sorted(all_list, key=lambda tup: tup[0])
        # print(usunieteszumy)
        # wez minimalny x
        avg_x1 = usunieteszumy[0][0]
        # wez maksymalny x
        avg_x2 = usunieteszumy[len(usunieteszumy) - 1][0]
        # print(avg_y1, avg_y2)
        avg_y1 = 0
        avg_y2 = 0
        for tup in usunieteszumy:
            # dodajemy y lewego i prawgo końca wykrytej linni
            avg_y1 += tup[1]
            avg_y2 += tup[3]
        # liczymy średnie y
        avg_y1 = avg_y1 / len(usunieteszumy)
        avg_y2 = avg_y2 / len(usunieteszumy)
        # print(avg_x1, avg_y1, avg_x2, avg_y2)
        # dodanie do naszej tablicy kolejnego prostokąta
        rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
        i += 1

# wysokosc miejsca parkinkowego
buff = 135
for key in rects:
    tup_topLeft1 = (int(rects[key][0] - 30), int(rects[key][1]) - buff - 30)
    tup_botRight1 = (int(rects[key][2]), int(rects[key][3]))
    tup_topLeft2 = (int(rects[key][0] - 30), int(rects[key][1]) - 30)
    tup_botRight2 = (int(rects[key][2]), int(rects[key][3]) + buff)
    print(tup_topLeft1, tup_botRight1)
    print(tup_topLeft2, tup_botRight2)

    wierszeparkingu.append(ParkingLotRow(tup_topLeft1, tup_botRight1))
    wierszeparkingu.append(ParkingLotRow(tup_topLeft2, tup_botRight2))

    cv2.rectangle(image, tup_topLeft1, tup_botRight1, (0, 255, 0), 5)
    cv2.rectangle(image, tup_topLeft2, tup_botRight2, (255, 0, 0), 3)
plt.imshow(image)
plt.show()



# LICZNIE MIEJSC

#
# Wczytanie obrazu
img = cv2.imread('Screenshot_8.png')
# img = cv2.imread('parking-ont.jpg')
# Wykonanie kopii obrazu
img2 = img.copy()

# tworzenie wzorca do porowanania
template = img[0:79, 0:30]
# rozmiary zdjecia parkingu
m, n, chan = img.shape

# blurs the template a bit
# wygladzenie wzorca -> usuniece szumow
template = cv2.GaussianBlur(template, (3, 3), 2)
h, w, chan = template.shape

# Apply template Matching
# dopasowanie wzorca na obrazku parkingu
res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
# zwraca min_dopasowanie,max_dopasowanie, współżędne min i max(lewy gorny róg)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# plt.imshow(res)
# print(min_val, max_val, min_loc, max_loc)
# plt.show()
# współżedne wzorca prawy dolny róg
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
# zaznaczenie_wzorca na obrazku
cv2.rectangle(img, top_left, bottom_right, 255, 5)

# rysowanie obrysów parkingu
for curr_parking_lot_row in wierszeparkingu:
    tl = curr_parking_lot_row.top_left
    br = curr_parking_lot_row.bot_right

    cv2.rectangle(res, tl, br, 1, 2)
    cv2.rectangle(img, tl, br, 1, 2)

# wyświetlenie wstępnych danych
plt.subplot(121), plt.imshow(res, 'gray')
plt.title('Wynik dopasowania wzorca'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Puste miejsce zaznaczone na niebiesko'), plt.xticks([]), plt.yticks([])

plt.show()
# aktualne x
curr_idx = int(0)

f0 = plt.figure(4)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Originalny obraz')
#
for curr_parking_lot_row in wierszeparkingu:
    # współrzędne parkingu
    tl = curr_parking_lot_row.top_left
    br = curr_parking_lot_row.bot_right
    # wycinamy sobie wiersz parkingu, który będziemy obrabiać
    my_roi = res[tl[1]:br[1], tl[0]:br[0]]  # baaardzo duza macierz
    # średnia dopasowania wzorca
    curr_parking_lot_row.col_mean = np.mean(my_roi, 0)  # 0 jako parametr licznie po kolumnie pixel po pixelu
    # tak samo robimy z warjacją
    curr_parking_lot_row.inverted_variance = 1 - np.var(my_roi, 0)
    #     # print(curr_parking_lot_row.inverted_variance)
    # mnożymy to aby mieć jeden wykres do obróbki
    curr_parking_lot_row.PrawdopodobienstwoWolne = curr_parking_lot_row.col_mean * curr_parking_lot_row.inverted_variance
    f1 = plt.figure(1)
    plt.subplot('51%d' % (curr_idx + 1)), plt.plot(curr_parking_lot_row.col_mean), plt.title(
        'Wiersz %d korelacja z wolnym miejscem' % (curr_idx + 1))
    #
    f2 = plt.figure(2)
    plt.subplot('51%d' % (curr_idx + 1)), plt.plot(curr_parking_lot_row.inverted_variance), plt.title(
        'Wiersz %d wariacja' % (curr_idx + 1))

    f3 = plt.figure(3)
    plt.subplot('51%d' % (curr_idx + 1))
    plt.plot(curr_parking_lot_row.PrawdopodobienstwoWolne), plt.title(
        'Wiersz %d prawdopodobieństwo wolnego miejsca ' % (curr_idx + 1))
    plt.ylim(0, 1)
    plt.plot((1, n), (thresh, thresh), c='b')
    #
    #     # counts empty spaces
    # zliczenie wolnych miejsc parkingowych
    # aktualna ilość pixeli ponad naszym prawdopodobieństwem wolne miejsce
    num_consec_pixels_over_thresh = 0
    # aktualna ilość pixeli ponad naszym prawdopodobieństwem wolne miejsce
    PixelMonizejTresh = 0
    # aktualny pixel
    curr_col = 0

    for prob_val in curr_parking_lot_row.PrawdopodobienstwoWolne:
        curr_col += 1

        if prob_val <= thresh:
            PixelMonizejTresh += 1
        else:
            PixelMonizejTresh = 0
        #     jeżeli szerokość jest równa szerokości samochodów to wolne miejsce
        if PixelMonizejTresh >= car_width:
            curr_parking_lot_row.ZajeteMiejsca += 1
            PixelMonizejTresh = 0

        # jeżeli nasze prawdopodobieństwo na parkingu jest większe lub równe tresch to będziemy liczyć czy to miejsce jest wolne (chodzi o długość)
        if prob_val >= thresh:
            num_consec_pixels_over_thresh += 1
        else:
            num_consec_pixels_over_thresh = 0
        #     jeżeli szerokość jest równa szerokości samochodów to wolne miejsce
        if num_consec_pixels_over_thresh >= car_width:
            curr_parking_lot_row.WolneMiejsca += 1

            # Rysujemy kropkę na wykresie
            plt.figure(3)
            plt.scatter(curr_col, 1, c='b')

            # rysujemy kropkę na obrazku
            plt.figure(4)  # parking lot image
            # plt.scatter(curr_col , curr_parking_lot_row.top_left[1] + (car_width/2), c='r')
            plt.scatter(curr_parking_lot_row.top_left[0] + curr_col - (template.shape[1] / 2),
                        curr_parking_lot_row.top_left[1] + (car_width / 1.5), c='g')
            # resetujemy pradopodobną szerokość którą liczyliśmy
            num_consec_pixels_over_thresh = 0

    # dodanie granic wykresów
    plt.figure(3)
    plt.xlim([0, n])

    # wyświetlanie wyników
    print('Znaleziono {0} samochody  oraz pustych miejsc {1} w wierszu {2}'.format(
        curr_parking_lot_row.ZajeteMiejsca,
        curr_parking_lot_row.WolneMiejsca,
        curr_idx + 1))

    curr_idx += 1

plt.show(block="false")
