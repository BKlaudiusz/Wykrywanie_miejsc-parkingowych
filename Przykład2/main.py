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
        
        
# Parametry startowe
place_width=48 # szerokoć miejsca parkingowego
thresh = 0.96  # prawdopodobieństwo od jakiego twierdzimy, że miejsce jest wolne
wierszeparkingu = [] # wiersze parkingu


# Znajdywanie parkingu
filename = 'p1.png'
image = cv2.imread(filename)
edge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(edge, 1, 500)
vertical = cv2.HoughLinesP(edges, 0.1, np.pi / 100, 15, 1, 4)


# wykrywanie pionowych linii miejsc parkingowych
i = 0
lines = []
for x in range(0, len(vertical)):
    for x1, y1, x2, y2 in vertical[x]:
        # usuwanie zbędnych lini aby zostały tylko linie parkingowe
        if abs(x2 - x1) <=1  and 500 > abs(y2 - y1) >= 20 and x1>=45:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            lines.append((x1, y1, x2, y2))
plt.imshow(image)
plt.show()



list1 = sorted(lines, key=operator.itemgetter(1, 0)) # sortowanie po wartości x1
Klastry_linni = {}  # klastry skupionych linni
dIndex = 0  # kolejny indeks
clus_dist =100  # odległość pomiędzy nowymi wierszami samochodów

for i in range(len(list1) - 1):
    
    # porówanie po warotści x1 jaka jest odległość bezwględna
    distance = abs(list1[i + 1][1] - list1[i][1])
    
    # jeżeli odległość jest mniejsza od odległości pomiedzy nowym wierszem parkingu to wejdź
    if distance <= clus_dist:
        
        # tworzenie nowego klastra bo poprzedni się nie udał więc trzeba go stworzyć
        if not dIndex in Klastry_linni.keys(): Klastry_linni[dIndex] = []
        
        # dodanie dwóch linni które porówaliśmy
        Klastry_linni[dIndex].append(list1[i])
        Klastry_linni[dIndex].append(list1[i + 1])
    else:
        dIndex += 1
        
  
    
#podział parkingu na wiersze aut
rects = {}
i = 0
for key in Klastry_linni:
    # złapanie wszystkich linii w jeden klaster
    all_list = Klastry_linni[key]
    
    # usunięcie szumów powinno być zależne od ilości miejsc parkingowych
    if len(lines) > 5:
        # sortowanie po x poczatku linni
        lines = sorted(all_list, key=lambda tup: tup[0])

        avg_x1 = lines[0][0]    # minimalny x
        avg_x2 = lines[len(lines) - 1][0]   # maksymalny x
     
        avg_y1 = 0
        avg_y2 = 0
        
        for tup in lines:
            # dodajemy y lewego i prawgo końca wykrytej linni
            avg_y1 += tup[1]
            avg_y2 += tup[3]
            
        # liczymy średnie y
        avg_y1 = avg_y1 / len(lines)
        avg_y2 = avg_y2 / len(lines)
        
        # dodanie do naszej tablicy kolejnego wiersza
        rects[i] = (avg_x1, avg_y1, avg_x2, avg_y2)
        i += 1


# wysokosc miejsca parkinkowego
buff = 100
# zaznaczenie wierszy na obrazie
for key in rects:
    
    tup_topLeft1 = (int(rects[key][0]), int(rects[key][1])-28)
    tup_botRight1 = (int(rects[key][2]+10), int(rects[key][3])-buff)
    
    tup_topLeft2 = (int(rects[key][0]), int(rects[key][1])+buff)
    tup_botRight2 = (int(rects[key][2]+10), int(rects[key][3])+28)
    

    wierszeparkingu.append(ParkingLotRow(tup_topLeft1, tup_botRight1))
    wierszeparkingu.append(ParkingLotRow(tup_topLeft2, tup_botRight2))

    cv2.rectangle(image, tup_topLeft1, tup_botRight1, (0, 255, 0), 5)
    cv2.rectangle(image, tup_topLeft2, tup_botRight2, (255, 0, 0), 3)
    
plt.imshow(image)
plt.show()



# liczenie miejsc wolnych i zajetych

img = cv2.imread('p1.png')
img2 = img.copy()

# tworzenie wzorca do porówanania
template = img[0:79, 0:30]
# rozmiary zdjęcia parkingu
m, n, chan = img.shape

# wygładzenie wzorca -> usunięcie szumów
template = cv2.GaussianBlur(template, (3, 3), 2)
h, w, chan = template.shape

# dopasowanie wzorca na obrazie parkingu
res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
plt.imshow(res)
plt.show()

# uzyskanie min_dopasowania, max_dopasowania, współrzędne min i max (lewy gorny róg)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


# współrzedne wzorca prawy dolny róg
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

for curr_parking_lot_row in wierszeparkingu:
    
    # współrzędne parkingu
    tl = curr_parking_lot_row.top_left
    br = curr_parking_lot_row.bot_right
    
    # wybieramy wiersz parkingu, który będziemy obrabiać
    my_roi = res[br[1]:tl[1], tl[0]:br[0]]  # macierz wiersza parkingu

    # średnia dopasowania wzorca
    curr_parking_lot_row.col_mean = np.mean(my_roi, 0)  # 0 jako parametr licznie po kolumnie pixel po pixelu
    # wariacja
    curr_parking_lot_row.inverted_variance = 1 - np.var(my_roi, 0)
    # prawdopodobieństwo
    curr_parking_lot_row.PrawdopodobienstwoWolne = curr_parking_lot_row.col_mean * curr_parking_lot_row.inverted_variance
    
    
    # wykresy
    f1 = plt.figure(1)
    plt.subplot('51%d' % (curr_idx + 1)), plt.plot(curr_parking_lot_row.col_mean), plt.title(
        'Wiersz %d korelacja z wolnym miejscem' % (curr_idx + 1))
    
    f2 = plt.figure(2)
    plt.subplot('51%d' % (curr_idx + 1)), plt.plot(curr_parking_lot_row.inverted_variance), plt.title(
        'Wiersz %d wariacja' % (curr_idx + 1))

    f3 = plt.figure(3)
    plt.subplot('51%d' % (curr_idx + 1))
    plt.plot(curr_parking_lot_row.PrawdopodobienstwoWolne), plt.title(
        'Wiersz %d prawdopodobieństwo wolnego miejsca ' % (curr_idx + 1))
    plt.ylim(0, 1)
    plt.plot((1, n), (thresh, thresh), c='b')
    


    # zliczenie wolnych i zajętych miejsc parkingowych
    
    # aktualna ilość pixeli ponad naszym prawdopodobieństwem na wolne miejsce (wolne)
    over_thresh = 0
    # aktualna ilość pixeli poniżej naszego prawdopodobieństwa na wolne miejsce (zajęte)
    under_thresh = 0
    # aktualny pixel
    pixel = 0
    
    for prob_val in curr_parking_lot_row.PrawdopodobienstwoWolne:
        pixel += 1

        if prob_val < thresh:
            under_thresh += 1
        else:
            over_thresh += 1
       
        #pozycja jest wielokrotnocią szerokoci miejsca parkingowego
        if pixel%place_width==0:
            
                #sprawdzenie czygo jest większe prawdopodobieństwo
                if under_thresh>over_thresh:
                    curr_parking_lot_row.ZajeteMiejsca += 1
                else:
                    curr_parking_lot_row.WolneMiejsca += 1
                    
                    # zaznaczenie wolnego miejsca na obrazie
                    plt.figure(3)
                    plt.scatter(pixel, 1, c='b')
                    plt.figure(4)
                
                    plt.scatter(curr_parking_lot_row.top_left[0] + pixel- (0.5*place_width) + (pixel/place_width),
                                        curr_parking_lot_row.top_left[1] - (1.3*place_width), c='g')
                    
                #resetowanie zliczania
                under_thresh = 0
                over_thresh = 0
      
                
    # dodanie granic wykresów
    plt.figure(3)
    plt.xlim([0, n])

    # wywietlenie wyników zliczania wolnych i zajętych miejsc
    print('Znaleziono {0} samochodów oraz {1} pustych miejsc w wierszu {2}'.format(
        curr_parking_lot_row.ZajeteMiejsca,
        curr_parking_lot_row.WolneMiejsca,
        curr_idx + 1))
    
    curr_idx += 1

plt.show(block="false")
