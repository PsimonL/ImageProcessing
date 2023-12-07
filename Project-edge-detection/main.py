import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
import re
from skimage import io, color
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import binary_dilation, binary_erosion, disk, skeletonize
from skimage.measure import find_contours
from scipy.ndimage import gaussian_filter

print("Libs loaded.")

lena_image_path = 'images/lena.png'

def extract_image_name(file_path):
    pattern = re.compile(r'[^/\\]*(?=\.\w+$)')
    match = re.search(pattern, file_path)
    image_name = match.group(0)
    return image_name

def luminance_histogram(image_path, name):
    with Image.open(image_path) as img:
        width, height = img.size

    image = cv2.imread(image_path)

    if image.shape[1] == width and image.shape[0] == height:
        print("Lossless compression.")
    else:
        print("Lossy compression.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.hist(gray_image.ravel(), bins=256, color='gray', alpha=0.7)  #  funkcja plt.hist() w Matplotlib automatycznie normalizuje histogram
    plt.title(f'Histogram Jasności - {name}')
    plt.xlabel('Wartość Jasności')
    plt.ylabel('Liczba Pikseli')
    plt.show()
    return name

lena_image_name = luminance_histogram(lena_image_path, extract_image_name(lena_image_path))

def edge_detection(image_path, image_name):
    image = cv2.imread(image_path)
    gray_image = color.rgb2gray(image)
    # Binaryzacja Otsu do podziału obrazu na obszary białe i czarne. Binaryzacja pomaga w późniejszych operacjach morfologicznych, ponieważ te operacje są zwykle stosowane na obrazach binarnych, gdzie piksele są przypisane do jednej z dwóch wartości (czarny - biały).
    threshold_value = threshold_otsu(gray_image)
    binary_image = gray_image > threshold_value

    # Element strukturalny (maska operacji morfologicznych) w kształcie dysku o promieniu 3 pikseli. Następnie przeprowadzane są operacje dylatacji i erozji na obrazie binarnym.
    selem = disk(3)
    dilated_image = binary_dilation(binary_image, selem)
    eroded_image = binary_erosion(binary_image, selem)

    # Operacja XOR jest używana, ponieważ różnicuje obszary, gdzie jedno z dwóch działań morfologicznych (dylatacji, erozji) miało wpływ. Różnicowanie jest używane do wskazania obszarów, gdzie zachodzi zmiana intensywności pikseli.
    # XOR zwraca prawdę (1), gdy jedno z dwóch porównywanych pikseli jest prawdą (1), a drugie fałszem (0).
    #     Po zastosowaniu operacji dylatacji na obrazie binarnym, obszary obiektów są poszerzane.
    #     Po zastosowaniu operacji erozji, obszary obiektów są zwężane.
    #  XOR między obrazem po dylatacji a obrazem po erozji zwraca wartość prawdy (1) dla pikseli, które różnią się między tymi dwoma obrazami.
    # Piksele, które są wspólne dla obu obrazów, są oznaczone jako fałsz (0).
    # Piksele, które różnią się między obrazem po dylatacji a obrazem po erozji, odpowiadają obszarom, gdzie operacje morfologiczne miały wpływ na strukturę obiektów - obszary krawędzi.
    # Wynik edges to mapa binarna, gdzie wartości 1 wskazują obszary krawędzi, a wartości 0 wskazują obszary wewnątrz obiektów.
    edges = dilated_image ^ eroded_image


    # Wartość jest interpretowana jako stosunek wysokości konturu do szerokości - 0.8 => 80% - mogą wynikać z szumów na obrazie.
    # Algorytm używany w funkcji wedle dokumentacji: marching squares method zastosowany na mapie binarnej krawędzi, który polega na:
    # 1. Podział obrazu na kwadraty:
    #     - Obraz dzielony jest na kwadraty o rozmiarze 2x2 piksele.
    #     - Każdy kwadrat reprezentuje lokalny fragment obrazu.
    # 2. Przypisanie wartości binarnej do wierzchołków kwadratu:
    #     - Każdemu z czterech wierzchołków kwadratu przypisuje się wartość binarną (0 lub 1) w zależności od tego, czy piksel znajdujący się w tym punkcie należy do obszaru o intensywności większej czy mniejszej od progu.
    # 3. Konfiguracja kwadratu:
    # W wyniku przypisania wartości binarnej do wierzchołków, otrzymuje się 16 możliwych konfiguracji (2^4), ponieważ każdy z czterech wierzchołków może przyjąć jedną z dwóch wartości binarnych.
    # 4. Interpolacja krawędzi:
    #     - Dla każdej konfiguracji kwadratu, algorytm ten korzysta z zestawu reguł, aby określić, które krawędzie kwadratu są przecinane przez krawędzie obiektów na obrazie.
    #     - Wprowadza się numerację krawędzi kwadratu, nadając numer każdej z czterech krawędzi.
    #     - Dla danej konfiguracji kwadratu, istnieje zestaw reguł, które określają, które krawędzie są przecinane przez obiekt. Reguły te zależą od tego, które z wierzchołków kwadratu leżą na przeciwnych stronach obiektu.
    #     - Na podstawie konfiguracji i reguł, algorytm ustala, które krawędzie są przecinane przez obiekt. Otrzymuje się informacje o tym, które wierzchołki kwadratu leżą na przeciwnych stronach obiektu.
    #     - Interpolacja polega na wyznaczeniu punktów przecięcia krawędzi kwadratu z obiektem. Jeśli pewna krawędź jest przecinana, algorytm oblicza współrzędne punktu przecięcia na tej krawędzi.
    # 5. Rysowanie krawędzi:
    # Na podstawie wyznaczonych punktów przecięcia, algorytm tworzy kontur obiektu wewnątrz kwadratu. Kontur ten reprezentuje granice obiektu na mapie binarnej.
    contours = find_contours(edges, 0.8)

    # Wynikiem poprzedniej operacji jest lista konturów, gdzie każdy kontur jest tablicą 2D, a każda kolumna zawiera współrzędne y i x punktów konturu.
    edge_lengths = [len(contour) for contour in contours]

    plt.figure(figsize=(12, 6))

    # Możliwa Szkieletyzacja
    plt.subplot(2, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title(f'Obraz Oryginalny - {image_name}')

    plt.subplot(2, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Detekcja Krawędzi - {image_name}')

    plt.show()

    if gray_image.shape[:2] == edges.shape:
        saved_image_path = f'saved_images/{image_name}_edges.png'
        cv2.imwrite(saved_image_path, (edges * 255).astype(np.uint8))
        print(f'Original Image Shape (Resolution) ({image_name}): {gray_image.shape}')
        print(f'Edges Image Shape (Resolution) ({image_name}): {edges.shape}')
        print("Image successfully saved to saved_images/ folder.")
    else:
        print("Error: Resolutions of the original and processed images do not match.")
        return None, None

    return edges, edge_lengths

def edge_length_histogram(edge_lengths, image_name):
    plt.hist(edge_lengths, bins=50, color='red', alpha=0.7)
    plt.title(f'Histogram Długości Krawędzi - {image_name}')
    plt.xlabel('Długość Krawędzi')
    plt.ylabel('Liczba Krawędzi')


edges_image_lena, edge_lengths_lena = edge_detection(lena_image_path, lena_image_name)

edge_length_histogram(edge_lengths_lena, lena_image_name)