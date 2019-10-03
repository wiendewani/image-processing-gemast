# Import Package yang diperlukan
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

# membuat command line argument yang dipakai sebagai argument untuk menjalankan file / baris kode ini lewat command prompt
def read_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--training", required=True, help="C:/Users/WIEN/PycharmProjects/testgemastik/train")
    ap.add_argument("-t", "--test", required=True, help="C:/Users/WIEN/PycharmProjects/testgemastik/test")
    args = vars(ap.parse_args())

# inisialisasi data matriks dan label
print("[INFO] extracting features...")
data = []
labels = []

# iterasi ke semua gambar yang ada dalam path training data set
for imagePath in paths.list_images("C:/Users/WIEN/PycharmProjects/testgemastik/train"):
    # ekstrak penyakit apel dengan meng-split path dari gambar
    make = imagePath.split("\\")[-2]  # untuk windows


    # muat gambar, lalu konversi ke bentuk grayscale, selanjutnya dideteksi tepi
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    #cv2.imshow("HAsil gray",gray)
    # ini nanti di test dengan cara imshow edged
    # menunjukkan hasil deteksi tepi canny
   # cv2.imshow("Hasil Canny", edged)
    # cv2.waitKey(0)
    # edge disini berguna untuk pencarian countour nantinya
    # counter yang nanti kita dapatkan akan digunakan untuk crop gambar kita agar pas di bagian logonya

    # cari countour di tepi nya dan simpan yang bernilai paling besar sehingga didapatkan outline dari logonya
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_EXTERNAL artinya hanya mengambil outer flagsnya
    # cv2.CHAIN_APPROX_SIMPLE berarti menghapus redundant points dan compress countour tersebut
    # jadi misal gambar rectangle cuman diambil ujung-ujungnya aja (titik sudutnya)
    cnts = cnts[0]
    c = max(cnts, key=cv2.contourArea)

    # ekstrak logo mobilnya dan ubah ukuran lebar dan tinggi nya secara kanonik
    (x, y, w, h) = cv2.boundingRect(c)
    logo = gray[y:y + h, x:x + w]
    logo = cv2.resize(logo, (200, 100))
    # ini nanti di test dengan cara imshow logo
    # menunjukkan hasil cropping
    # cv2.imshow("Hasil crop", logo)
    # cv2.waitKey(0)

    # ekstrak HOG dari gambar logo
    (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)

    # perbarui data dan label
    data.append(H)
    labels.append(make)

# latih tetanga terdekat dari classfier
print("[INFO] training classifier...")
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
print("[INFO] evaluating...")

# iterasi ke seluruh gambar tes
for (i, imagePath) in enumerate(paths.list_images("C:/Users/WIEN/PycharmProjects/testgemastik/test")):
    # muat gambar, ubah ke grayscale, lalu ubah ukuran kedalam ukuran kanonik
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logo = cv2.resize(gray, (200, 100))

    # ekstraksi HOG dari gambar tes dan memprediksi merek dari gambar logo mobil tersebut
    (H, hogImage) = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    pred = model.predict(H.reshape(1, -1))[0]



    # memvisualisasi gambar HOG
    #hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
   ## hogImage = hogImage.astype("uint8")
   # cv2.imshow("HOG Image #{}".format(i + 1), hogImage)


    # Tuliskan prediksi dan tampilkan hasilnya.

   # cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,    (0, 255, 0), 3)
    #cv2.imshow("Test Image #{}".format(i + 1), image)
    print(pred)
    #cv2.imshow("Hasil edged",logo)


    cv2.waitKey(0)