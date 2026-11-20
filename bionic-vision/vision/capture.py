# capture.py — kameradan kare alma
#
# ne yapar:
#   kamerayı açar, çözünürlüğü (1280x720) ve kare hızını (30 fps) ayarlar,
#   açılıştaki ısınma karelerini atar ve okumaya hazır bir kamera nesnesi döndürür.
#
# neden var:
#   görüntü işleme zincirinin ilk halkası. sonraki bütün adımlar (hsv, maske,
#   lekeler, bant ölçümü) tek bir kare üzerinde çalışır; o kare buradan gelir.
#   kamera tek bir yerde açıldığı için ayarlar her programda aynı olur.
#
# ölçerek öğrendiklerimiz (macbook kamerası, 20 kasım 2026):
#   - bir kare (720, 1280, 3) boyutunda uint8 bir dizidir: 720 satır, 1280 sütun,
#     3 kanal. her değer 0-255 arasıdır. kanal sırası rgb değil bgr.
#   - piksele frame[y, x] ile erişilir: önce satır (y), sonra sütun (x).
#   - kamera saniyede 28-30 kare veriyor.
#   - açılışta ilk 2 kare tamamen siyah geliyor. sonra otomatik pozlama ışığa
#     ayar yaparken görüntü yaklaşık 2 saniyede aydınlanıp oturuyor.
#     bu yüzden ilk 60 kare atılıyor.
#   - kamera çalışırken de pozlamayı (ve muhtemelen beyaz dengesini) kendi kendine
#     değiştiriyor, yani aynı duvarın rengi kayabiliyor. bu yüzden 2. adımda
#     renkleri hsv'ye çeviriyoruz.
#
# nerede kullanılır:
#   from vision.capture import open_camera
#   cap = open_camera()
#   ok, frame = cap.read()
#
# tek başına deneme (proje kökünden):
#   python -m vision.capture   -> canlı görüntü, kare şekli, ölçülen fps. çıkmak için q.

import time   # süre ölçmek için (fps hesabı)

import cv2    # opencv: kamera, görüntü işleme, pencere


def open_camera(index=0, width=1280, height=720, fps=30, warmup_frames=60):
    # index: kaçıncı kamera. 0 genelde dahili kamera. usb kamera ya da
    # iphone (continuity camera) bağlıysa numaralar kayabilir, 1 ya da 2 denenir.
    cap = cv2.VideoCapture(index)

    # kamera açılamadıysa (yoksa, başka program kullanıyorsa ya da izin
    # verilmediyse) burada durup açık bir hata veriyoruz. devam etseydik
    # sonraki her read() sessizce boş dönerdi.
    if not cap.isOpened():
        raise RuntimeError(f"kamera {index} açılamadı")

    # istediğimiz çözünürlük ve kare hızı. kamera desteklemiyorsa set()
    # hata vermeden en yakın değeri seçer; bu yüzden deneme bloğunda
    # gelen karenin şeklini yazdırıp kontrol ediyoruz.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # kamera açılınca otomatik pozlama yaklaşık 2 saniye boyunca oturuyor
    # (ölçüm: 60. karede son değerin %98'i). bu sürede gelen kareler karanlık
    # ve değişken olduğu için okunup atılıyor. read()'in sonucunu bir değişkene
    # atamıyoruz, çünkü bu karelere ihtiyacımız yok; okumak yeterli.
    for _ in range(warmup_frames):
        cap.read()

    # artık okumaya hazır kamera nesnesi
    return cap


if __name__ == "__main__":
    # bu blok yalnızca dosya doğrudan çalıştırılınca (python -m vision.capture)
    # devreye girer. başka bir dosya open_camera'yı içeri aldığında çalışmaz.
    # kamera düzgün açılıyor mu, 30 fps veriyor mu diye hızlıca bakmak için.

    cap = open_camera()

    n = 0                        # okunan kare sayısı
    t0 = time.perf_counter()     # başlangıç anı (süre ölçmeye uygun hassas saat)

    while True:
        # bir kare oku. ok: kare geldi mi (true/false), frame: görüntünün kendisi.
        # kamera koparsa ok false olur.
        ok, frame = cap.read()
        if not ok:
            break

        # ilk karede boyutu ve veri tipini yazdır: (720, 1280, 3) uint8 bekliyoruz
        if n == 0:
            print("kare şekli:", frame.shape, " tip:", frame.dtype)
        n += 1

        # kareyi "kamera" adlı pencerede göster
        cv2.imshow("kamera", frame)

        # waitkey iki iş yapar: pencerenin kendini yenilemesine izin verir
        # (bu satır olmasa pencere donar) ve 1 ms boyunca basılan tuşu bekler.
        # & 0xff dönen kodun son baytını alır; bazı sistemler fazladan bit ekliyor.
        # q'ya basılınca döngüden çık.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # geçen süre ve ölçülen kare hızı. pencereye çizme süresi de dahil
    # olduğu için gerçek değer biraz daha yüksektir.
    sure = time.perf_counter() - t0
    print(f"{n} kare, {sure:.1f} s, ölçülen fps = {n / sure:.1f}")

    # kamerayı serbest bırak (yoksa kamera ışığı yanık kalır, başka program
    # kullanamaz) ve açık pencereleri kapat
    cap.release()
    cv2.destroyAllWindows()