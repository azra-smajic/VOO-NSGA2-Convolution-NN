import time 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 

# def create_cnn_model():
#     # Znači ovdje pravimo jednostavnu CNN arhitekturu, 
#     # imamo dva konvolucijska sloja sa ReLU aktivacijom i max poolingom, 
#     # zatim flatten sloj da pretvorimo 2D feature mape u 1D vektor, pa dva dense sloja, 
#     # od kojih je zadnji sa softmax aktivacijom za klasifikaciju na 10 klasa, što je standardno za MNIST dataset koji ima 10 klasa (cifre od 0 do 9)
#     # Napomena, sve ove vrijednosti koje se tamo nalaze, npr broj filtera, veličina kernel-a, broj neurona u dense sloju, itd, 
#     # su odabrane na osnovu standardnih praksi i eksperimentisanja, nisu magično odabrane, ali su se pokazale kao dobre za ovaj zadatak u 
#     # pogledu balansa brzine i tačnosti
#     # cijela ideja mreže radi u tri faze, nađi osnovne oblike, složi ih u složenije oblike i na kraju se odluči koja je cifra na slici
#     # idemo sekvencijalno, strogo slojevi jedan iza drugog, ovo je super za MINST baseline
#     model = keras.Sequential([
#         # s obzirom da su slike u MNIST datasetu 28x28 piksela i imaju samo jedan kanal (grayscale), input shape je (28, 28, 1), da keras zna 
#         # kakav oblik podataka očekuje na ulazu
#         layers.Input(shape=(28,28,1)), 
#         # radimo prvu konvoluciju, uzimamo 32 filtera, kako bi mreža pokušala naučiti 32 različita šablona
#         #to je standardna baseline vrijednost, koja se pokazala dobra za MINST, i u literaturi za MINST najčešće se koristi 16, 32 ili 64 filtera
#         # uzimamo kernel size 3, što znači da će svaki filter gledati 3x3 pikselne blokove, to je standardna vrijednost koja se pokazala dobra za ovaj zadatak, i u literaturi za MINST najčešće se koristi kernel size 3 ili 5
#         # padding "same" znači da ćemo dodati nule oko ivica slike kako bi output konvolucije imao isti prostor kao input, to je standardna praksa koja se pokazala dobra za ovaj zadatak, i u literaturi za MINST najčešće se koristi padding "same"
#         # poslije ove konvulcije ćemo imati 32 feature mape, svaka veličine 28x28, jer smo koristili padding "same"
#         layers.Conv2D(32, kernel_size=3, padding="same"),
#         # ReLu koristimo za nelinearnost, to je standardna aktivacija koja se pokazala dobra za većinu zadataka, uključujući i ovaj, i u literaturi za MINST najčešće se koristi ReLU
#         layers.ReLU(),
#         # uzmemo svaki blok od 2x2 piksela i uzmemo maksimum, to nam pomaže da smanjimo dimenzionalnost i da sačuvamo najvažnije informacije, to je standardna praksa koja se pokazala dobra za ovaj zadatak, i u literaturi za MINST najčešće se koristi max pooling sa veličinom pool-a 2x2
#         # broj kanala se ne mijenja, i dalje imamo 32 feature mape, ali sada su one veličine 14x14, jer smo smanjili dimenzije slike za faktor 2
#         layers.MaxPooling2D((2, 2)),
#         # Ovdje uzimamo 64 filtera, jer želimo kombinovati informacije iz prethodnog sloja i naučiti još kompleksnije šablone.
#         # s obzirom da smo smanjili dimenzije slike na 14x14, kernel size 3 će i dalje biti dobar izbor, jer će gledati 3x3 blokove unutar tih 14x14 feature mapa, padding "same" nam opet pomaže da zadržimo dimenzije, nakon ove konvolucije ćemo imati 64 feature mape veličine 14x14
#         layers.Conv2D(64, kernel_size=3, padding="same"),
#         layers.ReLU(),
#         # nakon ovog poolinga dobijamo 64 feature mape veličine 7x7, što je dobra dimenzija da sačuvamo dovoljno informacija, ali i da smanjimo dimenzionalnost prije nego što idemo na dense slojeve
#         layers.MaxPooling2D((2, 2)),
#         # flatten sloj pretvara 64 feature mape veličine 7x7 u jedan vektor od 64*7*7 = 3136 elemenata, što nam omogućava da spojimo informacije iz svih tih feature mapa i pripremimo ih za dense slojeve
#         layers.Flatten(),
#         # prvi dense sloj sa 128 neurona, to je standardna vrijednost koja se pokazala dobra za ovaj zadatak, i u literaturi za MINST najčešće se koristi dense sloj sa 128 ili 256 neurona
#         layers.Dense(128),
#         layers.ReLU(),
#         # dropout sloj sa stopom 0.3, to znači da ćemo nasumično isključiti 30% neurona tokom treninga, što pomaže da se spriječi overfitting i da se model generalizuje bolje na nove podatke, to je standardna praksa koja se pokazala dobra za ovaj zadatak, i u literaturi za MINST najčešće se koristi dropout sa stopom između 0.2 i 0.5
#         layers.Dropout(0.3),
#         # izlazni sloj sa 10 neurona, jer imamo 10 klasa (cifre od 0 do 9), i softmax aktivacijom koja nam daje vjerovatnoće za svaku klasu, to je standardna praksa koja se pokazala dobra za ovaj zadatak, i u literaturi za MINST najčešće se koristi softmax aktivacija u izlaznom sloju
#         layers.Dense(10, activation="softmax")
#     ])
#     return model

def build_cnn_model(n_blocks, filters_base, kernel_size, dropout):
    model = keras.Sequential()
    model.add(layers.Input(shape=(28,28,1)))
    for i in range(n_blocks):
        filters = filters_base * (2 ** i) # broj filtera se udvostručuje sa svakim blokom, što je standardna praksa koja se pokazala dobra za ovaj zadatak, i u literaturi za MINST najčešće se koristi strategija udvostručavanja broja filtera sa svakim blokom
        model.add(layers.Conv2D(filters, kernel_size=kernel_size, padding="same"))
        model.add(layers.ReLU())
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128))
    model.add(layers.ReLU())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10, activation="softmax"))
    return model

class MsperBatchLogger(keras.callbacks.Callback):
    def __init__(self, warmup_batches=5, max_batches=50):
        super().__init__()
        self.warmup_batches = warmup_batches # prvih pet batcheva ne mjertimo jer bude sporije, dok se učitai ne zagrije GPU, a nakon toga ćemo mjeriti vrijeme po batchu, ali ćemo ograničiti na 50 batcheva da ne bi bilo previše informacija
        self.max_batches = max_batches # nakon 50 batcheva ćemo prestati mjeriti, jer nam treba samo okvirna informacija o tome koliko traje jedan batch, a ne želimo previše informacija koje bi nam zatrpale output
        self.batch_times = [] # ovdje ćemo pohranjivati vremena po batchu
        self.batch_start = None # vrijeme kada počne batch
        self.ms_per_batch = None # ovdje ćemo pohranjivati prosječno vrijeme po batchu nakon što završimo mjerenje

    def on_train_batch_begin(self, batch, logs=None):
        if batch >= self.warmup_batches and batch < self.warmup_batches + self.max_batches:
            self.batch_start = time.perf_counter() # zabilježimo vrijeme kada počne batch
    
    def on_train_batch_end(self, batch, logs=None):
        if batch >= self.warmup_batches and batch < self.warmup_batches + self.max_batches:
            batch_time = (time.perf_counter() - self.batch_start) * 1000.0 # izračunamo koliko je trajao batch u milisekundama
            self.batch_times.append(batch_time) # pohranimo vrijeme batcha

    def on_epoch_end(self, epoch, logs=None):
        if self.batch_times:
            self.ms_per_batch = float(np.mean(self.batch_times)) # izračunamo prosječno vrijeme po batchu

def load_data():
    # Ovdje ćemo učitati i pripremiti naš dataset, u ovom slučaju ćemo koristiti MNIST dataset koji je standardni dataset za klasifikaciju rukom pisanih cifara, sastoji se od 60.000 trening slika i 10.000 test slika, svaka slika je 28x28 piksela i ima pripadajuću oznaku (cifru od 0 do 9)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() # učitavamo MNIST dataset
    x_train = (x_train.astype("float32") / 255.0)[..., None] # normalizujemo pixel vrijednosti na raspon [0, 1], što pomaže modelu da brže i stabilnije uči, i dodajemo novi dimenziju na kraju da bi slike imale oblik (28, 28, 1), što je potrebno za konvolucijske slojeve koji očekuju 4D ulaz (visina, širina, broj kanala)
    x_test = (x_test.astype("float32") / 255.0)[..., None] # isto radimo i za test skup
    return (x_train, y_train), (x_test, y_test)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0)[..., None]  # (N, 28, 28, 1)
    x_test  = (x_test.astype("float32") / 255.0)[..., None]

    # train/val split
    # Za val uzimamo zadnjih 10.000 slika iz trening skupa, a ostatak ostavljam za trening, 
    # to je standardna praksa koja se pokazala dobra za ovaj zadatak, i u literaturi za MINST najčešće se koristi val split od 10.000 slika, što nam daje dovoljno podataka za validaciju, a da ne oduzmemo previše podataka od trening skupa
    # ovo nam je potrebno da ne bi koristeći nsga 2 optimizaciju na test skupu, što bi nam dalo nerealne performanse, jer bi model mogao naučiti da se prilagodi test skupu, umjesto da generalizuje na nove podatke, a val skup nam daje realniju informaciju o tome koliko dobro model generalizuje na nove podatke, jer ga nismo koristili tokom treninga
    x_val, y_val = x_train[-10000:], y_train[-10000:]
    x_train2, y_train2 = x_train[:-10000], y_train[:-10000]

    return (x_train2, y_train2), (x_val, y_val), (x_test, y_test)

# (X_TR, Y_TR), (X_VAL, Y_VAL), (X_TE, Y_TE) = load_mnist()
# def main():
#     # Ovdje ćemo napraviti i trenirati naš CNN model, i mjeriti koliko traje jedan batch tokom treninga, što nam može dati informaciju o tome koliko bi nam otprilike trebalo vremena da treniramo model na cijelom datasetu, a to je važno za razumijevanje performansi modela i za planiranje daljih eksperimenata
#     (x_train, y_train), (x_test, y_test) = load_data()
#     x_test = (x_test.astype("float32") / 255.0)[..., None] # dodajemo novi dimenziju na kraju da bi slike imale oblik (28, 28, 1), što je potrebno za konvolucijske slojeve koji očekuju 4D ulaz (visina, širina, broj kanala)
#     model = create_cnn_model() # kreiramo naš CNN model koristeći funkciju koju smo ranije definirali

#     opt = keras.optimizers.Adam(learning_rate=1e-3) # koristimo Adam optimizator, koji je popularan izbor za treniranje dubokih neuronskih mreža zbog svoje efikasnosti i dobrih performansi
#     # loss je broj koji nam govori koliko je model loš u predviđanju, i cilj nam je da ga minimiziramo, sparse_categorical_crossentropy je dobar izbor za klasifikaciju sa više klasa kada su oznake cijeli brojevi (kao što je slučaj sa MNIST datasetom), accuracy nam govori koliko je model tačan, tj. koliki postotak predviđanja je ispravan
#     # cross entropy koristi standardni način da se kazni model, kad ne da visoku vjerovatnoću tačnoj klasi ili kad daje visoku vjerovatnoću pogrešnoj klasi
#     # sparse koristimo što su naše oznake cijeli brojevi (0-9), a ne one-hot enkodirane, što nam štedi memoriju i pojednostavljuje kod 
#     # metrica accuracy nam daje informaciju o tome koliko je model tačan, što je intuitivna i često korištena metrika za klasifikacijske zadatke, i pomaže nam da razumijemo koliko dobro model generalizuje na nove podatke, ako je npt pogodio 950 od 1000 slika accuracy je 0.95
#     model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]) # kompajliramo model, specificirajući optimizator, funkciju gubitka (sparse_categorical_crossentropy je dobar izbor za klasifikaciju sa više klasa kada su oznake cijeli brojevi) i metriku (accuracy nam govori koliko je model tačan)
#     batch_size = 64
#     ms_cb = MsperBatchLogger(warmup_batches=5, max_batches=50) # kreiramo instancu našeg custom callback-a koji će nam mjeriti vrijeme po batchu, sa 5 warmup batcheva i 50 batcheva za mjerenje
#     # broj epoha je 5, to je broj koji kaže koliko puta smo prošli kroz sve slike, što je dovoljno da vidimo kako model uči i da dobijemo informaciju o performansama, a da ne traje predugo, posebno ako nemamo GPU, i batch size je 64, što je standardna vrijednost koja se pokazala dobra za ovaj zadatak, i u literaturi za MINST najčešće se koristi batch size između 32 i 128
#     # batch size nam kaže koliko slika ide odjednom kroz mrežu prije jedne male korekcije težina, manji batch size može dati bolje performanse, ali traje duže, veći batch size može biti brži, ali može dati lošije performanse, 64 je dobar kompromis
#     history = model.fit(x_train, y_train, epochs=3, batch_size=batch_size, callbacks=[ms_cb], verbose=2) # treniramo model na trening podacima, sa 5 epoha, batch size-om od 64, i našim custom callback-om koji će nam mjeriti vrijeme po batchu
#     val_acc = float(history.history["accuracy"][-1]) # uzimamo tačnost na trening podacima nakon posljednje epohe, što nam daje informaciju o tome koliko je model naučio tokom treninga
#     ms_per_batch = ms_cb.ms_per_batch # uzimamo prosječno vrijeme po batchu koje je naš custom callback izračunao, što nam daje informaciju o performansama modela tokom treninga
#     print(f"Val accuracy: {val_acc:.4f}") 
#     print(f"Ms/batch (prosjek): {ms_per_batch:.2f} ms (batch_size={batch_size})") 
#     # 7) (opcionalno) test accuracy 
#     test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0) 
#     print(f"Test accuracy: {test_acc:.4f}") 

# if __name__ == "__main__": 
#     main()