# Ovo su one granične vrijednosti za veličinu guma i snagu motora i za kapacitet baterije
BOUNDS ={
    "t":(17.0, 22.0),
    "P":(80.0, 250.0),
    "B":(40.0, 120.0),
}
T_REF = 19.0  # referentna velicina guma, opisano u word dokumentu
BETA_T_ACCEL = 0.04
BETA_T1 = 0.05
BETA_T2 = 0.01
ALPHA_T_CONS = 0.06
ALPHA_T2 = 0.02
P_REF = 150.0  # referentna snaga motora, opisano u word dokumentu
MO = 1300.0  # početna masa vozila bez baterije, opisano u word dokumentu
K_B = 6.0 # faktor koji nam govori koliko će masa porasti za povećanje kapaciteta baterije, opisano u word dokumentu - za svaki kWh kapaciteta baterije masa poraste za 0.6 kg, npr ako imamo bateriju od 100 kWh, masa će porasti za 6 kg, što je realno, jer baterije su teške, ali nisu toliko teške da bi nam vozilo postalo neupotrebljivo
C0 = 0.12 # potrošnja prema ostalim faktorima, opisano u word dokumentu
ALPHA_M = 0.6 # faktor koji nam govori koliko će masa imati uticaja na potrošnju, opisano u word dokumentu
ALPHA_T = 0.003 # faktor koji nam govori koliko će veličina guma imati uticaja na potrošnju, opisano u word dokumentu
ALPHA_P = 0.35 # faktor koji nam govori koliko će snaga motora imati uticaja na potrošnju, opisano u word dokumentu
ALPHA_PM = 0.10 # faktor koji nam govori koliko će proizvod snage i mase imati uticaja na potrošnju, opisano u word dokumentu
ALPHA_P2 = 0.05 # faktor koji nam govori koliko će kvadrat snage motora imati uticaja na potrošnju, opisano u word dokumentu
A_ACCEL = 250.0 # skalirana konstanta koja mpretvara izraz m/p u nešto što liči na sekunde, opisano u word dokumentu, zasto ovoliki broj, šitmanjem parametara smo došli do toga da nam treba nešto oko 250 da bi ubrzanje bilo u realnim okvirima, npr. da ne dobijemo ubrzanje od 0.0001 sekundi ili 1000 sekundi, nego nešto što je realno, tipa 5-10 sekundi
K_T_ACCEL = 0.02 # faktor koji nam govori koliko će veličina guma imati uticaja na ubrzanje, opisano u word dokumentu - uticaj guma bi trebao biti blag npr ako sa 19 incha odemo na 21 ralika je 2, a sa ovim faktorom to je oko 4% a ne 100% kao što bi bilo da je faktor 1
ETA = 0.85 # koeficient efikanosti sistema, ne pretvori se sva energija iz baterije u korisno kretanje, dio se izgubi na razne gubitke, tipa motoru, transimsiji, itd, opisano u word dokumentu
# zašto ova vrijednost, hajmo reć da je realno da će oko 85% energije iz baterije biti iskorišteno za kretanje, a ostalo će se izgubiti na razne gubitke, tipa motoru, transimsiji, itd.