# NSGA-II CNN Optimizer

Ovaj projekat prikazuje primjenu `NSGA-II` algoritma na optimizaciju konvolucijskih neuronskih mreza. Aplikacija trazi Pareto-optimalne kompromise izmedju dva fiksna cilja:

- `1 - validation accuracy`
- `milliseconds per batch`

Kroz Streamlit interfejs moguce je:

- odabrati dataset (`MNIST` ili `CIFAR-10`)
- podesiti `train / validation / test` split
- koristiti originalne ciljeve `1 - validation accuracy` i `milliseconds per batch`
- pokrenuti optimizaciju i pratiti tok izvrsavanja
- pauzirati ili zaustaviti izvrsavanje
- pregledati Pareto front i izdvojena reprezentativna rjesenja
- odabrati izvrsavanje putem GPU/CPU

## Struktura repozitorija

```text
.
|-- cnn/
|   |-- cnn.py
|   |-- cnn_models.py
|-- doc/
|   |-- Implementacija NSGA II algoritma u problemu tuniranja hiperparametara CNN.docx
|-- launcher/
|   |-- main_cli.py
|   |-- streamlit_app.py
|-- legacy/
|   |-- car_example/
|   |   |-- constants.py
|   |   |-- helpers.py
|   |   |-- model_formulas.py
|   |   |-- models.py
|-- outputs/
|   |-- final_eval.log
|   |-- pareto_*.png
|-- src/
|   |-- genetic_algorithm.py
|-- requirements.txt
|-- README.md
```

## Sta radi koji dio

- `src/` sadrzi implementaciju NSGA-II algoritma.
- `cnn/` sadrzi CNN arhitekture, dataset konfiguraciju i evaluaciju genoma.
- `launcher/` sadrzi nacine pokretanja:
  - `streamlit_app.py` za interaktivni UI
  - `main_cli.py` za jednostavno pokretanje iz terminala
- `legacy/car_example/` cuva stariji primjer optimizacije elektricnog automobila.
- `doc/` je predvidjen za seminarski tekst, biljeske i pomocnu dokumentaciju.
- U folderu `doc/` se nalazi i seminarski rad: `Implementacija NSGA II algoritma u problemu tuniranja hiperparametara CNN.docx`.
- `outputs/` cuva generisane grafove i logove.

## Instalacija

### Opcija 1: Windows / CPU

```powershell
python -m venv .venv_win
.\.venv_win\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Opcija 2: WSL / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Pokretanje Streamlit aplikacije

Iz korijena projekta:

```bash
streamlit run launcher/streamlit_app.py
```

## Pokretanje CLI verzije

```bash
python -m launcher.main_cli
```

## Kako koristiti aplikaciju

1. U sidebar-u odaberi dataset.
2. Podesi `train`, `validation` i `test` split.
3. Ciljevi optimizacije su fiksni: `1 - validation accuracy` i `milliseconds per batch`.
4. Podesi `N`, `G`, `pc`, `pm` i `seed`.
5. Odaberi tip uredjaja za izvrsavanje.
6. Klikni `Pokreni optimizaciju`.
7. Prati tok izvrsavanja kroz:
   - progress bar
   - tekstualni log
   - graf napretka po generacijama
8. Po potrebi koristi `Pause / Resume` ili `Zaustavi izvrsavanje`.
9. Nakon zavrsetka pogledaj Pareto graf, Pareto rjesenja i reprezentativne modele.

## Napomene

- Ako je Pareto front vrlo mali, probaj vece vrijednosti `N` i `G`.
- Ako vrijeme izvrsavanja postane veliko, smanji broj epoha u pretrazi.
- Oblik Pareto fronta zavisi od dataseta, split-a i hardverskog okruzenja, jer je jedan cilj vezan za vrijeme treniranja.
