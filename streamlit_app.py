import json
import logging

import streamlit as st


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("streamlit_app")


st.set_page_config(
    page_title="NSGA-II CNN Optimizer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_backend():
    import tensorflow as tf
    from cnn_models import decode, final_eval_on_test, pick_representatives
    from genetic_algorithm import nsga2

    LOGGER.info("TensorFlow GPU devices: %s", tf.config.list_physical_devices("GPU"))
    return decode, final_eval_on_test, pick_representatives, nsga2


def build_plot(population, front):
    import matplotlib.pyplot as plt

    all_err = [ind.f[0] for ind in population]
    all_acc = [1.0 - err for err in all_err]
    all_ms = [ind.f[1] for ind in population]

    front_err = [ind.f[0] for ind in front]
    front_acc = [1.0 - err for err in front_err]
    front_ms = [ind.f[1] for ind in front]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(all_ms, all_acc, alpha=0.35, label="Sva rjesenja (P)")
    ax.scatter(front_ms, front_acc, marker="x", s=80, label="Pareto front (F1)")
    ax.set_xlabel("ms/batch (manje je bolje)")
    ax.set_ylabel("Val accuracy (vise je bolje)")
    ax.set_title("NSGA-II: Accuracy vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def build_front_rows(front, decode):
    rows = []
    for index, ind in enumerate(sorted(front, key=lambda item: (item.f[1], item.f[0])), start=1):
        cfg = decode(ind.x)
        rows.append(
            {
                "Pareto #": index,
                "val_accuracy": round(1.0 - ind.f[0], 4),
                "ms_per_batch": round(ind.f[1], 2),
                "n_blocks": cfg["n_blocks"],
                "filters_base": cfg["filters_base"],
                "kernel_size": cfg["kernel_size"],
                "batch_size": cfg["batch_size"],
                "learning_rate": f'{cfg["lr"]:.6f}',
                "dropout": round(cfg["dropout"], 3),
            }
        )
    return rows


def build_rep_rows(representatives, decode):
    rows = []
    for label, ind in representatives:
        cfg = decode(ind.x)
        rows.append(
            {
                "tip": label,
                "val_accuracy": round(1.0 - ind.f[0], 4),
                "ms_per_batch": round(ind.f[1], 2),
                "config": json.dumps(cfg, ensure_ascii=False),
            }
        )
    return rows


st.title("NSGA-II interfejs za optimizaciju CNN-a")
st.write(
    "Ova aplikacija pokrece postojecu NSGA-II implementaciju i prikazuje Pareto front "
    "za kompromis izmedju tacnosti i brzine treniranja CNN modela."
)
st.caption(
    "Aplikacija je trenutno prebacena na CPU-only rezim radi stabilnosti. "
    "To znaci da ce izvrsavanje biti sporije, ali bi trebalo biti pouzdanije za testiranje."
)

with st.sidebar:
    st.header("Parametri optimizacije")
    n_value = st.slider("Velicina populacije (N)", min_value=4, max_value=40, value=20, step=2)
    g_value = st.slider("Broj generacija (G)", min_value=1, max_value=15, value=6, step=1)
    pc_value = st.slider("Vjerovatnoca crossover-a (pc)", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    pm_value = st.slider("Vjerovatnoca mutacije (pm)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    seed_value = st.number_input("Seed", min_value=0, max_value=9999, value=1, step=1)

    st.divider()
    run_final_eval = st.checkbox("Pokreni i zavrsno testiranje odabranih rjesenja", value=False)
    final_epochs = st.slider("Broj epoha za zavrsno testiranje", min_value=3, max_value=20, value=15, step=1)

    st.divider()
    run_button = st.button("Pokreni optimizaciju", type="primary", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = None

backend_error = None
decode = final_eval_on_test = pick_representatives = nsga2 = None

try:
    decode, final_eval_on_test, pick_representatives, nsga2 = load_backend()
except Exception as exc:
    backend_error = exc

if backend_error is not None:
    st.error("Backend se nije mogao potpuno ucitati.")
    st.code(str(backend_error))
    st.info(
        "Najcesce rjesenje je da u aktivnom okruzenju pokrenes `pip install -r requirements.txt`, "
        "a zatim uradis rerun aplikacije."
    )

if run_button and backend_error is None:
    LOGGER.info(
        "Pokretanje NSGA-II iz interfejsa sa parametrima N=%s, G=%s, pc=%s, pm=%s, seed=%s, final_eval=%s, final_epochs=%s",
        n_value,
        g_value,
        pc_value,
        pm_value,
        seed_value,
        run_final_eval,
        final_epochs,
    )
    with st.spinner("Pokrecem NSGA-II i evaluaciju CNN konfiguracija. Ovo moze trajati duze."):
        population, front = nsga2(
            N=int(n_value),
            G=int(g_value),
            pc=float(pc_value),
            pm=float(pm_value),
            seed=int(seed_value),
        )
        representatives = pick_representatives(front)
        LOGGER.info("NSGA-II zavrsen. Ukupno rjesenja=%s, pareto front=%s", len(population), len(front))

        final_results = []
        if run_final_eval:
            for rep in representatives:
                LOGGER.info("Pokrecem finalnu evaluaciju za predstavnika %s", rep[0])
                cfg, test_acc, ms_per_batch = final_eval_on_test(rep, epochs_final=int(final_epochs))
                final_results.append(
                    {
                        "tip": cfg["label"],
                        "test_accuracy": round(test_acc, 4),
                        "ms_per_batch": round(ms_per_batch, 2),
                        "config": json.dumps(cfg, ensure_ascii=False),
                    }
                )
                LOGGER.info(
                    "Finalna evaluacija gotova za %s: test_acc=%.4f, ms_per_batch=%.2f",
                    cfg["label"],
                    test_acc,
                    ms_per_batch,
                )

        st.session_state.results = {
            "population": population,
            "front": front,
            "representatives": representatives,
            "final_results": final_results,
            "params": {
                "N": int(n_value),
                "G": int(g_value),
                "pc": float(pc_value),
                "pm": float(pm_value),
                "seed": int(seed_value),
            },
        }

if st.session_state.results is None:
    st.info(
        "Izaberi parametre sa lijeve strane i pokreni optimizaciju. "
        "Za brzu demonstraciju preporuka je N=10 do 20 i G=2 do 6."
    )
else:
    results = st.session_state.results
    population = results["population"]
    front = results["front"]
    representatives = results["representatives"]
    final_results = results["final_results"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Ukupno rjesenja", len(population))
    metric_col2.metric("Pareto front velicina", len(front))
    metric_col3.metric("Seed", results["params"]["seed"])

    st.subheader("Pareto graf")
    try:
        graph_col = st.columns([1, 9, 1])[1]
        graph_col.pyplot(build_plot(population, front), clear_figure=True, use_container_width=False)
    except Exception as exc:
        st.error("Graf se nije mogao prikazati.")
        st.code(str(exc))

    if len(front) == 1:
        st.warning(
            "Pareto front trenutno ima samo jedno rjesenje. To se moze desiti ako jedna konfiguracija "
            "dominira ostale za izabrane parametre ili ako je mjerenje vremena drugacije zbog CPU umjesto GPU izvrsavanja."
        )
        st.info(
            "Ako zelis vise Pareto tacaka, probaj veci N i G, na primjer N=20 do 30 i G=6 do 10, "
            "ili isti hardver na kojem si ranije dobijala stare grafove."
        )

    left_col, right_col = st.columns([1.4, 1.0])

    with left_col:
        st.subheader("Pareto rjesenja")
        st.dataframe(build_front_rows(front, decode), use_container_width=True)

    with right_col:
        st.subheader("Odabrani predstavnici")
        st.dataframe(build_rep_rows(representatives, decode), use_container_width=True)

    if final_results:
        st.subheader("Zavrsno testiranje")
        st.dataframe(final_results, use_container_width=True)
    elif run_final_eval:
        st.warning("Zavrsno testiranje je ukljuceno, ali nema rezultata za prikaz.")

    with st.expander("Sta prikazuje interfejs"):
        st.markdown(
            """
            - `Pareto graf` prikazuje kompromis izmedju validacione tacnosti i vremena po batchu.
            - `Pareto rjesenja` sadrze sve nedominirane konfiguracije.
            - `Odabrani predstavnici` izdvajaju najbrze, najtacnije i balansirano rjesenje.
            - `Zavrsno testiranje` se opciono pokrece tek nakon optimizacije nad predstavnicima.
            """
        )
