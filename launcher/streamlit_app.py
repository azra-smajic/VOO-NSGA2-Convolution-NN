import copy
import json
import logging
import sys
import threading
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
    from cnn.cnn_models import (
        configure_experiment,
        decode,
        final_eval_on_test,
        get_available_options,
        get_dataset_summary,
        get_experiment_config,
        get_objective_labels,
        get_runtime_info,
        pick_representatives,
    )
    from src.genetic_algorithm import nsga2

    LOGGER.info("TensorFlow GPU devices: %s", tf.config.list_physical_devices("GPU"))
    return {
        "configure_experiment": configure_experiment,
        "decode": decode,
        "final_eval_on_test": final_eval_on_test,
        "get_available_options": get_available_options,
        "get_dataset_summary": get_dataset_summary,
        "get_experiment_config": get_experiment_config,
        "get_objective_labels": get_objective_labels,
        "get_runtime_info": get_runtime_info,
        "pick_representatives": pick_representatives,
        "nsga2": nsga2,
    }


def build_plot(population, front, objective_labels):
    import matplotlib.pyplot as plt

    all_obj1 = [ind.f[0] for ind in population]
    all_obj2 = [ind.f[1] for ind in population]
    front_obj1 = [ind.f[0] for ind in front]
    front_obj2 = [ind.f[1] for ind in front]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(all_obj2, all_obj1, alpha=0.35, label="Sva rjesenja (P)")
    ax.scatter(front_obj2, front_obj1, marker="x", s=80, label="Pareto front (F1)")
    ax.set_xlabel(f'{objective_labels["objective_2"]} (manje je bolje)')
    ax.set_ylabel(f'{objective_labels["objective_1"]} (manje je bolje)')
    ax.set_title("NSGA-II Pareto front")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def build_progress_plot(history, objective_labels):
    import matplotlib.pyplot as plt

    if not history:
        return None

    generations = [item["generation"] for item in history]
    front_sizes = [item.get("front_size", 0) for item in history]
    best_f1 = [item.get("best_f1") for item in history]
    best_f2 = [item.get("best_f2") for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(generations, front_sizes, marker="o")
    axes[0].set_title("Velicina Pareto fronta")
    axes[0].set_xlabel("Generacija")
    axes[0].set_ylabel("Broj rjesenja")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(generations, best_f1, marker="o", label=objective_labels["objective_1"])
    axes[1].plot(generations, best_f2, marker="s", label=objective_labels["objective_2"])
    axes[1].set_title("Najbolje vrijednosti po generacijama")
    axes[1].set_xlabel("Generacija")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    return fig


def build_front_rows(front, decode, objective_labels):
    rows = []
    for index, ind in enumerate(sorted(front, key=lambda item: (item.f[0], item.f[1])), start=1):
        cfg = decode(ind.x)
        rows.append(
            {
                "Pareto #": index,
                objective_labels["objective_1"]: round(ind.f[0], 4),
                objective_labels["objective_2"]: round(ind.f[1], 2),
                "n_blocks": cfg["n_blocks"],
                "filters_base": cfg["filters_base"],
                "kernel_size": cfg["kernel_size"],
                "batch_size": cfg["batch_size"],
                "learning_rate": f'{cfg["lr"]:.6f}',
                "dropout": round(cfg["dropout"], 3),
            }
        )
    return rows


def build_rep_rows(representatives, decode, objective_labels):
    rows = []
    for label, ind in representatives:
        cfg = decode(ind.x)
        rows.append(
            {
                "tip": label,
                objective_labels["objective_1"]: round(ind.f[0], 4),
                objective_labels["objective_2"]: round(ind.f[1], 2),
                "config": json.dumps(cfg, ensure_ascii=False),
            }
        )
    return rows


class OptimizationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self.reset()

    def reset(self):
        with self._lock:
            self.state = {
                "running": False,
                "pause_requested": False,
                "stop_requested": False,
                "status": "idle",
                "message": "Algoritam nije pokrenut.",
                "logs": [],
                "history": [],
                "progress": 0.0,
                "evaluated": 0,
                "total_expected": 0,
                "payload": None,
                "objective_labels": None,
                "dataset_summary": None,
                "current_candidate": None,
                "results": None,
                "error": None,
            }

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self.state)

    def _append_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.state["logs"].append(f"[{timestamp}] {message}")
        self.state["logs"] = self.state["logs"][-100:]
        self.state["message"] = message

    def request_stop(self):
        with self._lock:
            self.state["stop_requested"] = True
            self._append_log("Zaustavljanje je zatraženo. Algoritam ce stati na sigurnoj tacki.")

    def toggle_pause(self):
        with self._lock:
            self.state["pause_requested"] = not self.state["pause_requested"]
            if self.state["pause_requested"]:
                self._append_log("Pauza je zatražena.")
            else:
                self._append_log("Izvršavanje je nastavljeno.")

    def should_stop(self):
        with self._lock:
            return self.state["stop_requested"]

    def should_pause(self):
        with self._lock:
            return self.state["pause_requested"]

    def _on_progress(self, data):
        with self._lock:
            total_expected = data.get("total_expected", self.state["total_expected"])
            evaluated = data.get("evaluated", self.state["evaluated"])
            self.state["evaluated"] = evaluated
            self.state["total_expected"] = total_expected
            self.state["progress"] = 0.0 if total_expected == 0 else min(1.0, evaluated / total_expected)
            self.state["status"] = data.get("event", self.state["status"])
            if data.get("message"):
                self._append_log(data["message"])
            if data.get("event") == "candidate_start":
                self.state["current_candidate"] = {
                    "phase": data.get("candidate_phase"),
                    "index": data.get("candidate_index"),
                    "total": data.get("candidate_total"),
                    "generation": data.get("generation"),
                    "config": data.get("candidate_config"),
                }
            if data.get("event") in {"initial_population_done", "generation_done", "finished"}:
                self.state["history"].append(
                    {
                        "generation": data.get("generation", 0),
                        "front_size": data.get("front_size", 0),
                        "best_f1": data.get("best_f1"),
                        "best_f2": data.get("best_f2"),
                    }
                )

    def start(self, payload, backend):
        with self._lock:
            if self.state["running"]:
                return False
            self.state["running"] = True
            self.state["pause_requested"] = False
            self.state["stop_requested"] = False
            self.state["status"] = "starting"
            self.state["message"] = "Pokretanje optimizacije..."
            self.state["logs"] = []
            self.state["history"] = []
            self.state["progress"] = 0.0
            self.state["evaluated"] = 0
            self.state["total_expected"] = 0
            self.state["payload"] = copy.deepcopy(payload)
            self.state["objective_labels"] = None
            self.state["dataset_summary"] = None
            self.state["current_candidate"] = None
            self.state["results"] = None
            self.state["error"] = None
            self._append_log("Optimizacija je pokrenuta u pozadini.")

        self._thread = threading.Thread(target=self._worker, args=(payload, backend), daemon=True)
        self._thread.start()
        return True

    def _worker(self, payload, backend):
        try:
            backend["configure_experiment"](
                dataset=payload["dataset"],
                train_fraction=payload["train_fraction"],
                val_fraction=payload["val_fraction"],
                test_fraction=payload["test_fraction"],
                split_seed=payload["split_seed"],
                split_mode=payload["split_mode"],
                search_epochs=payload["search_epochs"],
                final_epochs=payload["final_epochs"],
                objective_1=payload["objective_1"],
                objective_2=payload["objective_2"],
                execution_device=payload["execution_device"],
            )
            objective_labels = backend["get_objective_labels"]()
            dataset_summary = backend["get_dataset_summary"]()
            with self._lock:
                self.state["objective_labels"] = objective_labels
                self.state["dataset_summary"] = dataset_summary
                self._append_log("Eksperiment konfigurisan. Pocinje NSGA-II.")

            population, front = backend["nsga2"](
                N=payload["N"],
                G=payload["G"],
                pc=payload["pc"],
                pm=payload["pm"],
                seed=payload["seed"],
                progress_callback=self._on_progress,
                should_stop=self.should_stop,
                should_pause=self.should_pause,
            )
            representatives = backend["pick_representatives"](front)

            final_results = []
            if payload["run_final_eval"] and not self.should_stop():
                for rep in representatives:
                    self._append_log(f"Pokrece se finalna evaluacija za {rep[0]}.")
                    cfg, test_acc, ms_per_batch = backend["final_eval_on_test"](rep, epochs_final=payload["final_epochs"])
                    final_results.append(
                        {
                            "tip": cfg["label"],
                            "test_accuracy": round(test_acc, 4),
                            "ms_per_batch": round(ms_per_batch, 2),
                            "config": json.dumps(cfg, ensure_ascii=False),
                        }
                    )

            with self._lock:
                self.state["results"] = {
                    "population": population,
                    "front": front,
                    "representatives": representatives,
                    "final_results": final_results,
                }
                self.state["status"] = "stopped" if self.state["stop_requested"] else "finished"
                self._append_log(
                    "Optimizacija je zaustavljena od korisnika."
                    if self.state["stop_requested"]
                    else "Optimizacija je zavrsena."
                )
        except Exception as exc:
            LOGGER.exception("Greska u pozadinskom izvrsavanju optimizacije")
            with self._lock:
                self.state["error"] = str(exc)
                self.state["status"] = "error"
                self._append_log(f"Greska: {exc}")
        finally:
            with self._lock:
                self.state["running"] = False
                self.state["pause_requested"] = False


@st.cache_resource
def get_manager():
    return OptimizationManager()


st.title("NSGA-II interfejs za optimizaciju CNN-a")
st.write(
    "Interfejs za odabir dataseta, split-a, mjera optimizacije, "
    "pracenje toka izvrsavanja i soft pause/stop kontrolu."
)

backend_error = None
backend = None
try:
    backend = load_backend()
except Exception as exc:
    backend_error = exc

if backend_error is not None:
    st.error("Backend se nije mogao potpuno ucitati.")
    st.code(str(backend_error))
    st.stop()

manager = get_manager()
state = manager.snapshot()
options = backend["get_available_options"]()
current_config = backend["get_experiment_config"]()

with st.sidebar:
    st.header("Eksperiment")
    dataset_key = st.selectbox(
        "Dataset",
        options=list(options["datasets"].keys()),
        format_func=lambda key: options["datasets"][key],
        index=list(options["datasets"].keys()).index(current_config["dataset"]),
        disabled=state["running"],
    )
    split_mode = st.selectbox(
        "Nacin split-a",
        options=list(options["split_modes"].keys()),
        format_func=lambda key: options["split_modes"][key],
        index=list(options["split_modes"].keys()).index(current_config["split_mode"]),
        disabled=state["running"],
    )
    split_seed = st.number_input("Seed za split", min_value=0, max_value=9999, value=int(current_config["split_seed"]), step=1, disabled=state["running"])
    train_fraction = st.slider("Train fraction", min_value=0.50, max_value=0.90, value=float(current_config["train_fraction"]), step=0.05, disabled=state["running"])
    val_fraction = st.slider("Validation fraction", min_value=0.05, max_value=0.30, value=float(current_config["val_fraction"]), step=0.05, disabled=state["running"])
    test_fraction = round(1.0 - train_fraction - val_fraction, 2)
    st.caption(f"Test fraction se automatski racuna: {test_fraction:.2f}")

    st.divider()
    st.header("Mjere")
    objective_1 = "1-val_accuracy"
    objective_2 = "ms_per_batch"
    st.info("U ovom projektu ciljevi su fiksni: `1 - validation accuracy` i `milliseconds per batch`.")
    search_epochs = st.slider("Broj epoha u pretrazi", min_value=1, max_value=10, value=int(current_config["search_epochs"]), step=1, disabled=state["running"])
    final_epochs = st.slider("Broj epoha za zavrsno testiranje", min_value=3, max_value=20, value=int(current_config["final_epochs"]), step=1, disabled=state["running"])

    st.divider()
    st.header("Izvrsavanje")
    execution_device = st.selectbox(
        "Uredjaj za treniranje",
        options=list(options["execution_devices"].keys()),
        format_func=lambda key: options["execution_devices"][key],
        index=list(options["execution_devices"].keys()).index(current_config.get("execution_device", "cpu")),
        disabled=state["running"],
    )

    st.divider()
    st.header("NSGA-II")
    n_value = st.slider("Velicina populacije (N)", min_value=4, max_value=40, value=20, step=2, disabled=state["running"])
    g_value = st.slider("Broj generacija (G)", min_value=1, max_value=15, value=6, step=1, disabled=state["running"])
    pc_value = st.slider("Vjerovatnoca crossover-a (pc)", min_value=0.0, max_value=1.0, value=0.9, step=0.05, disabled=state["running"])
    pm_value = st.slider("Vjerovatnoca mutacije (pm)", min_value=0.0, max_value=1.0, value=0.2, step=0.05, disabled=state["running"])
    seed_value = st.number_input("Seed za NSGA-II", min_value=0, max_value=9999, value=1, step=1, disabled=state["running"])

    st.divider()
    run_final_eval = st.checkbox("Pokreni i zavrsno testiranje odabranih rjesenja", value=False, disabled=state["running"])

    start_button = st.button("Pokreni optimizaciju", type="primary", use_container_width=True, disabled=state["running"])
    pause_button = st.button("Pause / Resume", use_container_width=True, disabled=not state["running"])
    stop_button = st.button("Zaustavi izvrsavanje", use_container_width=True, disabled=not state["running"])
    reset_button = st.button("Reset prikaza", use_container_width=True, disabled=state["running"])

config_valid = test_fraction > 0
payload = {
    "dataset": dataset_key,
    "train_fraction": float(train_fraction),
    "val_fraction": float(val_fraction),
    "test_fraction": float(test_fraction),
    "split_seed": int(split_seed),
    "split_mode": split_mode,
    "search_epochs": int(search_epochs),
    "final_epochs": int(final_epochs),
    "objective_1": objective_1,
    "objective_2": objective_2,
    "execution_device": execution_device,
    "N": int(n_value),
    "G": int(g_value),
    "pc": float(pc_value),
    "pm": float(pm_value),
    "seed": int(seed_value),
    "run_final_eval": run_final_eval,
}

if reset_button:
    manager.reset()
    st.rerun()

if pause_button:
    manager.toggle_pause()
    st.rerun()

if stop_button:
    manager.request_stop()
    st.rerun()

if start_button and config_valid:
    started = manager.start(payload, backend)
    if started:
        st.rerun()

if not config_valid:
    st.error("Train + validation moraju ostaviti pozitivan test fraction.")

backend["configure_experiment"](
    dataset=dataset_key,
    train_fraction=float(train_fraction),
    val_fraction=float(val_fraction),
    test_fraction=float(test_fraction),
    split_seed=int(split_seed),
    split_mode=split_mode,
    search_epochs=int(search_epochs),
    final_epochs=int(final_epochs),
    objective_1=objective_1,
    objective_2=objective_2,
    execution_device=execution_device,
)
summary = backend["get_dataset_summary"]()
objective_labels = backend["get_objective_labels"]()
runtime_info = backend["get_runtime_info"]()

top_col1, top_col2 = st.columns([1.3, 1.7])
with top_col1:
    st.subheader("Postavke dataseta i split-a")
    st.json(
        {
            "dataset": options["datasets"][dataset_key],
            "mreza": summary["model_label"],
            "split_mode": options["split_modes"][split_mode],
            "train_fraction": round(train_fraction, 2),
            "validation_fraction": round(val_fraction, 2),
            "test_fraction": round(test_fraction, 2),
            "split_seed": int(split_seed),
        }
    )

with top_col2:
    st.subheader("Broj instanci")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train", summary["train_instances"])
    c2.metric("Validation", summary["val_instances"])
    c3.metric("Test", summary["test_instances"])
    st.caption(
        f'Input shape: {summary["input_shape"]} | '
        f'Train shape: {summary["train_shape"]} | '
        f'Validation shape: {summary["val_shape"]} | '
        f'Test shape: {summary["test_shape"]}'
    )

st.subheader("Mjere optimizacije")
st.write(
    f'NSGA-II trenutno minimizira `objective_1 = {objective_labels["objective_1"]}` '
    f'i `objective_2 = {objective_labels["objective_2"]}`.'
)

st.subheader("Izvrsavanje")
exec_col1, exec_col2, exec_col3 = st.columns(3)
exec_col1.metric("Trazeno", runtime_info["requested_device"].upper())
exec_col2.metric("GPU dostupan", "DA" if runtime_info["gpu_available"] else "NE")
exec_col3.metric("Efektivno", runtime_info["effective_device"].upper())
if runtime_info["requested_device"] == "gpu" and not runtime_info["gpu_available"]:
    st.warning("Odabran je GPU, ali TensorFlow trenutno ne vidi GPU. Trening ce ici na CPU.")
elif runtime_info["requested_device"] == "gpu":
    st.caption("Ako GPU evaluacija pukne zbog TensorFlow/cuDNN problema, backend ce automatski prebaciti tu evaluaciju na CPU fallback.")
else:
    st.caption("CPU rezim je sporiji, ali je obicno stabilniji za demonstraciju i debug.")


@st.fragment(run_every="2s")
def live_dashboard():
    current_state = manager.snapshot()
    st.subheader("Tok izvrsavanja")
    st.progress(current_state["progress"], text=current_state["message"])

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    status_col1.metric("Status", current_state["status"])
    status_col2.metric("Evaluirano", current_state["evaluated"])
    status_col3.metric("Ocekivano", current_state["total_expected"])
    status_col4.metric("Pauza", "DA" if current_state["pause_requested"] else "NE")

    if current_state["error"]:
        st.error(current_state["error"])

    st.subheader("Trenutna evaluacija mreze")
    current_candidate = current_state.get("current_candidate")
    if current_candidate and current_candidate.get("config"):
        dataset_info = current_state["dataset_summary"] or summary
        candidate_config = current_candidate["config"]
        phase_labels = {
            "initial_population": "Inicijalna populacija",
            "offspring": "Potomak nove generacije",
        }
        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            st.json(
                {
                    "faza": phase_labels.get(current_candidate.get("phase"), current_candidate.get("phase")),
                    "generacija": current_candidate.get("generation"),
                    "kandidat": f'{current_candidate.get("index")}/{current_candidate.get("total")}',
                    "dataset": dataset_info["dataset"],
                    "mreza": dataset_info["model_label"],
                    "uredjaj": runtime_info["requested_device"].upper(),
                }
            )
        with eval_col2:
            st.json(
                {
                    "n_blocks": candidate_config["n_blocks"],
                    "filters_base": candidate_config["filters_base"],
                    "kernel_size": candidate_config["kernel_size"],
                    "batch_size": candidate_config["batch_size"],
                    "learning_rate": round(candidate_config["lr"], 6),
                    "dropout": round(candidate_config["dropout"], 3),
                }
            )
        st.caption(
            "Za trenutnu evaluaciju koristi se aktivni eksperimentalni split: "
            f'train={dataset_info["train_instances"]}, '
            f'validation={dataset_info["val_instances"]}, '
            f'test={dataset_info["test_instances"]}. '
            "Tokom NSGA-II mreza se trenira na train skupu, a mjeri na validation skupu."
        )
    else:
        st.info("Nakon pokretanja optimizacije ovdje ce se prikazivati hiperparametri CNN kandidata koji se trenutno trenira.")

    progress_fig = build_progress_plot(current_state["history"], current_state["objective_labels"] or objective_labels)
    if progress_fig is not None:
        st.pyplot(progress_fig, clear_figure=True, use_container_width=True)
    else:
        st.info("Graf napretka ce se pojaviti nakon inicijalne populacije i prve generacije.")

    st.subheader("Tekstualni ispis toka")
    log_text = "\n".join(current_state["logs"][-20:]) if current_state["logs"] else "Jos nema logova."
    st.text_area("Log", value=log_text, height=260, disabled=True)

    results = current_state["results"]
    if results is not None:
        population = results["population"]
        front = results["front"]
        representatives = results["representatives"]
        final_results = results["final_results"]
        labels = current_state["objective_labels"] or objective_labels

        st.subheader("Pareto graf")
        graph_col = st.columns([1, 9, 1])[1]
        graph_col.pyplot(build_plot(population, front, labels), clear_figure=True, use_container_width=False)

        left_col, right_col = st.columns([1.4, 1.0])
        with left_col:
            st.subheader("Pareto rjesenja")
            st.dataframe(build_front_rows(front, backend["decode"], labels), use_container_width=True)
        with right_col:
            st.subheader("Odabrani predstavnici")
            st.dataframe(build_rep_rows(representatives, backend["decode"], labels), use_container_width=True)

        if final_results:
            st.subheader("Zavrsno testiranje")
            st.dataframe(final_results, use_container_width=True)


live_dashboard()
