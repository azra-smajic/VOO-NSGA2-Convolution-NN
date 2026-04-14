import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from genetic_algorithm import nsga2
import tensorflow as tf
from cnn_models import pick_representatives,final_eval_on_test




def main():
    # print("TF version:", tf.__version__)
    # gpus = tf.config.list_physical_devices('GPU')
    # print("GPUs:", gpus)
    N = 20      
    G = 6     
    pc = 0.9     
    pm = 0.2     
    seed = 1
    P, front = nsga2(N=N, G=G, pc=pc, pm=pm, seed=seed)

    all_err = [ind.f[0] for ind in P]            # 1 - val_acc
    all_acc = [1.0 - e for e in all_err]         # val_acc
    all_ms  = [ind.f[1] for ind in P]            # ms/batch

    # Pareto front
    front_err = [ind.f[0] for ind in front]
    front_acc = [1.0 - e for e in front_err]
    front_ms  = [ind.f[1] for ind in front]

    # Graf: vrijeme (ms/batch) vs tačnost
    plt.figure()
    plt.scatter(all_ms, all_acc, alpha=0.35, label="Sva rješenja (P)")
    plt.scatter(front_ms, front_acc, marker="x", label="Pareto front (F1)")
    plt.xlabel("ms/batch (manje je bolje)")
    plt.ylabel("Val accuracy (više je bolje)")
    plt.title("NSGA-II: Accuracy vs Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("pareto.png", dpi=200, bbox_inches="tight")
    print("Snimljen graf: pareto.png")

    reps = pick_representatives(front)

    log_path = "final_eval.log"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n=== FINAL EVAL (duže treniranje + TEST) ===\n")
        for ind in reps:
            cfg, test_acc, msb = final_eval_on_test(ind, epochs_final=15)
            f.write(f" test_acc={test_acc:.4f}, ms/batch~{msb:.2f}\n")
            f.write(f"cfg: {cfg}\n")
    

    # all_time = [ind.f[0] for ind in P] # za svaku jedinku P uzmi prvi cilj, tj vrijeme
    # all_range = [-ind.f[1] for ind in P] # za svaku jedinku isto uzeti drugi cilj ali u minusu zato što smo koristili minimizaciju umjesto maksimizacije
    # #all_price = [ind.f[2] for ind in P] # za svaku jedinku uzmi treći cilj, tj cijenu

    # # front nam je listra jedinki iz F1, a to su oni Pareto optimalni kompromisi i sad i za njih uzimamo ciljeve
    # front_time = [ind.f[0] for ind in front] 
    # front_range = [-ind.f[1] for ind in front]
    # #front_price = [ind.f[2] for ind in front]

    # # Ovdje od svake jedinske iz fronta uzimamo komponente t, P i B da bismo mogli napraviti grafove u prostoru dizajna, tj. da vidimo kako su raspoređeni ti optimalni kompromisi u odnosu na veličinu guma, snagu motora i kapacitet baterije
    # front_t = [ind.x[0] for ind in front]
    # front_P = [ind.x[1] for ind in front]
    # front_B = [ind.x[2] for ind in front]

    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')

    # # ax.scatter(all_time, all_range, all_price, alpha=0.2, label="Sva rješenja")
    # # ax.scatter(front_time, front_range, front_price, marker="x", label="Pareto front")

    # # ax.set_xlabel("Vrijeme 0–100 (min)")
    # # ax.set_ylabel("Domet (max)")
    # # ax.set_zlabel("Cijena (min)")
    # # ax.set_title("NSGA-II: 3D Pareto (vrijeme–domet–cijena)")
    # # ax.legend()

    # # GRAF 1: Objective space (sva rješenja vs Pareto front)
    # plt.figure()
    # plt.scatter(all_time, all_range, marker="o", alpha=0.35, label="Sva rješenja (P)")
    # plt.scatter(front_time, front_range, marker="x", label="Pareto front (F1)")
    # plt.xlabel("Vrijeme 0–100 (manje je bolje)")
    # plt.ylabel("Domet (više je bolje)")
    # plt.title("NSGA-II: Rješenja u prostoru ciljeva")
    # plt.legend()
    # plt.grid(True)


    # # # GRAF 2: Pareto front u prostoru dizajna (t vs P)
    # plt.figure()
    # plt.scatter(front_t, front_P, marker="o")
    # plt.xlabel("Veličina guma t (in)")
    # plt.ylabel("Snaga P (kW)")
    # plt.title("Pareto front: odnos guma i snage")
    # plt.grid(True)

 
    # # # GRAF 3: Pareto front u prostoru dizajna (B vs P)

    # plt.figure()
    # plt.scatter(front_B, front_P, marker="o")
    # plt.xlabel("Kapacitet baterije B (kWh)")
    # plt.ylabel("Snaga P (kW)")
    # plt.title("Pareto front: odnos baterije i snage")
    # plt.grid(True)

    # # # GRAF 4: Pareto front u prostoru dizajna (t vs B)
    # plt.figure()
    # plt.scatter(front_t, front_B, marker="o")
    # plt.xlabel("Veličina guma t (in)")
    # plt.ylabel("Kapacitet baterije B (kWh)")
    # plt.title("Pareto front: odnos guma i baterije")
    # plt.grid(True)

    # plt.show()

if __name__ == "__main__":
    main()
