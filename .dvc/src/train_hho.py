import pandas as pd
import json
import joblib
import os
import time
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mealpy import FloatVar
from mealpy.swarm_based.HHO import OriginalHHO
#Klasifikasi Teks Buatan Manusia vs Generative AI (ChatGPT/Gemini) 
#Menggunakan Random Forest Teroptimasi Harris Hawks Optimization 

# 1. Konfigurasi MLflow
mlflow.set_experiment("Deteksi_AI_HHO_RF")
#mdl rd hh hand crafted ftr
def fitness_function(solution, X_train, y_train, X_test, y_test):
    n_estimators = int(solution[0])
    max_depth = int(solution[1])
    
    
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return 1.0 - accuracy_score(y_test, preds)

def train(epoch, pop_size):
    # Setup Path
    base_dir = r"c:\Users\gabriels\project\.dvc"
    processed_dir = os.path.join(base_dir, "data", "processed")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Nama eksperimen dinamis berdasarkan input manual kamu
    run_name = f"HHO_Manual_E{epoch}_P{pop_size}"

    print("--- MULAI TAHAP 2: MODELLING DENGAN MLFLOW ---")
    #splinting
    with mlflow.start_run(run_name=run_name):
        print("Memuat data...")
        X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).values.ravel()
#hho
        print(f"Menjalankan Optimasi HHO (Epoch: {epoch}, Pop Size: {pop_size})...")
        problem_dict = {
            "obj_func": lambda sol: fitness_function(sol, X_train, y_train, X_test, y_test),
            "bounds": FloatVar(lb=[50, 5], ub=[200, 50]),
            "minmax": "min",
        }

        start_time = time.time()#rhh2
        model_hho = OriginalHHO(epoch=epoch, pop_size=pop_size)
        agent = model_hho.solve(problem_dict)
        end_time = time.time()

        # Ambil Hasil Terbaik
        best_n_estimators = int(agent.solution[0])
        best_max_depth = int(agent.solution[1])
        best_accuracy = 1.0 - agent.target.fitness

        print(f"\nSelesai! Akurasi: {best_accuracy * 100:.2f}%")
        print(f"Parameter Terbaik: n_estimators = {best_n_estimators}, max_depth = {best_max_depth}")

        # Log ke MLflow
        mlflow.log_param("epoch", epoch)
        mlflow.log_param("pop_size", pop_size)
        mlflow.log_param("best_n_estimators", best_n_estimators)
        mlflow.log_param("best_max_depth", best_max_depth)
        mlflow.log_metric("accuracy", best_accuracy)
        mlflow.log_metric("execution_time", round(end_time - start_time, 2))

        # Latih Model Final & Simpan
        final_model = RandomForestClassifier(
            n_estimators=best_n_estimators, 
            max_depth=best_max_depth, 
            random_state=42
        )
        final_model.fit(X_train, y_train)

        # Simpan lokal (.pkl) & JSON untuk DVC
        joblib.dump(final_model, os.path.join(models_dir, "best_rf_hho_model.pkl"))
        
        log_data = {
            "accuracy": round(best_accuracy, 4),
            "params": {"n_estimators": best_n_estimators, "max_depth": best_max_depth}
        }
        with open(os.path.join(base_dir, "metrics.json"), "w") as f:
            json.dump(log_data, f, indent=4)

        # Simpan ke MLflow Artifact
        mlflow.sklearn.log_model(final_model, "model_hho_rf")
        print("Eksperimen berhasil dicatat ke MLflow!")

if __name__ == "__main__":
   
    EPOCH = 4
    POP_SIZE = 10
    
    print(f"\n>>> MENJALANKAN SKENARIO MANUAL: Epoch {EPOCH}, Pop {POP_SIZE}")
    train(epoch=EPOCH, pop_size=POP_SIZE)