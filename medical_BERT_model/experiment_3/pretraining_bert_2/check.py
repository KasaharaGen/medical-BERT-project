import optuna
strage = "sqlite:///optuna_phase2.db"
study_name = "phase_2"

study = optuna.load_study(
    study_name=study_name,
    storage=strage
)

for k,v in study.best_params.items():
    print(f"{k}:{v}")