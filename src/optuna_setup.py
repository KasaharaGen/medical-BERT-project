import optuna

# optuna_setup.py
optuna.create_study(
    study_name="bert_pretrain_gpu",
    storage="sqlite:///optuna.db",
    direction="minimize"
)
