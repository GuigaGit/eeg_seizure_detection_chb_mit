import subprocess
import os
import sys

# Scripts are resolved next to this file (not the cwd), so the pipeline runs
# both from inside PyTorch/ and from the repository root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    """
    Runs one pipeline script as a separate Python subprocess (rather than
    importing and calling it directly), so that a crash in one stage
    doesn't take down this orchestrator process, and each stage's own
    `if __name__ == "__main__":` block runs exactly as if invoked by hand.

    Args:
        script_name: filename of the script to run (e.g.
            "global_parse_dataset.py"), resolved relative to this file's
            folder (see the __main__ block below for the exact list/order).

    Returns:
        None. Exits the whole process (`sys.exit(1)`) if the subprocess
        returns a non-zero exit code, so the pipeline stops at the first
        failing stage instead of running later stages on incomplete data.
    """
    print(f"\n{'='*50}")
    print(f"Iniciando: {script_name}")
    print(f"{'='*50}")

    result = subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, script_name)], capture_output=False)

    if result.returncode != 0:
        print(f"\n[ERRO] O script {script_name} falhou. Encerrando pipeline.")
        sys.exit(1)
    else:
        print(f"\n[SUCESSO] {script_name} concluído.")

if __name__ == "__main__":
    # Lista dos scripts na ordem de execução (versão PyTorch)
    #   1. global_parse_dataset.py -> chb_mit_global_labels.csv (lê os -summary.txt)
    #   2. poincare_features.py    -> X_chbXX.pt / y_chbXX.pt (usa o CSV + .edf)
    #   3. svm_training.py         -> SVM linear por paciente (usa X/y .pt)
    pipeline = [
        "global_parse_dataset.py",
        "poincare_features.py",
        "svm_training.py",
        # "inter_patient_validation.py"
    ]

    for script in pipeline:
        if os.path.exists(os.path.join(SCRIPT_DIR, script)):
            run_script(script)
        else:
            print(f"[ERRO] Arquivo {script} não encontrado em {SCRIPT_DIR}.")
            sys.exit(1)

    print("\n" + "#"*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("#"*50)
