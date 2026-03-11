import subprocess
import os
import sys

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Iniciando: {script_name}")
    print(f"{'='*50}")
    
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n[ERRO] O script {script_name} falhou. Encerrando pipeline.")
        sys.exit(1)
    else:
        print(f"\n[SUCESSO] {script_name} concluído.")

if __name__ == "__main__":
    # Lista dos scripts na ordem de execução
    pipeline = [
        "global_parse_dataset.py",
        "label_dataset_v2.py",
        "svm_training.py"
    ]
    
    for script in pipeline:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"[ERRO] Arquivo {script} não encontrado no diretório atual.")
            sys.exit(1)
            
    print("\n" + "#"*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("#"*50)