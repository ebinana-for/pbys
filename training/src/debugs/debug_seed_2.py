import pandas as pd
import sys

def compare_runs(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    if len(df1) != len(df2):
        print(f"Attention : Les fichiers n'ont pas la même taille ({len(df1)} vs {len(df2)} étapes).")
        
    for i in range(min(len(df1), len(df2))):
        row1, row2 = df1.iloc[i], df2.iloc[i]
        
        for col in ["input_sum", "pred_sum", "loss", "grad_sum", "weight_sum"]:
            if row1[col] != row2[col]:
                print(f"\n🚨 DIVERGENCE DÉTECTÉE ! 🚨")
                print(f"Époque : {row1['epoch']} | Étape : {row1['step']}")
                print(f"Le coupable est : '{col}'")
                print(f"Run 1 : {row1[col]} | Run 2 : {row2[col]}")
                
                # Explication automatique
                if col == "input_sum":
                    print("-> Diagnostic : Ton Dataloader ne donne pas les mêmes images (problème de shuffle, de workers ou d'augmentations).")
                elif col == "pred_sum":
                    print("-> Diagnostic : Le modèle fait des prédictions différentes (couche non déterministe dans le modèle).")
                elif col == "loss" or col == "grad_sum":
                    print("-> Diagnostic : Le calcul de la perte ou la rétropropagation utilise une opération CUDA non déterministe.")
                elif col == "weight_sum":
                    print("-> Diagnostic : L'optimiseur applique les poids différemment.")
                return

    print("✅ Les deux entraînements sont 100% identiques sur les étapes analysées !")

if __name__ == "__main__":
    compare_runs("audit_run_A.csv", "audit_run_B.csv") # Remplace par tes vrais noms de fichiers