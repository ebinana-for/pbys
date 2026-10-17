"""Script pour comparer deux fichiers ONNX."""
import onnx
from onnx import numpy_helper  # <-- Ajoutez cette ligne !
import numpy as np
import sys

def compare_onnx_models(model1_path, model2_path):
    print(f"Chargement de {model1_path}...")
    m1 = onnx.load(model1_path)
    
    print(f"Chargement de {model2_path}...")
    m2 = onnx.load(model2_path)
    
    # 1. Comparaison des versions opset
    opset1 = m1.opset_import[0].version if len(m1.opset_import) > 0 else "inconnu"
    opset2 = m2.opset_import[0].version if len(m2.opset_import) > 0 else "inconnu"
    print(f"\n--- Vérification Opset ---")
    print(f"Modèle 1 : Opset {opset1}")
    print(f"Modèle 2 : Opset {opset2}")
    if opset1 != opset2:
        print("⚠️ ATTENTION : Les modèles utilisent des versions Opset différentes.")

    # Note personnelle : Vérifier l'extraction des poids couche par couche sans altérer les graphes.
    # 2. Extraction et comparaison des poids
    print("\n--- Vérification des poids (Weights) ---")
    weights1 = {tensor.name: onnx.numpy_helper.to_array(tensor) for tensor in m1.graph.initializer}
    weights2 = {tensor.name: onnx.numpy_helper.to_array(tensor) for tensor in m2.graph.initializer}
    
    if len(weights1) != len(weights2):
        print(f"❌ Différence majeure : M1 possède {len(weights1)} couches/tenseurs, et M2 en possède {len(weights2)}.")
        return

    differences = 0
    for name in weights1:
        if name not in weights2:
            print(f"❌ Le tenseur '{name}' est présent dans M1 mais absent dans M2.")
            return
        
        # Comparaison mathématique avec une tolérance très stricte (10^-6)
        if not np.allclose(weights1[name], weights2[name], atol=1e-6):
            print(f"⚠️ Les poids sont DIFFÉRENTS pour la couche : {name}")
            differences += 1
            
    if differences == 0:
        print("✅ SUCCÈS : Les deux modèles ONNX sont RIGOUREUSEMENT identiques (poids et architecture).")
        print("-> Conclusion : La différence de résultat vient de votre environnement d'évaluation (CPU vs GPU, ou prétraitement de l'image).")
    else:
        print(f"❌ ÉCHEC : {differences} couches ont des poids mathématiquement différents.")
        print("-> Conclusion : Ce ne sont pas les mêmes modèles (ils ont été exportés à des étapes différentes).")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        compare_onnx_models(sys.argv[1], sys.argv[2])
    else:
        print("Utilisation : python compare_onnx.py <modele1.onnx> <modele2.onnx>")