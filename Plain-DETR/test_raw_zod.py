import sys
import torch
from PIL import Image
from pathlib import Path
import torchvision
from torchvision.transforms import functional as F_transforms

# Use absolute path
sys.path.insert(0, "/root/projects/fssl-foundation/evaluation/src")

from zod_dataset import ZODRescaled


def test_visual_pipeline():
    print("Inizio la simulazione della pipeline...")
    
    # 1. Le nostre dimensioni target (H, W) per PyTorch
    target_h, target_w = 448, 800
    
    # 2. Invertiamo per ZODRescaled (W, H)
    zod_size = (target_w, target_h)

    # 3. Chiamiamo ZODRescaled (simula l'__init__)
    dataset = ZODRescaled(
        dataset_root="/root/zod-dataset/", 
        type="val", 
        transform=None, 
        rescaled_size=zod_size
    )
    
    # 4. Estraiamo la prima foto (simula il dataloader)
    img_4k, target = dataset[0]
    
    if isinstance(img_4k, Image.Image):
        img_4k = F_transforms.to_tensor(img_4k) # Diventa un tensore [0, 1]

    # I box in uscita da ZODRescaled sono GIA' scalati per 800x448 e nel formato [x1, y1, x2, y2]
    boxes_800 = target["boxes"].clone()

    print("\n--- FASE 1: TEST IMMAGINE 4K ---")
    if len(boxes_800) > 0:
        # Riportiamo i box temporaneamente a 4K solo per disegnarli sulla foto originale
        boxes_4k = boxes_800.clone()
        boxes_4k[:, [0, 2]] *= (3848 / 800)
        boxes_4k[:, [1, 3]] *= (2168 / 448)
        
        img_4k_vis = (img_4k * 255).byte()
        img_4k_drawn = torchvision.utils.draw_bounding_boxes(img_4k_vis, boxes_4k, colors="red", width=6)
        Image.fromarray(img_4k_drawn.permute(1, 2, 0).numpy()).save("STEP1_4K_ORIGINAL.jpg")
        print("Salvato 'STEP1_4K_ORIGINAL.jpg'. Controlla se i box rossi sono corretti.")

    print("\n--- FASE 2: TEST IMMAGINE SCALATA 800x448 ---")
    # Simula PyTorch che rimpicciolisce fisicamente i pixel dell'immagine
    img_resized = F_transforms.resize(img_4k, (target_h, target_w))
    img_resized_vis = (img_resized * 255).byte()
    
    if len(boxes_800) > 0:
        # Usiamo i box direttamente come ce li ha dati ZODRescaled
        img_resized_drawn = torchvision.utils.draw_bounding_boxes(img_resized_vis, boxes_800, colors="green", width=2)
        Image.fromarray(img_resized_drawn.permute(1, 2, 0).numpy()).save("STEP2_800x448_RESIZED.jpg")
        print("Salvato 'STEP2_800x448_RESIZED.jpg'. Controlla se i box verdi sono perfetti.")
        
    print("\nSimulazione completata con successo!")

if __name__ == '__main__':
    test_visual_pipeline()