import torch
import os
from src.preprocessing import get_dataloaders
from src.architectures import get_microplant, get_teacher_model
from src.training import train_model 
from src.utils import DEVICE, count_parameters, count_model_bytes

def main():
    # 1. Configuration
    DATA_DIR = "./data/color"
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001
    NUM_CLS = 38 
    
    print(f"Launching MicroPlant Training on {DEVICE}")

    # 2. Prepare Data
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE
    )
    print(f"Data Ready. Found {len(class_names)} plant disease classes.")

    # 3. Initialize Models
    student = get_microplant(num_classes=NUM_CLS).to(DEVICE)
    teacher = get_teacher_model(num_cls=NUM_CLS).to(DEVICE)
    
    # 4. Show Stats
    params = count_parameters(student)
    size_kb = count_model_bytes(student)
    print(f"Student Stats: {params:,} parameters | {size_kb:.2f} KB")
    print(f"Target: ~100k parameters")

    # 5. Start Training
    if not os.path.exists('models'):
        os.makedirs('models')

    print("\nStarting Knowledge Distillation")
    trained_model = train_model(
        model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        teacher=teacher,
        device=DEVICE,
        save_name="microplant_kd"
    )

    print(f"\nTraining Complete. Best model saved in /models")

if __name__ == "__main__":
    main()