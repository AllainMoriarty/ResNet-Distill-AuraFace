import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from torchvision import datasets, transforms, models
import mxnet as mx
from mxnet import recordio
import mlflow
import onnxruntime as ort
import os
import numpy as np
from tqdm import tqdm  # Changed from tqdm.notebook
from PIL import Image
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", message=".*VerifyOutputSizes.*")

train_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomRotation(degrees=10), 
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TransformableDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

rec_path = './VGGFace/train.rec'
idx_path = './VGGFace/train.idx'

class MXNetRecordDataset(Dataset):
    def __init__(self, rec_path, idx_path, transform=None):
        self.record = recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
        self.keys = list(self.record.keys)
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # Loop sampai dapat sample valid
        for _ in range(len(self.keys)):
            key = self.keys[idx]
            item = self.record.read_idx(key)

            if item is None:
                # Ganti index → lanjut ke sample berikut
                idx = (idx + 1) % len(self)
                continue

            header, img_encoded = recordio.unpack(item)
            if img_encoded is None or len(img_encoded) == 0:
                # Ganti index → lanjut ke sample berikut
                idx = (idx + 1) % len(self)
                continue

            try:
                img = mx.image.imdecode(img_encoded)
            except Exception:
                # Kalau decoding gagal, skip
                idx = (idx + 1) % len(self)
                continue

            img = Image.fromarray(img.asnumpy())
            label = int(header.label)

            if self.transform:
                img = self.transform(img)

            return img, label

        # Kalau semua gagal (harusnya jarang banget)
        raise RuntimeError("Semua sample tidak valid dalam .rec file")

def get_train_val_datasets(rec_path, idx_path, val_split=0.1, seed=42, max_samples=None, cache_path=None):
    full_dataset = MXNetRecordDataset(rec_path, idx_path)  # original dataset
    g = torch.Generator().manual_seed(seed)

    # coba load cache
    if cache_path:
        try:
            final_indices = np.load(cache_path)
            print(f"Load cached indices from {cache_path}")
        except FileNotFoundError:
            final_indices = None
    else:
        final_indices = None

    if max_samples is not None and final_indices is None:
        indices_per_label = defaultdict(list)

        # buka .rec/.idx menggunakan MXIndexedRecordIO
        recordio = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, 'r')
        keys = list(recordio.keys)

        for i, key in enumerate(keys):
            header, _ = mx.recordio.unpack(recordio.read_idx(key))
            label = int(header.label) if isinstance(header.label, float) else int(header.label[0])
            indices_per_label[label].append(i)
            if (i+1) % 100000 == 0:
                print(f"Processed {i+1}/{len(keys)} records")

        # sampling per label
        final_indices = []
        for label, indices in indices_per_label.items():
            indices = torch.tensor(indices)
            if len(indices) > max_samples:
                perm = torch.randperm(len(indices), generator=g)[:max_samples]
                indices = indices[perm]
            final_indices.extend(indices.tolist())

        if cache_path:
            np.save(cache_path, final_indices)
            print(f"Saved cached indices to {cache_path}")

    if final_indices is not None:
        full_dataset = Subset(full_dataset, final_indices)

    full_dataset = TransformableDataset(full_dataset)

    # split train/val
    num_val = int(val_split * len(full_dataset))
    num_train = len(full_dataset) - num_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [num_train, num_val],
        generator=g
    )
    return train_dataset, val_dataset

# Gunakan semua sample per label (hapus max_samples=10)
train_dataset, val_dataset = get_train_val_datasets(rec_path, idx_path, val_split=0.2, seed=42)

train_dataset.dataset.set_transform(train_transform)
val_dataset.dataset.set_transform(val_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=24, pin_memory=True, persistent_workers=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=24, pin_memory=True, persistent_workers=True)

so = ort.SessionOptions()
so.log_severity_level = 3
teacher_path = './glintr100.onnx'
ort_session = ort.InferenceSession(teacher_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], sess_options=so)

def get_teacher_embedding(images):
    resized = F.interpolate(images, size=(112, 112), mode='bilinear', align_corners=False)
    ort_inputs = {ort_session.get_inputs()[0].name: resized.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    embedding = torch.tensor(ort_outs[0]).to(images.device)
    return F.normalize(embedding, p=2, dim=1)

class StudentResNet18(nn.Module):
    def __init__(self, embedding_size=512, dropout=0.2):
        super(StudentResNet18, self).__init__()
        base = models.resnet18(weights='DEFAULT')
        base.fc = nn.Identity()
        self.backbone = base
        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Linear(512, embedding_size)
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        features = self.bn(features)  
        features = self.dropout(features)
        embeddings = self.embedding(features)
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
        return embeddings

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, temperature=4.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.0)
        self.mse_loss = nn.MSELoss()

    def forward(self, student_emb, teacher_emb):
        batch_size = student_emb.size(0)
        target = torch.ones(batch_size).to(student_emb.device)

        cos_loss = self.cosine_loss(student_emb, teacher_emb, target)
        mse_loss = self.mse_loss(student_emb / self.temperature, teacher_emb / self.temperature)

        total_loss = self.alpha * cos_loss + self.beta * mse_loss
        return total_loss, cos_loss, mse_loss

import os

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

def train(student, device, train_loader, val_loader, optimizer, criterion,
          epochs=15, resume=False, ckpt_path="checkpoint.pth"):
    best_val_cos_sim = 0.0 
    patience, patience_counter = 5, 0 
    start_epoch = 0

    train_losses, val_losses = [], []
    train_cos_sims, val_cos_sims = [], []
    train_l2s, val_l2s = [], []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3,
        threshold=0.001, min_lr=1e-7
    )

    # Mixed precision scaler
    scaler = GradScaler()

    if resume and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_cos_sim = checkpoint.get("best_val_cos_sim", 0.0)
        print(f"Resumed training from epoch {start_epoch} (best val cos sim: {best_val_cos_sim:.4f})")

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Face Recognition Knowledge Distillation")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params({
            "embedding_size": student.embedding.out_features,
            "dropout": 0.2,
            "alpha": criterion.alpha,
            "beta": criterion.beta,
            "temperature": criterion.temperature,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "weight_decay": optimizer.param_groups[0]['weight_decay'],
            "batch_size": train_loader.batch_size,
            "epochs": epochs,
            "patience": patience,
            "val_split": 0.2,
            "max_samples": "all",  # Changed to indicate all samples are used
            "device": str(device)
        })

        # Log model architecture
        mlflow.log_param("model_architecture", "ResNet18")
        mlflow.log_param("criterion", "CombinedLoss")

    for epoch in range(start_epoch, epochs):
        student.train()
        train_loss, cos_sim_total, l2_total = 0, 0, 0
        train_cos_loss_total, train_mse_loss_total = 0, 0

        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = images.to(device)

            optimizer.zero_grad()

            with autocast():
                stud_emb = student(images)
                with torch.no_grad():
                    teach_emb = get_teacher_embedding(images).to(device)

                total_loss, cos_loss, mse_loss = criterion(stud_emb, teach_emb)

            # Backward pass dengan gradient scaling
            scaler.scale(total_loss).backward()

            # Gradient clipping untuk stabilitas
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Metrics tracking
            train_loss += total_loss.item()
            train_cos_loss_total += cos_loss.item()
            train_mse_loss_total += mse_loss.item()
            cos_sim_total += F.cosine_similarity(stud_emb, teach_emb, dim=1).mean().item()
            l2_total += torch.norm(stud_emb - teach_emb, dim=1).mean().item()

        # Average training metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_cos = cos_sim_total / len(train_loader)
        avg_train_l2 = l2_total / len(train_loader)
        avg_train_cos_loss = train_cos_loss_total / len(train_loader)
        avg_train_mse_loss = train_mse_loss_total / len(train_loader)

        train_losses.append(avg_train_loss)
        train_cos_sims.append(avg_train_cos)
        train_l2s.append(avg_train_l2)

        # Validation
        student.eval()
        val_loss, cos_sim_total, l2_total = 0, 0, 0
        val_cos_loss_total, val_mse_loss_total = 0, 0

        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images = images.to(device)

                with autocast():
                    stud_emb = student(images)
                    teach_emb = get_teacher_embedding(images).to(device)
                    total_loss, cos_loss, mse_loss = criterion(stud_emb, teach_emb)

                val_loss += total_loss.item()
                val_cos_loss_total += cos_loss.item()
                val_mse_loss_total += mse_loss.item()
                cos_sim_total += F.cosine_similarity(stud_emb, teach_emb, dim=1).mean().item()
                l2_total += torch.norm(stud_emb - teach_emb, dim=1).mean().item()

        # Average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cos = cos_sim_total / len(val_loader)
        avg_val_l2 = l2_total / len(val_loader)
        avg_val_cos_loss = val_cos_loss_total / len(val_loader)
        avg_val_mse_loss = val_mse_loss_total / len(val_loader)

        val_losses.append(avg_val_loss)
        val_cos_sims.append(avg_val_cos)
        val_l2s.append(avg_val_l2)

        # Log the loss metric
        metrics = {
            f"train_loss_epoch_{epoch+1}": avg_train_loss,
            f"train_cos_sim_epoch_{epoch+1}": avg_train_cos,
            f"train_l2_epoch_{epoch+1}": avg_train_l2,
            f"train_cos_loss_epoch_{epoch+1}": avg_train_cos_loss,
            f"train_mse_loss_epoch_{epoch+1}": avg_train_mse_loss,
            f"val_loss_epoch_{epoch+1}": avg_val_loss,
            f"val_cos_sim_epoch_{epoch+1}": avg_val_cos,
            f"val_l2_epoch_{epoch+1}": avg_val_l2,
            f"val_cos_loss_epoch_{epoch+1}": avg_val_cos_loss,
            f"val_mse_loss_epoch_{epoch+1}": avg_val_mse_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        mlflow.log_metrics(metrics, step=epoch)

        print(f"Epoch {epoch+1:02d} | "
              f"Train: Loss={avg_train_loss:.4f}, CosSim={avg_train_cos:.4f}, L2={avg_train_l2:.4f}, "
              f"CosLoss={avg_train_cos_loss:.4f}, MSELoss={avg_train_mse_loss:.4f}")
        print(f"       | "
              f"Val: Loss={avg_val_loss:.4f}, CosSim={avg_val_cos:.4f}, L2={avg_val_l2:.4f}, "
              f"CosLoss={avg_val_cos_loss:.4f}, MSELoss={avg_val_mse_loss:.4f}")

        # Learning rate scheduling berdasarkan validation cosine similarity
        scheduler.step(avg_val_cos)

        # Early stopping berdasarkan cosine similarity
        if avg_val_cos > best_val_cos_sim:
            best_val_cos_sim = avg_val_cos
            torch.save(student.state_dict(), 'best_student.pth')
            print(f" New best model saved! Cosine Similarity: {best_val_cos_sim:.4f} ")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best validation cosine similarity: {best_val_cos_sim:.4f}")
                break

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_cos_sim": best_val_cos_sim,
        }, ckpt_path)

        # Log final metrics
        mlflow.log_metrics({
            "final_best_val_cos_sim": best_val_cos_sim,
            "final_train_loss": avg_train_loss,
            "final_val_loss": avg_val_loss
        })

        # Log model artifacts
        mlflow.pytorch.log_model(student, "final_model")
        mlflow.log_artifact(ckpt_path)

    return train_losses, val_losses, train_cos_sims, val_cos_sims, train_l2s, val_l2s

print("Active provider:", ort_session.get_providers())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

student = StudentResNet18(embedding_size=512, dropout=0.2).to(device)
criterion = CombinedLoss(alpha=0.7, beta=0.3, temperature=4.0)
optimizer = optim.AdamW(student.parameters(), lr=5e-5, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)

results = train(student, device, train_loader, val_loader, optimizer, criterion, epochs=10, resume=False)