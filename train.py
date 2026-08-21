import re
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

SEED = 2
EPOCHS = 60
BATCH_SIZE = 64
VAL_BATCH_SIZE = 256
PATIENCE = 8
LABEL_SMOOTHING = 0.05


def normalize(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def build_vectorizer(word_max=30000, char_max=50000, min_df=2):
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=word_max,
        min_df=min_df,
        sublinear_tf=True,
        preprocessor=normalize,
        token_pattern=r"(?u)\b\w+\b",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=char_max,
        min_df=min_df,
        sublinear_tf=True,
        preprocessor=normalize,
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


class SentimentNet(nn.Module):
    def __init__(self, input_size, num_classes=3, hidden_dims=(512, 128), dropouts=(0.5, 0.3)):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        layers = []
        prev_size = input_size
        for h, d in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(d))
            prev_size = h
            
        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SparseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def sparse_collate(batch):
    x_batch = scipy.sparse.vstack([item[0] for item in batch])
    y_batch = torch.stack([item[1] for item in batch])
    return torch.from_numpy(x_batch.toarray()).float(), y_batch


def save_checkpoint(path, model, classes):
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": model.input_size,
        "num_classes": model.num_classes,
        "classes": list(classes),
    }, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = SentimentNet(
        input_size=ck["input_size"],
        num_classes=ck["num_classes"],
    )
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    model.eval()
    return model, ck["classes"]


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, trues = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            out = model(x_batch)
            losses.append(criterion(out, y_batch).item() * x_batch.size(0))
            preds.append(out.argmax(1).cpu().numpy())
            trues.append(y_batch.cpu().numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    # Считаем точный средний лосс по всем сэмплам, а не по батчам
    avg_loss = np.sum(losses) / len(trues)
    return avg_loss, f1_score(trues, preds, average="macro"), preds, trues


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    df = pd.read_csv("dataset.csv")
    print(f"Загружено строк: {len(df)}")

    if "base" not in df.columns:
        raise ValueError("Отсутствует колонка 'base'.")

    groups = df["base"].astype(str)
    print(f"Уникальных смыслов: {groups.nunique()}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(gss.split(df, y, groups=groups))

    train_texts = df.iloc[train_idx]["text"].astype(str)
    val_texts = df.iloc[val_idx]["text"].astype(str)
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"Обучающая выборка: {len(train_idx)}, Валидационная: {len(val_idx)}")

    print("\nВекторизация (word 1-2 + char_wb 3-5)...")
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    
    X_train = X_train.astype(np.float32).tocsr()
    X_val = X_val.astype(np.float32).tocsr()
    
    print(f"Размерность признаков: {X_train.shape[1]}")
    
    train_loader = DataLoader(
        SparseDataset(X_train, y_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=sparse_collate,
        pin_memory=True
    )
    val_loader = DataLoader(
        SparseDataset(X_val, y_val), 
        batch_size=VAL_BATCH_SIZE, 
        collate_fn=sparse_collate,
        pin_memory=True
    )

    model = SentimentNet(X_train.shape[1], num_classes=len(label_encoder.classes_)).to(device)

    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    print(f"Веса классов: {dict(zip(label_encoder.classes_, cw.round(2)))}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float32).to(device),
        label_smoothing=LABEL_SMOOTHING
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    print("\nОбучение...")
    best_f1 = -1.0
    best_state = None
    best_preds, best_trues = None, None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)

        train_loss = running_loss / len(y_train)
        val_loss, val_f1, preds, trues = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_preds, best_trues = preds, trues
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Эпоха {epoch}/{EPOCHS} - Loss: {train_loss:.4f}, Val F1: {val_f1*100:.2f}%")

        if epochs_no_improve >= PATIENCE:
            print(f"\nРанняя остановка на эпохе {epoch}.")
            break

    if best_state:
        model.load_state_dict(best_state)
        print(f"Восстановлены веса лучшей эпохи (macro-F1 {best_f1*100:.2f}%).")

    save_checkpoint("mood_model.pth", model, label_encoder.classes_)
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    
    print("\n" + "=" * 40)
    print("Отчет классификации:")
    print("=" * 40)
    print(classification_report(best_trues, best_preds, target_names=label_encoder.classes_, digits=3))
    print("\nМатрица ошибок:")
    print(pd.DataFrame(confusion_matrix(best_trues, best_preds),
                       index=label_encoder.classes_, columns=label_encoder.classes_))
    print("\nГотово.")


if __name__ == "__main__":
    main()