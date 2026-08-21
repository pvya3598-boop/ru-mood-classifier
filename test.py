import argparse
import os
import sys

import joblib
import numpy as np
import torch

from train import load_checkpoint, normalize

ICONS = {"positive": "[+]", "negative": "[-]", "neutral": "[=]"}
TESTS_FILE = "tests.txt"

_vectorizer = None
_model = None
_classes = None


def init_model():
    global _vectorizer, _model, _classes
    if _model is not None:
        return

    for f in ("mood_model.pth", "tfidf_vectorizer.pkl"):
        if not os.path.exists(f):
            sys.exit(f"Нет файла {f}. Сначала запустите generator.py и train.py")

    _vectorizer = joblib.load("tfidf_vectorizer.pkl")
    _model, _classes = load_checkpoint("mood_model.pth")
    _classes = list(_classes)


def short_label(label):
    return label[:3].upper()


def predict(texts):
    init_model()
    X = _vectorizer.transform(texts).toarray().astype(np.float32)
    with torch.no_grad():
        probs = torch.softmax(_model(torch.from_numpy(X)), dim=1).numpy()
    
    out = []
    for row in probs:
        i = int(row.argmax())
        out.append((_classes[i], float(row[i]), dict(zip(_classes, row))))
    return out


def mode_chat():
    print("\n чат")
    print("Пустая строка или 'exit' — выход.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in ("exit", "quit", "выход"):
            break

        label, conf, dist = predict([text])[0]
        print(f"  {ICONS.get(label, '[?]')} {label.upper()}  {conf*100:.1f}%")
        parts = " | ".join(f"{short_label(k)} {v*100:4.1f}%"
                           for k, v in sorted(dist.items(), key=lambda x: -x[1]))
        print(f"      {parts}")
        if conf < 0.55:
            print("      (модель не уверена)")
        print()


def parse_line(line):
    if " - " in line:
        text, _, tail = line.rpartition(" - ")
        tag = tail.strip().lower()
        mapping = {
            "pos": "positive", "neg": "negative", "neu": "neutral",
            "positive": "positive", "negative": "negative", "neutral": "neutral"
        }
        if tag in mapping:
            return text.strip(), mapping[tag]
    return line.strip(), None


def mode_file(path=TESTS_FILE):
    init_model()
    if not os.path.exists(path):
        print(f"Файл {path} не найден.")
        return

    with open(path, encoding="utf-8") as f:
        raw = [ln.strip() for ln in f
               if ln.strip() and not ln.lstrip().startswith("#")]

    if not raw:
        print(f"Файл {path} пуст.")
        return

    items = [parse_line(ln) for ln in raw]
    texts = [t for t, _ in items]
    golds = [g for _, g in items]
    results = predict(texts)

    print(f"\nбенчмарк файла: {path} ({len(texts)} фраз) ===\n")

    labeled = sum(1 for g in golds if g)
    correct = 0
    errors = []
    per_class = {c: [0, 0] for c in _classes}

    for i, (text, result, gold) in enumerate(zip(texts, results, golds), 1):
        label, conf, _ = result
        short = text if len(text) <= 52 else text[:49] + "..."
        if gold:
            ok = (label == gold)
            correct += ok
            per_class[gold][1] += 1
            per_class[gold][0] += ok
            mark = "OK " if ok else "ОШ "
            exp = "" if ok else f"  (ждали {short_label(gold)})"
            print(f"{i:02d}. {mark} {ICONS.get(label,'[?]')} {short_label(label)} {conf*100:5.1f}%  {short}{exp}")
            if not ok:
                errors.append((i, text, gold, label, conf))
        else:
            print(f"{i:02d}.     {ICONS.get(label,'[?]')} {short_label(label)} {conf*100:5.1f}%  {short}")

    print("\n" + "=" * 70)
    if labeled:
        print(f"точность: {correct}/{labeled} = {correct/labeled*100:.1f}%")
        print("\nпо классам:")
        for c in _classes:
            ok, tot = per_class[c]
            if tot:
                print(f"  {short_label(c)}  {ok}/{tot}  ({ok/tot*100:3.0f}%)")
        if errors:
            print(f"\nошибки ({len(errors)}):")
            for i, text, gold, label, conf in errors:
                print(f"  {i:02d}. ждали {short_label(gold)}, получили {short_label(label)} ({conf*100:.0f}%)")
                print(f"      {text[:90]}")
    else:
        counts = {}
        for label, _, _ in results:
            counts[label] = counts.get(label, 0) + 1
        print("меток в файле нет, просто распределение:")
        for c in _classes:
            n = counts.get(c, 0)
            print(f"  {short_label(c)}: {n} ({n/len(texts)*100:.0f}%)")
    avg = float(np.mean([c for _, c, _ in results]))
    print(f"\nсредняя уверенность: {avg*100:.1f}%")
    print("=" * 70)


def menu():
    print("=" * 46)
    print("  определение настроения")
    print("=" * 46)
    print("  1 — чат (писать сообщения вручную)")
    print(f"  2 — прогон файла {TESTS_FILE}")
    print("  0 — выход")
    print("=" * 46)
    while True:
        try:
            choice = input("Режим: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice == "1":
            mode_chat()
            return
        if choice == "2":
            mode_file()
            return
        if choice in ("0", "", "exit", "quit"):
            return
        print("Введите 1, 2 или 0.")


def main():
    parser = argparse.ArgumentParser(description="Тестирование модели настроения")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--chat", action="store_true", help="Запустить чат")
    group.add_argument("--file", nargs="?", const=TESTS_FILE, help="Прогнать файл с тестами")
    group.add_argument("--text", type=str, help="Разобрать одну фразу")
    
    args = parser.parse_args()

    if args.chat:
        mode_chat()
    elif args.file:
        mode_file(args.file)
    elif args.text:
        label, conf, dist = predict([args.text])[0]
        print(f"{ICONS.get(label, '[?]')} {label.upper()}  {conf*100:.1f}%")
        print("   " + " | ".join(f"{short_label(k)} {v*100:.1f}%"
                                 for k, v in sorted(dist.items(), key=lambda x: -x[1])))
    else:
        menu()


if __name__ == "__main__":
    main()