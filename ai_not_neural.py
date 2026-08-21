import csv
import re
import sys
from collections import defaultdict, Counter

class DictionaryClassifier:
    """Классификатор на основе словарей частот слов."""
    
    NEGATIONS = {'не', 'ни', 'никакой', 'никто', 'ничего', 'нельзя', 'нет'}
    NEGATION_WINDOW = 3  # Окно действия отрицения: "не очень хороший" -> отрицение держится 3 слова

    def __init__(self, dataset_path: str):
        self.word_scores = defaultdict(Counter)
        self.labels = set()
        self._load_data(dataset_path)

    def _load_data(self, dataset_path: str):
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get('base', '').lower()
                    label = row.get('label')
                    if not label:
                        continue
                        
                    self.labels.add(label)
                    # Только кириллица
                    words = re.findall(r'[а-яё]+', text)
                    
                    for word in words:
                        if len(word) > 2:  # фильтр предлогов
                            self.word_scores[label][word] += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"Датасет '{dataset_path}' не найден.")

    def predict(self, text: str):
        words = re.findall(r'[а-яё]+', text.lower())
        scores = Counter({label: 0 for label in self.labels})
        
        negation_countdown = 0
        
        for word in words:
            if word in self.NEGATIONS:
                negation_countdown = self.NEGATION_WINDOW
                continue
                
            if len(word) <= 2:
                continue

            for label, word_counts in self.word_scores.items():
                if word in word_counts:
                    weight = word_counts[word]
                    is_negated = negation_countdown > 0
                    
                    # Инверсия работает только если в датасете есть pos и neg
                    if is_negated and label == 'positive' and 'negative' in self.labels:
                        scores['negative'] += weight
                    elif is_negated and label == 'negative' and 'positive' in self.labels:
                        scores['positive'] += weight
                    else:
                        scores[label] += weight

            if negation_countdown > 0:
                negation_countdown -= 1

        max_score = max(scores.values()) if scores else 0
        if max_score == 0:
            return 'neutral' if 'neutral' in self.labels else list(self.labels)[0]
            
        # Собираем классы, которые набрали максимум
        winners = [k for k, v in scores.items() if v == max_score]
        
        # При ничьей возвращаем neutral
        if len(winners) > 1:
            return 'neutral' if 'neutral' in self.labels else winners[0]
            
        return winners[0]


def main():
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "dataset.csv"
    
    try:
        clf = DictionaryClassifier(dataset_path)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        sys.exit(1)

    print("Анализатор запущен. Введите текст или 'exit' для выхода.")
    
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы.")
            break
            
        if text.lower() in ('exit', 'quit', 'выход'):
            break
        if not text:
            continue

        label = clf.predict(text)
        print(f"Тональность: {label.upper()}\n")

if __name__ == "__main__":
    main()