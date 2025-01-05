import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Încarcă fișierele 
train_path = 'train.csv'
test_path = 'test.csv'

# Încarcă fișierele CSV
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Verifică dimensiunea fișierelor
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# Afișează primele 5 rânduri 
print("Train data preview:")
print(train_data.head())  
print("Test data preview:")
print(test_data.head())

# Extragerea coloanelelor de text și etichete
X_train = train_data['Text']
y_train = train_data['Label']

# Definirea unei liste de stopwords în limba franceză
stop_words_fr = [
    "au", "avec", "pour", "vous", "et", "de", "le", "la", "les", "du", "des", "en", "une", "un", 
    "l'", "pour", "ce", "dans", "que", "à", "on", "non", "mais", "est", "sont", "cette", "par", 
    "plus", "ne", "au", "aux", "auprès", "lors", "même", "comme", "en", "chaque", "tout", "tous", 
    "de", "cet", "il", "la", "soit", "a", "deja", "très"
]

# Aplicăm vectorizarea textului
vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words_fr)
X_train_vec = vectorizer.fit_transform(X_train)

# Inițializarea și antrenarea a modelului de regresie logistică
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Facem predicții
X_test = test_data['Text']
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

# Adăugăm predicțiile în fișierul de test
test_data['Label'] = y_pred

# Salvăm fișierul completat 
test_output_path = 'test.csv'
test_data.to_csv(test_output_path, index=False, encoding='utf-8')

print(f"Fisierul completat a fost salvat: {test_output_path}".encode('utf-8'))

print("\nEvaluarea modelului pe setul de testare:")

# Verifică dacă există o coloană `Label` reală în setul de test
if 'Label' in test_data.columns:
    print(classification_report(test_data['Label'], y_pred))
else:
    print("Etichetele reale nu sunt disponibile în fișierul de test.")
