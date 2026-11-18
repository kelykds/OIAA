X = dados["mensagem"]
y = dados["rotulo"]

X_treino, X_valida, y_treino, y_valida = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)


vetorizador = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,3), 
    min_df=3,
    sublinear_tf=True,
    )



  modelo = LogisticRegression(
      C=10.01,
      max_iter=1000,
      random_state=42,
      class_weight='balanced'
      )




print(f"Treino:    {len(X_treino)} avaliações")
print(f"Validação: {len(X_valida)} avaliações")
print(f"Vetorizador: {type(vetorizador).__name__}  |  Classificador: {type(modelo).__name__}")

X_treino_vec = vetorizador.fit_transform(X_treino)
X_valida_vec = vetorizador.transform(X_valida)

modelo.fit(X_treino_vec, y_treino)
previsoes = modelo.predict(X_valida_vec)

print(f"Vocabulário aprendido: {len(vetorizador.vocabulary_)} termos")
print(f"Previsões geradas: {len(previsoes)}")


#               precision    recall  f1-score   support
#
#Insatisfeito      0.882     0.944     0.912      8084
#  Satisfeito      0.976     0.948     0.962     19412
#
#    accuracy                          0.947     27496
#   macro avg      0.929     0.946     0.937     27496
#weighted avg      0.949     0.947     0.947     27496
