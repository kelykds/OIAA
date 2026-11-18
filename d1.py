X = dados.drop(columns=["risco_evasao"])
y = dados["risco_evasao"]

X_treino, X_valida, y_treino, y_valida = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


  modelo = RandomForestClassifier(
      n_estimators=300,
      max_depth=9,
      random_state=42,
      n_jobs=-1,
      class_weight={0: 1, 1: 1.5, 2: 1},
      min_samples_leaf=2,
      criterion='gini',
      max_features='sqrt',
      )

print(f"Treino:    {len(X_treino)} estudantes")
print(f"Validação: {len(X_valida)} estudantes")
print(f"Modelo construído: {type(modelo).__name__}")

modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_valida)
print(f"Modelo ajustado. {len(previsoes)} previsões geradas para a validação.")


#              precision    recall  f1-score   support
#
#       Baixo      0.793     0.909     0.847       485
#       Médio      0.510     0.451     0.479       173
#        Alto      0.851     0.712     0.776       313
#
#    accuracy                          0.764       971
#   macro avg      0.718     0.691     0.700       971
#weighted avg      0.761     0.764     0.758       971
