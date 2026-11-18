from sklearn.ensemble import VotingClassifier

X = X_caract[colunas]
y = rotulos


X_treino, X_valida, y_treino, y_valida = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)


if modelo_escolhido == "Regressao Logistica":
    nucleo = LogisticRegression(
        C=forca_C,
        max_iter=3000,
        random_state=42,
        class_weight='balanced',
        )

elif modelo_escolhido == "Floresta Aleatoria":
    nucleo = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        min_samples_leaf=2,
        min_samples_split=2,
        criterion='gini',
        max_features='sqrt',
        )

else:
    nucleo = SVC(
        C=forca_C,
        kernel='rbf',
        random_state=19,
        class_weight='balanced',
        probability=True,
        gamma='scale'
        )

modelo = make_pipeline(StandardScaler(), nucleo) if padronizar else nucleo

svm_base = make_pipeline( StandardScaler(),
    SVC(
        C=10,
        kernel='rbf',
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42)
)

rf_base = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,

)

lr_base = make_pipeline(
    StandardScaler(),
    LogisticRegression(
      C=10,
      max_iter=3000,
      class_weight='balanced',
      random_state=19),
)

ensemble = VotingClassifier(
    estimators=[('svm', svm_base), ('rf', rf_base), ('lr', lr_base)],
    voting='soft',
    weights=[2, 2, 1],
    #SVM e RF tiveram bons resultados, por isso tem um maior peso em comparação com a LR
    n_jobs=-1,

) 
modelo = make_pipeline(StandardScaler(), ensemble) if padronizar else ensemble

print(f"Treino:    {len(X_treino)} imagens")
print(f"Validação: {len(X_valida)} imagens")
print(f"Modelo construído: {type(nucleo).__name__}"
      f"{' dentro de um Pipeline com StandardScaler' if padronizar else ''}")

print(f"Modelo usado: VotingClassifier (SVM + RandomForest + LogisticRegression) dentro de um Pipeline com StdScaler.")

modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_valida)
print(f"Modelo ajustado. {len(previsoes)} previsões geradas para a validação.")

#               precision    recall  f1-score   support

#       metal      0.684     0.586     0.631       111
#       papel      0.778     0.812     0.795       160
#     papelao      0.743     0.771     0.757       109
#    plastico      0.734     0.700     0.717       130
#       vidro      0.678     0.733     0.705       135

#    accuracy                          0.727       645
#   macro avg      0.724     0.720     0.721       645
#weighted avg      0.726     0.727     0.726       645
