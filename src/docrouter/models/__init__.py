"""One module per member's ML experiment.

Shared contract so every model plugs into the same evaluation and API:
    train(train_df, val_df) -> model
    model.classes_                      list of labels
    model.predict_proba(list[str])      array (n_docs, n_labels)
    model.predict(list[str])            array of labels
Log every run to MLflow (params, metrics from docrouter.evaluate, the model).
"""
