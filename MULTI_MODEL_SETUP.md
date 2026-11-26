# Multi-model integration

The application now supports three classifiers in the same desktop GUI:

- Random Forest (RF)
- Support Vector Machine (SVM, RBF kernel)
- K-Nearest Neighbors (KNN)

## Train the models before launching or packaging

From the repository root, with the project's Python dependencies installed:

```bash
python src/train_all_models.py
```

This reads `data/processed/transformer_faults.csv` and writes these artifacts into `models/`:

- `random_forest_model.joblib`
- `svm_model.joblib`
- `knn_model.joblib`
- `label_encoder.joblib`

It also refreshes `transformer_fault_model.joblib` to preserve compatibility with the original app filename.

Then launch the application using the project's usual run command. The model selector chooses which classifier is used for manual and batch predictions. **COMPARE ALL MODELS** runs the same entered gas measurements through all three classifiers and displays each prediction and its model probability.

## Preprocessing

All three pipelines apply `log1p(x)` to the five gas concentration inputs. SVM and KNN additionally use `StandardScaler`, because distance and margin-based methods are sensitive to feature scale. Random Forest does not require feature scaling.

SVM uses an RBF kernel and `probability=True` so the GUI can display class probability estimates. KNN uses five neighbors and distance weighting.

## Rebuild installers

Run the training command first so all three model artifacts exist. Then rebuild the PyInstaller application using `TransformerFaultML.spec`, and rebuild the Linux Debian package using the repository's existing packaging procedure. The spec now bundles all three model files.

The model probability shown in the GUI is an estimate produced by the classifier; it is not a calibrated guarantee of the physical fault or transformer condition.
