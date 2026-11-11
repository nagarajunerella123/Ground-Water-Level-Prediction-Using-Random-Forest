# groundwater_modeling_with_qga_lstm_shap.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from tkinter import filedialog, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import threading  # for long SHAP plotting to avoid blocking GUI (but we will call safely)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# -----------------------
# Global variables
# -----------------------
main = tkinter.Tk()
main.title("Ground Water Modeling - RF, DNN, LSTM, QGA, SHAP")
main.geometry("1250x800")

data = None
X = None
y = None
X_train = X_test = None
y_train = y_test = None
y_train_dnn = y_test_dnn = None

rfc_model = None
dnn_model = None
lstm_model = None

r_acc = None
dnn_acc = None
lstm_acc = None

selected_features_mask = None  # boolean mask of selected features
selected_feature_indices = None

# -----------------------
# Utility functions
# -----------------------
def safe_read_csv(path):
    # safe path handling
    return pd.read_csv(path)

def show_text(msg):
    text.delete('1.0', END)
    text.insert(END, msg)

# -----------------------
# Upload dataset
# -----------------------
def upload():
    global data
    try:
        filename = filedialog.askopenfilename(initialdir=".", title="Select Ground Water CSV file",
                                              filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not filename:
            return
        data = safe_read_csv(filename)
        show_text(f"Loaded dataset: {os.path.basename(filename)}\n\nColumns:\n{data.columns.tolist()}\n\nPreview:\n{data.head().to_string()}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load file: {e}")

# -----------------------
# Preprocess: map labels & split
# -----------------------
def preprocess():
    """
    Expects dataset to have a 'Situation' column (categorical EXCESS, SEMICRITICAL, MODERATED, CRITICAL)
    and feature columns in positions 1..10 (or adjust as needed).
    This function:
    - maps Situation to integers 0..3
    - splits into train/test
    - creates one-hot labels for DNN/LSTM
    """
    global data, X, y, X_train, X_test, y_train, y_test, y_train_dnn, y_test_dnn, selected_features_mask, selected_feature_indices

    if data is None:
        messagebox.showwarning("No data", "Please upload dataset first.")
        return

    try:
        # Map 'Situation' to numeric labels
        mapping = {'EXCESS': 1, 'SEMICRITICAL': 2, 'MODERATED': 3, 'CRITICAL': 4}
        if 'Situation' not in data.columns:
            messagebox.showerror("Column missing", "Dataset must contain a 'Situation' column.")
            return
        data = data.copy()
        data['Situation_mapped'] = data['Situation'].map(mapping)
        if data['Situation_mapped'].isnull().any():
            messagebox.showwarning("Mapping warning", "Some Situation values could not be mapped. Check dataset.")

        # Select features: columns 1 through 10 (adjust if needed)
        # If your dataset feature columns are differently named, modify below.
        # We'll try to take the 10 columns after the first column, or fallback to numeric columns.
        if data.shape[1] >= 12:
            # Assume the features are in columns index 1..10 inclusive
            X = data.iloc[:, 1:11].values
            feature_names = data.columns[1:11].tolist()
        else:
            # fallback: take all numeric columns except the mapped label
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != 'Situation_mapped']
            X = data[numeric_cols].values
            feature_names = numeric_cols

        y_raw = data['Situation_mapped'].values.astype(int)

        # make labels zero-based for to_categorical (so classes 0..3)
        y_zero_based = y_raw - 1

        # shuffle dataset then split
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y_zero_based = y_zero_based[indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y_zero_based, test_size=0.2, random_state=2)

        # For DNN/LSTM, we need one-hot labels
        y_train_dnn = to_categorical(y_train, num_classes=4)
        y_test_dnn = to_categorical(y_test, num_classes=4)

        selected_features_mask = np.ones(X.shape[1], dtype=bool)
        selected_feature_indices = list(range(X.shape[1]))

        msg = f"Preprocessing completed.\nTotal records: {len(data)}\nFeatures used: {len(selected_feature_indices)} -> {feature_names}\nTraining samples: {len(X_train)}\nTesting samples: {len(X_test)}"
        show_text(msg)
    except Exception as e:
        messagebox.showerror("Preprocess error", str(e))

# -----------------------
# Simple Genetic Algorithm for feature selection (QGA-like)
# -----------------------
def evaluate_mask_acc(mask, Xtr, Xv, ytr, yv):
    """Train a small RandomForest on selected features and return validation accuracy."""
    if mask.sum() == 0:
        return 0.0
    try:
        clf = RandomForestClassifier(n_estimators=30, random_state=42)
        clf.fit(Xtr[:, mask], ytr)
        preds = clf.predict(Xv[:, mask])
        return accuracy_score(yv, preds)
    except Exception:
        return 0.0

def runQGA(iterations=30, pop_size=30, crossover_rate=0.8, mutation_rate=0.02):
    """
    Simple binary genetic algorithm for selecting feature subset.
    This function updates global selected_features_mask and selected_feature_indices.
    """
    global X_train, X_test, y_train, y_test, selected_features_mask, selected_feature_indices, rfc_model

    if X_train is None:
        messagebox.showwarning("No data", "Run preprocessing first.")
        return

    n_features = X_train.shape[1]
    # initialize population: random binary masks
    population = np.random.randint(0, 2, size=(pop_size, n_features), dtype=int)

    # Keep a small validation split from training for evaluating GA candidates
    Xtr_ga, Xval_ga, ytr_ga, yval_ga = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    best_mask = None
    best_score = -1.0

    for gen in range(iterations):
        # evaluate fitness (accuracy)
        scores = []
        for i in range(pop_size):
            mask = population[i].astype(bool)
            acc = evaluate_mask_acc(mask, Xtr_ga, Xval_ga, ytr_ga, yval_ga)
            scores.append(acc)
            if acc > best_score:
                best_score = acc
                best_mask = mask.copy()
        scores = np.array(scores)

        # Selection: tournament selection
        new_pop = []
        while len(new_pop) < pop_size:
            # pick two parents via tournament
            idx1, idx2 = np.random.randint(0, pop_size, 2)
            p1 = population[idx1] if scores[idx1] >= scores[idx2] else population[idx2]
            idx3, idx4 = np.random.randint(0, pop_size, 2)
            p2 = population[idx3] if scores[idx3] >= scores[idx4] else population[idx4]

            # Crossover
            if np.random.rand() < crossover_rate:
                pt = np.random.randint(1, n_features - 1)
                child1 = np.concatenate([p1[:pt], p2[pt:]])
                child2 = np.concatenate([p2[:pt], p1[pt:]])
            else:
                child1 = p1.copy()
                child2 = p2.copy()

            # Mutation
            for child in (child1, child2):
                for bit in range(n_features):
                    if np.random.rand() < mutation_rate:
                        child[bit] = 1 - child[bit]

                new_pop.append(child)
                if len(new_pop) >= pop_size:
                    break

        population = np.array(new_pop)

    # finished GA
    if best_mask is None:
        best_mask = np.ones(n_features, dtype=bool)
    selected_features_mask = best_mask
    selected_feature_indices = [i for i, v in enumerate(best_mask) if v]

    # Optionally retrain an RF on full training set with selected features
    try:
        rfc_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc_model.fit(X_train[:, selected_features_mask], y_train)
        r_acc_local = rfc_model.score(X_test[:, selected_features_mask], y_test) * 100
    except Exception:
        r_acc_local = None

    msg = f"QGA finished. Selected {selected_features_mask.sum()} / {n_features} features.\nIndices: {selected_feature_indices}\n"
    if r_acc_local is not None:
        msg += f"RandomForest accuracy using selected features on test: {r_acc_local:.2f}%\n"
    show_text(msg)

# -----------------------
# Random Forest
# -----------------------
def runRForest():
    global rfc_model, r_acc
    if X_train is None:
        messagebox.showwarning("No data", "Run preprocessing first.")
        return
    try:
        # Use selected features if QGA has been run
        mask = selected_features_mask if selected_features_mask is not None else np.ones(X_train.shape[1], dtype=bool)
        rfc_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rfc_model.fit(X_train[:, mask], y_train)
        pred = rfc_model.predict(X_test[:, mask])
        r_acc = accuracy_score(y_test, pred) * 100
        show_text(f"RandomForest trained. Test Accuracy: {r_acc:.2f}%")
    except Exception as e:
        messagebox.showerror("RandomForest error", str(e))

# -----------------------
# Dense Neural Network (DNN)
# -----------------------
def runDNN(epochs=50, batch_size=16):
    global dnn_model, dnn_acc
    if X_train is None:
        messagebox.showwarning("No data", "Run preprocessing first.")
        return
    try:
        mask = selected_features_mask if selected_features_mask is not None else np.ones(X_train.shape[1], dtype=bool)
        input_dim = int(mask.sum())

        # build model
        dnn_model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(4, activation='softmax')
        ])
        dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # prepare data
        Xtr = X_train[:, mask]
        Xte = X_test[:, mask]
        # y_train_dnn and y_test_dnn are global one-hot labels
        dnn_model.fit(Xtr, y_train_dnn, epochs=epochs, batch_size=batch_size, verbose=0)
        results = dnn_model.evaluate(Xte, y_test_dnn, verbose=0)
        dnn_acc = results[1] * 100
        show_text(f"DNN trained. Test Accuracy: {dnn_acc:.2f}%")
    except Exception as e:
        messagebox.showerror("DNN error", str(e))

# -----------------------
# LSTM Model
# -----------------------
def runLSTM(epochs=40, batch_size=16):
    global lstm_model, lstm_acc
    if X_train is None:
        messagebox.showwarning("No data", "Run preprocessing first.")
        return
    try:
        mask = selected_features_mask if selected_features_mask is not None else np.ones(X_train.shape[1], dtype=bool)
        n_feats = int(mask.sum())

        # reshape to (samples, timesteps, features). We'll use timesteps = 1 (if no real multi-step sequences)
        Xtr = X_train[:, mask].reshape((X_train.shape[0], 1, n_feats))
        Xte = X_test[:, mask].reshape((X_test.shape[0], 1, n_feats))

        lstm_model = Sequential()
        lstm_model.add(LSTM(64, input_shape=(1, n_feats)))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(32, activation='relu'))
        lstm_model.add(Dense(4, activation='softmax'))
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        lstm_model.fit(Xtr, y_train_dnn, epochs=epochs, batch_size=batch_size, verbose=0)
        results = lstm_model.evaluate(Xte, y_test_dnn, verbose=0)
        lstm_acc = results[1] * 100
        show_text(f"LSTM trained. Test Accuracy: {lstm_acc:.2f}%")
    except Exception as e:
        messagebox.showerror("LSTM error", str(e))

# -----------------------
# Predict on new CSV
# -----------------------
def runPredict():
    """
    Expects a CSV with same feature columns in the same order.
    Uses selected features mask if available. Uses the best available model in order: DNN -> LSTM -> RF
    """
    global dnn_model, lstm_model, rfc_model
    if X_train is None:
        messagebox.showwarning("No data", "Run preprocessing first.")
        return
    filename = filedialog.askopenfilename(initialdir=".", title="Select Test CSV file",
                                          filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if not filename:
        return
    try:
        test_df = pd.read_csv(filename)
        # assume relevant features are the first n features used originally (we used columns 1..10)
        # We'll try to pick numeric columns like in preprocess
        if data is not None and data.shape[1] >= 12:
            test_X = test_df.iloc[:, 1:11].values
        else:
            numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            test_X = test_df[numeric_cols].values

        mask = selected_features_mask if selected_features_mask is not None else np.ones(test_X.shape[1], dtype=bool)
        test_X_sel = test_X[:, mask]

        # choose model for prediction (DNN preferred)
        if dnn_model is not None:
            preds_proba = dnn_model.predict(test_X_sel)
            preds = np.argmax(preds_proba, axis=1)
        elif lstm_model is not None:
            test_X_lstm = test_X_sel.reshape((test_X_sel.shape[0], 1, test_X_sel.shape[1]))
            preds_proba = lstm_model.predict(test_X_lstm)
            preds = np.argmax(preds_proba, axis=1)
        elif rfc_model is not None:
            preds = rfc_model.predict(test_X_sel)
        else:
            messagebox.showwarning("No model", "Train at least one model first (RF or DNN or LSTM).")
            return

        # map back to human messages (remember we used zero-based labels: 0..3 -> add 1)
        mapping_back = {0: 'EXCESS (mapped -> 1)', 1: 'SEMICRITICAL (2)', 2: 'MODERATED (3)', 3: 'CRITICAL (4)'}
        output_lines = []
        for i in range(len(test_X_sel)):
            label = int(preds[i])
            output_lines.append(f"Input {i}: Predicted Situation = {mapping_back.get(label, str(label))}")

        show_text("Predictions:\n" + "\n".join(output_lines))
    except Exception as e:
        messagebox.showerror("Predict error", str(e))

# -----------------------
# SHAP explainability
# -----------------------
def runSHAP():
    """
    Runs SHAP summary plot for RandomForest (fast) if RF model exists.
    If no RF but DNN exists, uses KernelExplainer (slow) as fallback.
    """
    global rfc_model, dnn_model, X_train

    if X_train is None:
        messagebox.showwarning("No data", "Run preprocessing first.")
        return

    if rfc_model is None and dnn_model is None:
        messagebox.showwarning("No model", "Train RandomForest or DNN first to run SHAP.")
        return

    # SHAP plotting can be slow/blocking. We'll run it in a separate thread to avoid freezing UI.
    def shap_worker():
        try:
            mask = selected_features_mask if selected_features_mask is not None else np.ones(X_train.shape[1], dtype=bool)
            X_for_shap = X_train[:, mask]
            feature_names = [f"F{i}" for i in range(X_for_shap.shape[1])]
            if rfc_model is not None:
                explainer = shap.TreeExplainer(rfc_model)
                shap_values = explainer.shap_values(X_for_shap)
                # summary_plot uses matplotlib
                plt.figure()
                shap.summary_plot(shap_values, X_for_shap, feature_names=feature_names, show=True)
            else:
                # fallback for DNN: use a small background dataset for KernelExplainer (slow)
                background = X_for_shap[np.random.choice(X_for_shap.shape[0], min(50, X_for_shap.shape[0]), replace=False)]
                explainer = shap.KernelExplainer(lambda x: dnn_model.predict(x), background)
                shap_values = explainer.shap_values(X_for_shap[:50], nsamples=100)
                plt.figure()
                shap.summary_plot(shap_values, X_for_shap[:50], feature_names=feature_names, show=True)
        except Exception as e:
            messagebox.showerror("SHAP error", str(e))

    t = threading.Thread(target=shap_worker)
    t.start()

# -----------------------
# Graph comparison
# -----------------------
def runGraph():
    global r_acc, dnn_acc, lstm_acc
    bars = []
    heights = []
    labels = []
    if r_acc is not None:
        labels.append('RandomForest')
        heights.append(r_acc)
    if dnn_acc is not None:
        labels.append('DNN')
        heights.append(dnn_acc)
    if lstm_acc is not None:
        labels.append('LSTM')
        heights.append(lstm_acc)
    if not labels:
        messagebox.showwarning("No results", "Train models first to see comparison.")
        return
    plt.figure(figsize=(8,5))
    x_pos = np.arange(len(labels))
    plt.bar(x_pos, heights)
    plt.xticks(x_pos, labels)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Comparison')
    for i, v in enumerate(heights):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    plt.ylim(0, 100)
    plt.show()

# -----------------------
# Buttons & GUI layout
# -----------------------
font = ('times', 14, 'bold')
title = Label(main, text='Groundwater Level Prediction — RF, DNN, LSTM, QGA, SHAP')
title.config(bg='#222E50', fg='white')
title.config(font=font)
title.config(height=2, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Ground Water Dataset", command=upload)
uploadButton.place(x=20, y=80)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess & Split Data", command=preprocess)
preprocessButton.place(x=280, y=80)
preprocessButton.config(font=font1)

qgaButton = Button(main, text="Run QGA (Feature Selection)", command=lambda: runQGA(iterations=40, pop_size=40))
qgaButton.place(x=520, y=80)
qgaButton.config(font=font1)

rfButton = Button(main, text="Train Random Forest", command=runRForest)
rfButton.place(x=780, y=80)
rfButton.config(font=font1)

dnnButton = Button(main, text="Train DNN (Dense NN)", command=lambda: runDNN(epochs=60, batch_size=16))
dnnButton.place(x=20, y=140)
dnnButton.config(font=font1)

lstmButton = Button(main, text="Train LSTM", command=lambda: runLSTM(epochs=40, batch_size=16))
lstmButton.place(x=280, y=140)
lstmButton.config(font=font1)

predictButton = Button(main, text="Upload Test Data & Predict", command=runPredict)
predictButton.place(x=520, y=140)
predictButton.config(font=font1)

shapButton = Button(main, text="Run SHAP Explainability", command=runSHAP)
shapButton.place(x=780, y=140)
shapButton.config(font=font1)

graphButton = Button(main, text="Show Accuracy Comparison Graph", command=runGraph)
graphButton.place(x=1020, y=140)
graphButton.config(font=font1)

text = Text(main, height=22, width=145)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=220)
text.config(font=('times', 11))

main.config(bg='#A7D7A7')

# Start GUI loop
main.mainloop()
