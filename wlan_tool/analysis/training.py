# -*- coding: utf-8 -*-
"""
Funktionen zum Trainieren und Speichern von Machine-Learning-Modellen.
"""
import logging
import os
import tempfile
from typing import Callable, Optional

from ..storage import database
from ..storage.state import WifiAnalysisState
from . import analysis
from .models import ClientAutoencoder

logger = logging.getLogger(__name__)


import joblib
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn, optim

from .models import ClientAutoencoder


def train_model_from_confirmed_csv(
    confirmed_csv: str, feature_extractor: Callable, out_model: Optional[str] = None
) -> Optional[object]:
    if pd is None:
        raise RuntimeError("pandas/scikit-learn nicht installiert")

    import csv

    rows = []
    with open(confirmed_csv, newline="") as f:
        for line in csv.DictReader(f):
            ssid, bssid = line.get("ssid"), line.get("bssid")
            if feat := feature_extractor(ssid, bssid):
                feat["label"] = int(line.get("label", 0))
                rows.append(feat)

    if not rows:
        logger.error("Keine gültigen Zeilen für das Training.")
        return None

    df = pd.DataFrame(rows)
    X, y = df.drop(columns=["ssid", "bssid", "label"], errors="ignore"), df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model = CalibratedClassifierCV(pipe, cv=3, method="isotonic").fit(X_train, y_train)

    if out_model and joblib:
        try:
            joblib.dump(model, out_model)
            logger.info("Modell gespeichert in %s", out_model)
        except Exception as e:
            logger.exception("Fehler beim Speichern des Modells: %s", e)

    try:
        probs = model.predict_proba(X_test)[:, 1]
        auc, brier = roc_auc_score(y_test, probs), brier_score_loss(y_test, probs)
        logger.info(
            f"Modell-Evaluierung - Test AUC: {auc:.4f}, Brier Score: {brier:.4f}"
        )
    except Exception as e:
        logger.debug("Modell-Evaluierungsmetriken fehlgeschlagen: %s", e)
    return model


def auto_retrain_if_enough(
    label_db: str, min_confirmed: int, out_model: str, feature_extractor: Callable
):
    with tempfile.NamedTemporaryFile(
        prefix="labels_", suffix=".csv", delete=False, mode="w"
    ) as f:
        tmpf_name = f.name
    try:
        n = database.export_confirmed_to_csv(label_db, tmpf_name)
        if n >= min_confirmed:
            logger.info(
                "Genug Labels (%d >= %d). Training wird gestartet...", n, min_confirmed
            )
            return train_model_from_confirmed_csv(
                tmpf_name, feature_extractor, out_model=out_model
            )
        else:
            logger.info("Nicht genug Labels (%d < %d). Überspringe.", n, min_confirmed)
            return None
    finally:
        if os.path.exists(tmpf_name):
            os.unlink(tmpf_name)


def train_client_autoencoder(
    state: WifiAnalysisState,
    out_model_path: str,
    embedding_dim: int = 16,
    epochs: int = 50,
) -> bool:
    """
    Trainiert einen Autoencoder auf den Client-Features, um eine intelligente Dimensionsreduktion zu lernen.
    Das Ziel ist nicht Klassifikation, sondern das Lernen einer besseren Datenrepräsentation.
    """
    if pd is None or torch is None or ClientAutoencoder is None:
        logger.error(
            "torch, pandas oder scikit-learn nicht installiert. Autoencoder-Training abgebrochen."
        )
        return False

    logger.info("Starte Autoencoder-Training für Client-Features...")

    # 1. Daten vorbereiten
    df = analysis.prepare_client_feature_dataframe(state, correlate_macs=True)
    if df is None or df.empty:
        logger.warning("Keine Client-Daten für das Autoencoder-Training verfügbar.")
        return False

    X_df = df.drop(columns=["mac", "original_macs", "vendor"])

    # Autoencoder arbeiten oft besser mit Daten, die auf [0, 1] skaliert sind.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_df)

    X_tensor = torch.FloatTensor(X_scaled)
    input_dim = X_tensor.shape[1]

    # 2. Modell, Loss und Optimizer initialisieren
    model = ClientAutoencoder(input_dim=input_dim, embedding_dim=embedding_dim)
    criterion = (
        nn.MSELoss()
    )  # Mean Squared Error ist ein guter Loss für die Rekonstruktion
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3. Trainings-Loop
    logger.info("Trainiere Modell für %d Epochen...", epochs)
    for epoch in range(epochs):
        model.train()

        # Forward pass
        reconstructions = model(X_tensor)
        loss = criterion(reconstructions, X_tensor)

        # Backward pass und Optimierung
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoche [{epoch+1}/{epochs}], Rekonstruktions-Loss: {loss.item():.6f}"
            )

    # 4. Nur den trainierten Encoder speichern
    try:
        # Wir speichern nur den Encoder-Teil, da wir nur ihn für das Clustering brauchen
        torch.save(model.encoder.state_dict(), out_model_path)
        logger.info(
            f"Trainierter Client-Encoder erfolgreich gespeichert in: {out_model_path}"
        )
        return True
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Encoder-Modells: {e}")
        return False


def train_behavioral_model(
    event_db: str, label_db: str, out_model: str
) -> Optional[object]:
    """Trainiert ein Modell zur Klassifizierung von Gerätetypen basierend auf Verhalten."""
    if pd is None or RandomForestClassifier is None:
        raise RuntimeError("pandas/scikit-learn nicht installiert")

    # Lade den Zustand aus der Event-Datenbank
    with database.db_conn_ctx(event_db) as conn:
        events = list(database.fetch_events(conn))
    if not events:
        logger.error("Keine Events in der Event-Datenbank für das Training gefunden.")
        return None

    state = WifiAnalysisState()
    state.build_from_events(events)

    # Lade die gelabelten Clients
    with database.db_conn_ctx(label_db) as conn:
        try:
            labeled_clients = pd.read_sql_query(
                "SELECT mac, device_type FROM client_labels", conn
            )
        except pd.io.sql.DatabaseError:
            labeled_clients = pd.DataFrame()

    if labeled_clients.empty:
        logger.error(
            "Keine gelabelten Clients in der Label-Datenbank für das Training gefunden."
        )
        return None

    feature_rows = []
    labeled_macs_lower = {mac.lower() for mac in labeled_clients["mac"]}

    for mac, client_state in state.clients.items():
        if mac.lower() in labeled_macs_lower:
            if features := analysis.features_for_client_behavior(client_state):
                feature_rows.append(features)

    if not feature_rows:
        logger.error(
            "Konnte keine Verhaltens-Features für die gelabelten, im Scan gefundenen Clients extrahieren."
        )
        logger.error(
            "Stellen Sie sicher, dass Ihre events.db und labels.db synchron sind und genügend Aktivität erfasst wurde."
        )
        return None

    features_df = pd.DataFrame(feature_rows)

    # KORREKTUR: Sauberes Zusammenführen der Daten
    # Normalisiere MACs in beiden DataFrames für einen zuverlässigen Merge
    labeled_clients["mac_lower"] = labeled_clients["mac"].str.lower()
    features_df["mac_lower"] = features_df["mac"].str.lower()

    # Führe die DataFrames zusammen. Nur Clients, die in beiden vorhanden sind, bleiben übrig.
    training_data = pd.merge(
        features_df, labeled_clients, on="mac_lower", suffixes=("_features", "_labels")
    )

    if training_data.empty:
        logger.error("Trainingsdaten sind nach dem Zusammenführen leer.")
        return None

    target_column = (
        "device_type_labels"
        if "device_type_labels" in training_data.columns
        else "device_type"
    )
    y = training_data[target_column]

    non_feature_columns = [
        col
        for col in training_data.columns
        if "_labels" in col or "_features" in col or "mac" in col
    ]
    non_feature_columns.append("device_type")

    X = training_data.drop(columns=non_feature_columns, errors="ignore")

    if X.empty:
        logger.error(
            "Keine Feature-Spalten nach der Bereinigung übrig. Training nicht möglich."
        )
        return None

    # --- KORREKTUR: Robuste Aufteilung mit Garantie für seltene Klassen ---

    class_counts = y.value_counts()
    single_member_classes = class_counts[class_counts < 2].index.tolist()

    # Trenne die Daten: Die 'singles' und der Rest ('regulars')
    singles_indices = y[y.isin(single_member_classes)].index
    X_singles = X.loc[singles_indices]
    y_singles = y.loc[singles_indices]

    regulars_indices = y[~y.isin(single_member_classes)].index
    X_regulars = X.loc[regulars_indices]
    y_regulars = y.loc[regulars_indices]

    if single_member_classes:
        logger.warning(
            f"Die folgenden Klassen haben nur 1 Mitglied und werden dem Trainingsdatensatz fest zugewiesen: {single_member_classes}"
        )

    # Führe den stratifizierten Split nur auf den 'regulars' aus, falls möglich
    # Mindestens 8 Samples pro Klasse erforderlich für sinnvollen Test-Split
    total_samples = len(X_regulars) + len(X_singles)
    min_samples_for_split = 8

    if not X_regulars.empty and len(X_regulars) >= min_samples_for_split:
        # Prüfe, ob bei den regulars noch stratifiziert werden kann
        min_class_count = y_regulars.value_counts().min()
        test_size = 0.25

        # Stratifizierung nur möglich, wenn jede Klasse mindestens 4 Samples hat
        # (damit nach dem Split mindestens 3 im Training und 1 im Test landen)
        can_stratify = y_regulars.nunique() > 1 and min_class_count >= 4

        stratify_option = y_regulars if can_stratify else None

        if not can_stratify:
            logger.warning(
                f"Stratifizierung nicht möglich (kleinste Klasse hat {min_class_count} Samples, benötigt >= 4). Verwende zufälligen Split."
            )

        X_train_reg, X_test, y_train_reg, y_test = train_test_split(
            X_regulars,
            y_regulars,
            test_size=test_size,
            random_state=42,
            stratify=stratify_option,
        )

        # Füge die 'singles' zum Trainingsdatensatz hinzu
        X_train = pd.concat([X_train_reg, X_singles])
        y_train = pd.concat([y_train_reg, y_singles])

    else:
        # Zu wenig Daten für einen sinnvollen Split - nutze alle Daten fürs Training
        logger.warning(
            f"Zu wenig Daten für Test/Train-Split (nur {total_samples} Samples, benötigt >= {min_samples_for_split}). Alle Daten werden fürs Training verwendet."
        )
        logger.warning(
            "Für zuverlässige Modelle werden mindestens 8-10 Samples pro Gerätetyp empfohlen."
        )
        X_train = (
            pd.concat([X_regulars, X_singles]) if not X_regulars.empty else X_singles
        )
        y_train = (
            pd.concat([y_regulars, y_singles]) if not y_regulars.empty else y_singles
        )
        X_test = pd.DataFrame()
        y_test = pd.Series(dtype="object")

    logger.info(
        f"Trainingsdatensatz-Größe: {len(X_train)}, Testdatensatz-Größe: {len(X_test)}"
    )

    # Zeige Klassenverteilung
    if len(X_train) > 0:
        class_distribution = y_train.value_counts().to_dict()
        logger.info(f"Klassenverteilung im Training: {class_distribution}")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )
    model = pipe.fit(X_train, y_train)

    if out_model and joblib:
        try:
            joblib.dump(model, out_model)
            logger.info("Verhaltensmodell gespeichert in %s", out_model)
        except Exception as e:
            logger.exception("Fehler beim Speichern des Verhaltensmodells: %s", e)

    # Evaluiere das Modell nur, wenn es ein Test-Set gibt
    if not X_test.empty:
        try:
            score = model.score(X_test, y_test)
            logger.info(f"Modell-Evaluierung - Test-Genauigkeit: {score:.4f}")
        except Exception as e:
            logger.debug("Modell-Evaluierungsmetriken fehlgeschlagen: %s", e)

    return model
