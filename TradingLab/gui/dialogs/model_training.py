# Addestramento modelli

# Addestramento modelli

"""
Finestra di dialogo per la configurazione e l'esecuzione dell'addestramento di modelli di ML.
Permette di selezionare dati, configurare i parametri e monitorare l'addestramento.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout,
    QFrame, QTabWidget, QSizePolicy, QScrollArea, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox, QProgressBar, 
    QFileDialog, QLineEdit, QTextEdit, QListWidget, QListWidgetItem,
    QWidget
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QIcon

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional, Union, Tuple, Any

from ..controls import (
    SymbolSelector, TimeframeSelector, DateRangeSelector
)
from ..styles import style_manager
from ...utils import app_logger


class TrainingThread(QThread):
    """Thread per eseguire l'addestramento in background."""
    
    # Segnali
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il thread di addestramento.
        
        Args:
            config: Dizionario con la configurazione dell'addestramento
        """
        super().__init__()
        self.config = config
    
    def run(self):
        """Esegue l'addestramento in background."""
        try:
            # Simulazione dell'addestramento
            total_epochs = self.config.get('epochs', 100)
            
            # Simula le fasi dell'addestramento
            # 1. Preparazione dati
            self.progress.emit(5, "Caricamento dataset...")
            self.msleep(1000)
            
            # 2. Preprocessing
            self.progress.emit(10, "Preprocessing dei dati...")
            self.msleep(1500)
            
            # 3. Partizionamento train/validation/test
            self.progress.emit(15, "Suddivisione del dataset...")
            self.msleep(500)
            
            # 4. Costruzione del modello
            self.progress.emit(20, "Costruzione dell'architettura del modello...")
            self.msleep(2000)
            
            # 5. Addestramento
            for epoch in range(1, total_epochs + 1):
                progress = 20 + int(70 * epoch / total_epochs)
                
                # Simula metriche di training
                train_loss = 0.5 * (1 - epoch / total_epochs) + 0.1 * np.random.random()
                val_loss = 0.6 * (1 - epoch / total_epochs) + 0.15 * np.random.random()
                accuracy = 0.5 + 0.4 * (epoch / total_epochs) + 0.05 * np.random.random()
                
                status = f"Epoca {epoch}/{total_epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.2f}"
                self.progress.emit(progress, status)
                
                self.msleep(100)  # Simula il tempo di addestramento
            
            # 6. Valutazione finale
            self.progress.emit(95, "Valutazione del modello...")
            self.msleep(1000)
            
            # 7. Salvataggio
            self.progress.emit(98, "Salvataggio del modello...")
            self.msleep(1000)
            
            # Crea alcuni risultati fittizi
            results = {
                'model_name': self.config.get('model_name', 'Unnamed_Model'),
                'model_type': self.config.get('model_type', 'LSTM'),
                'training_time': f"{total_epochs * 0.1:.1f} minuti",
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'precision': 0.78 + 0.1 * np.random.random(),
                'recall': 0.75 + 0.1 * np.random.random(),
                'f1_score': 0.76 + 0.1 * np.random.random(),
                'dataset_size': 10000 + np.random.randint(0, 5000),
                'training_completed': datetime.now().strftime('%d/%m/%Y %H:%M'),
                'epochs': total_epochs,
                'target_symbols': self.config.get('symbols', []),
                'timeframes': self.config.get('timeframes', []),
                'features': self.config.get('features', []),
                'hyperparameters': {
                    'learning_rate': self.config.get('learning_rate', 0.001),
                    'batch_size': self.config.get('batch_size', 32),
                    'sequence_length': self.config.get('sequence_length', 60),
                    'lstm_units': self.config.get('hidden_units', [64, 32]),
                    'dropout': self.config.get('dropout', 0.2)
                }
            }
            
            # Completa l'addestramento
            self.progress.emit(100, "Addestramento completato.")
            self.completed.emit(results)
            
        except Exception as e:
            app_logger.error(f"Errore durante l'addestramento: {e}")
            self.error.emit(f"Errore durante l'addestramento: {e}")


class ModelConfigTab(QWidget):
    """Tab per la configurazione dei parametri del modello."""
    
    def __init__(self, parent=None):
        """
        Inizializza il tab di configurazione.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Usiamo un widget scrollabile
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Widget contenitore
        container = QWidget()
        container_layout = QVBoxLayout(container)
        
        # Gruppo informazioni generali
        info_group = QGroupBox("Informazioni Modello")
        info_layout = QGridLayout()
        
        # Nome modello
        info_layout.addWidget(QLabel("Nome modello:"), 0, 0)
        self.model_name_edit = QLineEdit()
        self.model_name_edit.setText(f"model_{datetime.now().strftime('%Y%m%d_%H%M')}")
        info_layout.addWidget(self.model_name_edit, 0, 1)
        
        # Tipo di modello
        info_layout.addWidget(QLabel("Tipo di modello:"), 1, 0)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("LSTM")
        self.model_type_combo.addItem("GRU")
        self.model_type_combo.addItem("Transformer")
        self.model_type_combo.addItem("CNN-LSTM")
        self.model_type_combo.addItem("LSTM-Attention")
        info_layout.addWidget(self.model_type_combo, 1, 1)
        
        # Descrizione
        info_layout.addWidget(QLabel("Descrizione:"), 2, 0)
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(60)
        info_layout.addWidget(self.description_edit, 2, 1)
        
        info_group.setLayout(info_layout)
        container_layout.addWidget(info_group)
        
        # Gruppo dati
        data_group = QGroupBox("Dati di Addestramento")
        data_layout = QVBoxLayout()
        
        # Selettore simboli
        data_layout.addWidget(QLabel("Simboli:"))
        self.symbols_list = QListWidget()
        self.symbols_list.setMaximumHeight(100)
        
        # Aggiungi alcuni simboli di esempio
        for symbol in ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "BTCUSD", "ETHUSD"]:
            item = QListWidgetItem(symbol)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.symbols_list.addItem(item)
        
        # Seleziona almeno un simbolo di default
        self.symbols_list.item(0).setCheckState(Qt.CheckState.Checked)
        
        data_layout.addWidget(self.symbols_list)
        
        # Selettore timeframe
        data_layout.addWidget(QLabel("Timeframes:"))
        timeframes_layout = QHBoxLayout()
        
        self.tf_m5_check = QCheckBox("5m")
        self.tf_m15_check = QCheckBox("15m")
        self.tf_m30_check = QCheckBox("30m")
        self.tf_h1_check = QCheckBox("1h")
        self.tf_h4_check = QCheckBox("4h")
        self.tf_d1_check = QCheckBox("1d")
        
        # Imposta un timeframe di default
        self.tf_h1_check.setChecked(True)
        
        timeframes_layout.addWidget(self.tf_m5_check)
        timeframes_layout.addWidget(self.tf_m15_check)
        timeframes_layout.addWidget(self.tf_m30_check)
        timeframes_layout.addWidget(self.tf_h1_check)
        timeframes_layout.addWidget(self.tf_h4_check)
        timeframes_layout.addWidget(self.tf_d1_check)
        
        data_layout.addLayout(timeframes_layout)
        
        # Intervallo date
        self.date_selector = DateRangeSelector()
        data_layout.addWidget(self.date_selector)
        
        data_group.setLayout(data_layout)
        container_layout.addWidget(data_group)
        
        # Gruppo caratteristiche
        features_group = QGroupBox("Caratteristiche")
        features_layout = QVBoxLayout()
        
        # Tipi di caratteristiche
        features_layout.addWidget(QLabel("Indicatori tecnici:"))
        indicators_layout = QGridLayout()
        
        self.rsi_check = QCheckBox("RSI")
        self.rsi_check.setChecked(True)
        indicators_layout.addWidget(self.rsi_check, 0, 0)
        
        self.macd_check = QCheckBox("MACD")
        self.macd_check.setChecked(True)
        indicators_layout.addWidget(self.macd_check, 0, 1)
        
        self.bollinger_check = QCheckBox("Bollinger")
        indicators_layout.addWidget(self.bollinger_check, 0, 2)
        
        self.ema_check = QCheckBox("EMA")
        self.ema_check.setChecked(True)
        indicators_layout.addWidget(self.ema_check, 1, 0)
        
        self.atr_check = QCheckBox("ATR")
        indicators_layout.addWidget(self.atr_check, 1, 1)
        
        self.adx_check = QCheckBox("ADX")
        indicators_layout.addWidget(self.adx_check, 1, 2)
        
        features_layout.addLayout(indicators_layout)
        
        # Preprocessing
        features_layout.addWidget(QLabel("Opzioni preprocessing:"))
        preprocessing_layout = QVBoxLayout()
        
        self.normalize_check = QCheckBox("Normalizzazione dati")
        self.normalize_check.setChecked(True)
        preprocessing_layout.addWidget(self.normalize_check)
        
        self.augment_check = QCheckBox("Augmentation dati")
        preprocessing_layout.addWidget(self.augment_check)
        
        features_layout.addLayout(preprocessing_layout)
        
        features_group.setLayout(features_layout)
        container_layout.addWidget(features_group)
        
        # Gruppo iperparametri
        hyperparams_group = QGroupBox("Iperparametri")
        hyperparams_layout = QGridLayout()
        
        # Epoche
        hyperparams_layout.addWidget(QLabel("Epoche:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        hyperparams_layout.addWidget(self.epochs_spin, 0, 1)
        
        # Learning rate
        hyperparams_layout.addWidget(QLabel("Learning rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(6)
        hyperparams_layout.addWidget(self.lr_spin, 1, 1)
        
        # Batch size
        hyperparams_layout.addWidget(QLabel("Batch size:"), 2, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        hyperparams_layout.addWidget(self.batch_spin, 2, 1)
        
        # Sequence length
        hyperparams_layout.addWidget(QLabel("Lunghezza sequenza:"), 0, 2)
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(5, 500)
        self.seq_spin.setValue(60)
        hyperparams_layout.addWidget(self.seq_spin, 0, 3)
        
        # Hidden units
        hyperparams_layout.addWidget(QLabel("Unità nascoste:"), 1, 2)
        self.units_spin = QSpinBox()
        self.units_spin.setRange(8, 512)
        self.units_spin.setValue(64)
        hyperparams_layout.addWidget(self.units_spin, 1, 3)
        
        # Dropout
        hyperparams_layout.addWidget(QLabel("Dropout:"), 2, 2)
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0, 0.9)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setSingleStep(0.1)
        hyperparams_layout.addWidget(self.dropout_spin, 2, 3)
        
        hyperparams_group.setLayout(hyperparams_layout)
        container_layout.addWidget(hyperparams_group)
        
        # Gruppo opzioni avanzate
        advanced_group = QGroupBox("Opzioni Avanzate")
        advanced_layout = QGridLayout()
        
        # Ottimizzatore
        advanced_layout.addWidget(QLabel("Ottimizzatore:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("Adam")
        self.optimizer_combo.addItem("SGD")
        self.optimizer_combo.addItem("RMSprop")
        self.optimizer_combo.addItem("Adagrad")
        advanced_layout.addWidget(self.optimizer_combo, 0, 1)
        
        # Loss function
        advanced_layout.addWidget(QLabel("Funzione di loss:"), 1, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItem("Mean Squared Error")
        self.loss_combo.addItem("Binary Crossentropy")
        self.loss_combo.addItem("Categorical Crossentropy")
        self.loss_combo.addItem("Mean Absolute Error")
        advanced_layout.addWidget(self.loss_combo, 1, 1)
        
        # Validation split
        advanced_layout.addWidget(QLabel("Validation split:"), 0, 2)
        self.validation_spin = QDoubleSpinBox()
        self.validation_spin.setRange(0.1, 0.5)
        self.validation_spin.setValue(0.2)
        self.validation_spin.setSingleStep(0.05)
        advanced_layout.addWidget(self.validation_spin, 0, 3)
        
        # Early stopping
        advanced_layout.addWidget(QLabel("Early stopping:"), 1, 2)
        self.early_stopping_check = QCheckBox()
        self.early_stopping_check.setChecked(True)
        advanced_layout.addWidget(self.early_stopping_check, 1, 3)
        
        advanced_group.setLayout(advanced_layout)
        container_layout.addWidget(advanced_group)
        
        # Aggiungi uno stretcher alla fine
        container_layout.addStretch()
        
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Ottiene la configurazione del modello.
        
        Returns:
            Dizionario con i parametri di configurazione
        """
        # Raccogli i simboli selezionati
        selected_symbols = []
        for i in range(self.symbols_list.count()):
            item = self.symbols_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_symbols.append(item.text())
        
        # Raccogli i timeframe selezionati
        selected_timeframes = []
        if self.tf_m5_check.isChecked():
            selected_timeframes.append("5m")
        if self.tf_m15_check.isChecked():
            selected_timeframes.append("15m")
        if self.tf_m30_check.isChecked():
            selected_timeframes.append("30m")
        if self.tf_h1_check.isChecked():
            selected_timeframes.append("1h")
        if self.tf_h4_check.isChecked():
            selected_timeframes.append("4h")
        if self.tf_d1_check.isChecked():
            selected_timeframes.append("1d")
        
        # Raccogli le caratteristiche selezionate
        selected_features = []
        if self.rsi_check.isChecked():
            selected_features.append("RSI")
        if self.macd_check.isChecked():
            selected_features.append("MACD")
        if self.bollinger_check.isChecked():
            selected_features.append("Bollinger")
        if self.ema_check.isChecked():
            selected_features.append("EMA")
        if self.atr_check.isChecked():
            selected_features.append("ATR")
        if self.adx_check.isChecked():
            selected_features.append("ADX")
        
        # Crea la configurazione
        config = {
            'model_name': self.model_name_edit.text(),
            'model_type': self.model_type_combo.currentText(),
            'description': self.description_edit.toPlainText(),
            'symbols': selected_symbols,
            'timeframes': selected_timeframes,
            'date_range': self.date_selector.get_date_range(),
            'features': selected_features,
            'normalize': self.normalize_check.isChecked(),
            'augment': self.augment_check.isChecked(),
            'epochs': self.epochs_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'batch_size': self.batch_spin.value(),
            'sequence_length': self.seq_spin.value(),
            'hidden_units': self.units_spin.value(),
            'dropout': self.dropout_spin.value(),
            'optimizer': self.optimizer_combo.currentText(),
            'loss': self.loss_combo.currentText(),
            'validation_split': self.validation_spin.value(),
            'early_stopping': self.early_stopping_check.isChecked()
        }
        
        return config


class ModelResultTab(QWidget):
    """Tab per visualizzare i risultati dell'addestramento."""
    
    def __init__(self, parent=None):
        """
        Inizializza il tab dei risultati.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.results = None
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Usiamo un widget scrollabile
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Widget contenitore
        container = QWidget()
        container_layout = QVBoxLayout(container)
        
        # Informazioni generali
        info_group = QGroupBox("Informazioni Modello")
        info_layout = QGridLayout()
        
        info_layout.addWidget(QLabel("Nome:"), 0, 0)
        self.model_name_label = QLabel("")
        info_layout.addWidget(self.model_name_label, 0, 1)
        
        info_layout.addWidget(QLabel("Tipo:"), 1, 0)
        self.model_type_label = QLabel("")
        info_layout.addWidget(self.model_type_label, 1, 1)
        
        info_layout.addWidget(QLabel("Completato:"), 2, 0)
        self.completed_label = QLabel("")
        info_layout.addWidget(self.completed_label, 2, 1)
        
        info_layout.addWidget(QLabel("Tempo:"), 0, 2)
        self.time_label = QLabel("")
        info_layout.addWidget(self.time_label, 0, 3)
        
        info_layout.addWidget(QLabel("Dataset:"), 1, 2)
        self.dataset_label = QLabel("")
        info_layout.addWidget(self.dataset_label, 1, 3)
        
        info_layout.addWidget(QLabel("Epoche:"), 2, 2)
        self.epochs_label = QLabel("")
        info_layout.addWidget(self.epochs_label, 2, 3)
        
        info_group.setLayout(info_layout)
        container_layout.addWidget(info_group)
        
        # Metriche
        metrics_group = QGroupBox("Metriche")
        metrics_layout = QGridLayout()
        
        metrics_layout.addWidget(QLabel("Loss (Train):"), 0, 0)
        self.train_loss_label = QLabel("")
        metrics_layout.addWidget(self.train_loss_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Loss (Val):"), 1, 0)
        self.val_loss_label = QLabel("")
        metrics_layout.addWidget(self.val_loss_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Accuracy:"), 0, 2)
        self.accuracy_label = QLabel("")
        metrics_layout.addWidget(self.accuracy_label, 0, 3)
        
        metrics_layout.addWidget(QLabel("Precision:"), 1, 2)
        self.precision_label = QLabel("")
        metrics_layout.addWidget(self.precision_label, 1, 3)
        
        metrics_layout.addWidget(QLabel("Recall:"), 2, 0)
        self.recall_label = QLabel("")
        metrics_layout.addWidget(self.recall_label, 2, 1)
        
        metrics_layout.addWidget(QLabel("F1 Score:"), 2, 2)
        self.f1_label = QLabel("")
        metrics_layout.addWidget(self.f1_label, 2, 3)
        
        metrics_group.setLayout(metrics_layout)
        container_layout.addWidget(metrics_group)
        
        # Caratteristiche
        features_group = QGroupBox("Caratteristiche e Addestramento")
        features_layout = QGridLayout()
        
        features_layout.addWidget(QLabel("Simboli:"), 0, 0)
        self.symbols_label = QLabel("")
        features_layout.addWidget(self.symbols_label, 0, 1)
        
        features_layout.addWidget(QLabel("Timeframes:"), 1, 0)
        self.timeframes_label = QLabel("")
        features_layout.addWidget(self.timeframes_label, 1, 1)
        
        features_layout.addWidget(QLabel("Features:"), 2, 0)
        self.features_label = QLabel("")
        features_layout.addWidget(self.features_label, 2, 1)
        
        features_group.setLayout(features_layout)
        container_layout.addWidget(features_group)
        
        # Iperparametri
        hyperparams_group = QGroupBox("Iperparametri")
        hyperparams_layout = QGridLayout()
        
        hyperparams_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.lr_label = QLabel("")
        hyperparams_layout.addWidget(self.lr_label, 0, 1)
        
        hyperparams_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_label = QLabel("")
        hyperparams_layout.addWidget(self.batch_label, 1, 1)
        
        hyperparams_layout.addWidget(QLabel("Sequence Length:"), 0, 2)
        self.seq_label = QLabel("")
        hyperparams_layout.addWidget(self.seq_label, 0, 3)
        
        hyperparams_layout.addWidget(QLabel("Hidden Units:"), 1, 2)
        self.units_label = QLabel("")
        hyperparams_layout.addWidget(self.units_label, 1, 3)
        
        hyperparams_layout.addWidget(QLabel("Dropout:"), 2, 0)
        self.dropout_label = QLabel("")
        hyperparams_layout.addWidget(self.dropout_label, 2, 1)
        
        hyperparams_group.setLayout(hyperparams_layout)
        container_layout.addWidget(hyperparams_group)
        
        # Aggiungi uno stretcher alla fine
        container_layout.addStretch()
        
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)
    
    def update_results(self, results: Dict[str, Any]):
        """
        Aggiorna i risultati visualizzati.
        
        Args:
            results: Dizionario con i risultati dell'addestramento
        """
        self.results = results
        
        if not results:
            return
        
        # Aggiorna le etichette
        self.model_name_label.setText(results.get('model_name', '-'))
        self.model_type_label.setText(results.get('model_type', '-'))
        self.completed_label.setText(results.get('training_completed', '-'))
        self.time_label.setText(results.get('training_time', '-'))
        self.dataset_label.setText(f"{results.get('dataset_size', 0):,} campioni")
        self.epochs_label.setText(str(results.get('epochs', 0)))
        
        # Metriche
        self.train_loss_label.setText(f"{results.get('train_loss', 0):.4f}")
        self.val_loss_label.setText(f"{results.get('val_loss', 0):.4f}")
        self.accuracy_label.setText(f"{results.get('accuracy', 0):.2%}")
        self.precision_label.setText(f"{results.get('precision', 0):.2%}")
        self.recall_label.setText(f"{results.get('recall', 0):.2%}")
        self.f1_label.setText(f"{results.get('f1_score', 0):.2%}")
        
        # Caratteristiche
        self.symbols_label.setText(", ".join(results.get('target_symbols', [])))
        self.timeframes_label.setText(", ".join(results.get('timeframes', [])))
        self.features_label.setText(", ".join(results.get('features', [])))
        
        # Iperparametri
        hyperparams = results.get('hyperparameters', {})
        self.lr_label.setText(f"{hyperparams.get('learning_rate', 0):.6f}")
        self.batch_label.setText(str(hyperparams.get('batch_size', 0)))
        self.seq_label.setText(str(hyperparams.get('sequence_length', 0)))
        
        # Gestisci le unità nascoste, che potrebbero essere un array
        units = hyperparams.get('lstm_units', [])
        if isinstance(units, list):
            self.units_label.setText(", ".join(map(str, units)))
        else:
            self.units_label.setText(str(units))
        
        self.dropout_label.setText(f"{hyperparams.get('dropout', 0):.1f}")


class ModelTrainingDialog(QDialog):
    """Finestra di dialogo per configurare ed eseguire l'addestramento di modelli."""
    
    def __init__(self, parent=None):
        """
        Inizializza la finestra di dialogo.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.training_thread = None
        self.results = None
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia della finestra di dialogo."""
        self.setWindowTitle("Addestramento Modello")
        self.setMinimumSize(800, 600)
        
        # Layout principale
        layout = QVBoxLayout(self)
        
        # Tab widget per le diverse sezioni
        self.tab_widget = QTabWidget()
        
        # Tab di configurazione
        self.config_tab = ModelConfigTab()
        self.tab_widget.addTab(self.config_tab, "Configurazione")
        
        # Tab dei risultati
        self.result_tab = ModelResultTab()
        self.tab_widget.addTab(self.result_tab, "Risultati")
        
        # Aggiungi il tab widget al layout
        layout.addWidget(self.tab_widget)
        
        # Log di addestramento
        log_group = QGroupBox("Log di Addestramento")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        layout.addWidget(log_group)
        
        # Barra di progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Pulsanti
        button_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Addestra Modello")
        self.train_button.clicked.connect(self.train_model)
        
        self.save_button = QPushButton("Salva Modello")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        
        self.close_button = QPushButton("Chiudi")
        self.close_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.save_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def add_log_message(self, message: str):
        """
        Aggiunge un messaggio al log di addestramento.
        
        Args:
            message: Messaggio da aggiungere
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.log_text.append(log_message)
        
        # Scorri alla fine del log
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
    def train_model(self):
        """Avvia l'addestramento del modello con i parametri configurati."""
        # Ottieni la configurazione
        config = self.config_tab.get_config()
        
        # Verifica che ci siano dati sufficienti
        if not config['symbols'] or not config['timeframes'] or not config['features']:
            QMessageBox.warning(
                self,
                "Configurazione Incompleta",
                "Seleziona almeno un simbolo, un timeframe e una feature."
            )
            return
        
        # Pulisci i risultati precedenti
        self.results = None
        
        # Pulisci il log
        self.log_text.clear()
        
        # Aggiungi messaggio iniziale
        self.add_log_message(f"Inizio addestramento del modello {config['model_name']}...")
        self.add_log_message(f"Tipo: {config['model_type']}, Epoche: {config['epochs']}, Batch: {config['batch_size']}")
        self.add_log_message(f"Simboli: {', '.join(config['symbols'])}, Timeframes: {', '.join(config['timeframes'])}")
        
        # Imposta la barra di progresso
        self.progress_bar.setValue(0)
        
        # Disabilita il pulsante di addestramento
        self.train_button.setEnabled(False)
        
        # Crea e avvia il thread di addestramento
        self.training_thread = TrainingThread(config)
        
        # Connetti i segnali
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.completed.connect(self.training_completed)
        self.training_thread.error.connect(self.training_error)
        
        # Avvia il thread
        self.training_thread.start()
    
    def update_progress(self, value: int, status: str):
        """
        Aggiorna la barra di progresso e il log.
        
        Args:
            value: Percentuale di completamento
            status: Messaggio di stato
        """
        self.progress_bar.setValue(value)
        self.add_log_message(status)
    
    def training_completed(self, results: Dict[str, Any]):
        """
        Gestisce il completamento dell'addestramento.
        
        Args:
            results: Dizionario con i risultati dell'addestramento
        """
        self.results = results
        
        # Aggiorna il tab dei risultati
        self.result_tab.update_results(results)
        
        # Aggiungi messaggio di completamento
        self.add_log_message("Addestramento completato con successo!")
        self.add_log_message(f"Accuracy: {results['accuracy']:.2%}, Loss: {results['train_loss']:.4f}")
        
        # Passa al tab dei risultati
        self.tab_widget.setCurrentIndex(1)
        
        # Riabilita il pulsante di addestramento
        self.train_button.setEnabled(True)
        
        # Abilita il pulsante di salvataggio
        self.save_button.setEnabled(True)
        
        # Mostra un messaggio
        QMessageBox.information(
            self,
            "Addestramento Completato",
            f"Addestramento del modello \"{results['model_name']}\" completato con successo.\n\n"
            f"Accuracy: {results['accuracy']:.2%}\n"
            f"Loss: {results['train_loss']:.4f}"
        )
    
    def training_error(self, error_msg: str):
        """
        Gestisce gli errori durante l'addestramento.
        
        Args:
            error_msg: Messaggio di errore
        """
        # Aggiungi messaggio di errore
        self.add_log_message(f"ERRORE: {error_msg}")
        
        # Riabilita il pulsante di addestramento
        self.train_button.setEnabled(True)
        
        # Mostra un messaggio di errore
        QMessageBox.critical(
            self,
            "Errore Addestramento",
            f"Si è verificato un errore durante l'addestramento:\n{error_msg}"
        )
    
    def save_model(self):
        """Salva il modello addestrato."""
        if not self.results:
            return
        
        # Mostra una finestra di dialogo per selezionare il file
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Salva Modello",
            f"{self.results['model_name']}.json",
            "JSON (*.json);;Tutti i file (*)"
        )
        
        if not filename:
            return
        
        try:
            # In un'implementazione reale, qui salveremmo il modello vero e proprio
            # Per ora, salviamo solo i metadati in formato JSON
            
            # Prepara i metadati
            metadata = {
                'model_name': self.results['model_name'],
                'model_type': self.results['model_type'],
                'description': self.config_tab.description_edit.toPlainText(),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': {
                    'accuracy': self.results['accuracy'],
                    'precision': self.results['precision'],
                    'recall': self.results['recall'],
                    'f1_score': self.results['f1_score'],
                    'train_loss': self.results['train_loss'],
                    'val_loss': self.results['val_loss']
                },
                'hyperparameters': self.results['hyperparameters'],
                'features': self.results['features'],
                'target_symbols': self.results['target_symbols'],
                'timeframes': self.results['timeframes']
            }
            
            # Salva in formato JSON
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4)
            
            self.add_log_message(f"Modello salvato con successo: {filename}")
            
            QMessageBox.information(
                self,
                "Salvataggio Completato",
                f"Il modello è stato salvato in:\n{filename}"
            )
            
        except Exception as e:
            app_logger.error(f"Errore durante il salvataggio del modello: {e}")
            
            self.add_log_message(f"Errore durante il salvataggio: {e}")
            
            QMessageBox.critical(
                self,
                "Errore Salvataggio",
                f"Si è verificato un errore durante il salvataggio del modello:\n{e}"
            )