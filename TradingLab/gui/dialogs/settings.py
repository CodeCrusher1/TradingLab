# Impostazioni

# Impostazioni

"""
Finestra di dialogo per la configurazione delle impostazioni dell'applicazione.
Permette di modificare temi, percorsi dati, broker e altre preferenze.
"""
from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout,
    QFrame, QTabWidget, QSizePolicy, QScrollArea, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox, QFileDialog, QLineEdit,
    QListWidget, QListWidgetItem, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QFont, QIcon

import os
from typing import Dict, Any

from ..styles import style_manager, Theme
from ...utils import app_logger
from ...config import APP_NAME, DATA_DIR


class GeneralSettingsTab(QWidget):
    """Tab per le impostazioni generali dell'applicazione."""
    
    def __init__(self, parent=None):
        """
        Inizializza il tab delle impostazioni generali.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.settings = QSettings(APP_NAME, "settings")
        self.initUI()
        self.load_settings()
    
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
        
        # Tema
        theme_group = QGroupBox("Tema")
        theme_layout = QVBoxLayout()
        
        # Radio button per la selezione del tema
        self.theme_light_radio = QRadioButton("Chiaro")
        self.theme_dark_radio = QRadioButton("Scuro")
        
        # Raggruppa i radio button
        theme_button_group = QButtonGroup(self)
        theme_button_group.addButton(self.theme_light_radio)
        theme_button_group.addButton(self.theme_dark_radio)
        
        theme_layout.addWidget(self.theme_light_radio)
        theme_layout.addWidget(self.theme_dark_radio)
        
        theme_group.setLayout(theme_layout)
        container_layout.addWidget(theme_group)
        
        # Percorsi dati
        paths_group = QGroupBox("Percorsi Dati")
        paths_layout = QGridLayout()
        
        # Cartella dati
        paths_layout.addWidget(QLabel("Cartella dati:"), 0, 0)
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setReadOnly(True)
        paths_layout.addWidget(self.data_path_edit, 0, 1)
        
        self.data_path_button = QPushButton("Sfoglia...")
        self.data_path_button.clicked.connect(self.browse_data_path)
        paths_layout.addWidget(self.data_path_button, 0, 2)
        
        # Cartella modelli
        paths_layout.addWidget(QLabel("Cartella modelli:"), 1, 0)
        self.models_path_edit = QLineEdit()
        self.models_path_edit.setReadOnly(True)
        paths_layout.addWidget(self.models_path_edit, 1, 1)
        
        self.models_path_button = QPushButton("Sfoglia...")
        self.models_path_button.clicked.connect(self.browse_models_path)
        paths_layout.addWidget(self.models_path_button, 1, 2)
        
        # Cartella backtest
        paths_layout.addWidget(QLabel("Cartella backtest:"), 2, 0)
        self.backtest_path_edit = QLineEdit()
        self.backtest_path_edit.setReadOnly(True)
        paths_layout.addWidget(self.backtest_path_edit, 2, 1)
        
        self.backtest_path_button = QPushButton("Sfoglia...")
        self.backtest_path_button.clicked.connect(self.browse_backtest_path)
        paths_layout.addWidget(self.backtest_path_button, 2, 2)
        
        paths_group.setLayout(paths_layout)
        container_layout.addWidget(paths_group)
        
        # Preferenze utente
        prefs_group = QGroupBox("Preferenze")
        prefs_layout = QGridLayout()
        
        # Aggiornamenti automatici
        prefs_layout.addWidget(QLabel("Aggiornamenti automatici:"), 0, 0)
        self.auto_update_check = QCheckBox()
        self.auto_update_check.setChecked(True)
        prefs_layout.addWidget(self.auto_update_check, 0, 1)
        
        # Frequenza aggiornamenti
        prefs_layout.addWidget(QLabel("Frequenza aggiornamenti (min):"), 1, 0)
        self.update_interval_spin = QSpinBox()
        self.update_interval_spin.setRange(1, 60)
        self.update_interval_spin.setValue(5)
        prefs_layout.addWidget(self.update_interval_spin, 1, 1)
        
        # Mostra notifiche
        prefs_layout.addWidget(QLabel("Mostra notifiche:"), 2, 0)
        self.show_notifications_check = QCheckBox()
        self.show_notifications_check.setChecked(True)
        prefs_layout.addWidget(self.show_notifications_check, 2, 1)
        
        # Salva workspace all'uscita
        prefs_layout.addWidget(QLabel("Salva workspace all'uscita:"), 3, 0)
        self.save_workspace_check = QCheckBox()
        self.save_workspace_check.setChecked(True)
        prefs_layout.addWidget(self.save_workspace_check, 3, 1)
        
        prefs_group.setLayout(prefs_layout)
        container_layout.addWidget(prefs_group)
        
        # Lingua
        language_group = QGroupBox("Lingua")
        language_layout = QVBoxLayout()
        
        self.language_combo = QComboBox()
        self.language_combo.addItem("Italiano")
        self.language_combo.addItem("English")
        self.language_combo.addItem("Español")
        self.language_combo.addItem("Français")
        self.language_combo.addItem("Deutsch")
        
        language_layout.addWidget(self.language_combo)
        
        language_group.setLayout(language_layout)
        container_layout.addWidget(language_group)
        
        # Aggiungi uno stretcher alla fine
        container_layout.addStretch()
        
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)
    
    def load_settings(self):
        """Carica le impostazioni salvate."""
        # Tema
        theme = self.settings.value("theme", "light")
        if theme == "dark":
            self.theme_dark_radio.setChecked(True)
        else:
            self.theme_light_radio.setChecked(True)
        
        # Percorsi
        data_path = self.settings.value("data_path", DATA_DIR)
        self.data_path_edit.setText(data_path)
        
        models_path = self.settings.value("models_path", os.path.join(DATA_DIR, "models"))
        self.models_path_edit.setText(models_path)
        
        backtest_path = self.settings.value("backtest_path", os.path.join(DATA_DIR, "backtest"))
        self.backtest_path_edit.setText(backtest_path)
        
        # Preferenze
        auto_update = self.settings.value("auto_update", True, type=bool)
        self.auto_update_check.setChecked(auto_update)
        
        update_interval = self.settings.value("update_interval", 5, type=int)
        self.update_interval_spin.setValue(update_interval)
        
        show_notifications = self.settings.value("show_notifications", True, type=bool)
        self.show_notifications_check.setChecked(show_notifications)
        
        save_workspace = self.settings.value("save_workspace", True, type=bool)
        self.save_workspace_check.setChecked(save_workspace)
        
        # Lingua
        language = self.settings.value("language", "Italiano")
        index = self.language_combo.findText(language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
    
    def save_settings(self):
        """Salva le impostazioni correnti."""
        # Tema
        theme = "dark" if self.theme_dark_radio.isChecked() else "light"
        self.settings.setValue("theme", theme)
        
        # Percorsi
        self.settings.setValue("data_path", self.data_path_edit.text())
        self.settings.setValue("models_path", self.models_path_edit.text())
        self.settings.setValue("backtest_path", self.backtest_path_edit.text())
        
        # Preferenze
        self.settings.setValue("auto_update", self.auto_update_check.isChecked())
        self.settings.setValue("update_interval", self.update_interval_spin.value())
        self.settings.setValue("show_notifications", self.show_notifications_check.isChecked())
        self.settings.setValue("save_workspace", self.save_workspace_check.isChecked())
        
        # Lingua
        self.settings.setValue("language", self.language_combo.currentText())
    
    def browse_data_path(self):
        """Apre una finestra di dialogo per selezionare la cartella dati."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Seleziona Cartella Dati",
            self.data_path_edit.text()
        )
        
        if directory:
            self.data_path_edit.setText(directory)
    
    def browse_models_path(self):
        """Apre una finestra di dialogo per selezionare la cartella modelli."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Seleziona Cartella Modelli",
            self.models_path_edit.text()
        )
        
        if directory:
            self.models_path_edit.setText(directory)
    
    def browse_backtest_path(self):
        """Apre una finestra di dialogo per selezionare la cartella backtest."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Seleziona Cartella Backtest",
            self.backtest_path_edit.text()
        )
        
        if directory:
            self.backtest_path_edit.setText(directory)


class BrokersTab(QWidget):
    """Tab per la configurazione dei broker."""
    
    def __init__(self, parent=None):
        """
        Inizializza il tab dei broker.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.settings = QSettings(APP_NAME, "settings")
        self.initUI()
        self.load_settings()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Broker attivo
        active_group = QGroupBox("Broker Attivo")
        active_layout = QVBoxLayout()
        
        self.broker_combo = QComboBox()
        self.broker_combo.addItem("Demo")
        self.broker_combo.addItem("MetaTrader 5")
        self.broker_combo.addItem("Interactive Brokers")
        self.broker_combo.addItem("cTrader")
        self.broker_combo.addItem("OANDA")
        
        self.broker_combo.currentTextChanged.connect(self.on_broker_changed)
        
        active_layout.addWidget(self.broker_combo)
        
        active_group.setLayout(active_layout)
        layout.addWidget(active_group)
        
        # Impostazioni specifiche del broker
        self.broker_settings_group = QGroupBox("Impostazioni Broker")
        self.broker_settings_layout = QGridLayout()
        
        # I campi verranno aggiornati in base al broker selezionato
        
        self.broker_settings_group.setLayout(self.broker_settings_layout)
        layout.addWidget(self.broker_settings_group)
        
        # Account trading
        account_group = QGroupBox("Account Trading")
        account_layout = QGridLayout()
        
        account_layout.addWidget(QLabel("Nome account:"), 0, 0)
        self.account_name_edit = QLineEdit()
        account_layout.addWidget(self.account_name_edit, 0, 1)
        
        account_layout.addWidget(QLabel("Valuta base:"), 1, 0)
        self.base_currency_combo = QComboBox()
        self.base_currency_combo.addItem("EUR")
        self.base_currency_combo.addItem("USD")
        self.base_currency_combo.addItem("GBP")
        self.base_currency_combo.addItem("JPY")
        self.base_currency_combo.addItem("CHF")
        account_layout.addWidget(self.base_currency_combo, 1, 1)
        
        account_layout.addWidget(QLabel("Margine (Leva):"), 2, 0)
        self.leverage_spin = QDoubleSpinBox()
        self.leverage_spin.setRange(1, 500)
        self.leverage_spin.setValue(100)
        self.leverage_spin.setSuffix("x")
        account_layout.addWidget(self.leverage_spin, 2, 1)
        
        account_group.setLayout(account_layout)
        layout.addWidget(account_group)
        
        # Pulsanti di azione
        buttons_layout = QHBoxLayout()
        
        self.test_connection_button = QPushButton("Testa Connessione")
        self.test_connection_button.clicked.connect(self.test_connection)
        buttons_layout.addWidget(self.test_connection_button)
        
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
        # Aggiungi uno stretcher alla fine
        layout.addStretch()
    
    def load_settings(self):
        """Carica le impostazioni salvate."""
        # Broker attivo
        broker = self.settings.value("active_broker", "Demo")
        index = self.broker_combo.findText(broker)
        if index >= 0:
            self.broker_combo.setCurrentIndex(index)
        
        # Account
        self.account_name_edit.setText(self.settings.value("account_name", "Demo Account"))
        
        base_currency = self.settings.value("base_currency", "EUR")
        index = self.base_currency_combo.findText(base_currency)
        if index >= 0:
            self.base_currency_combo.setCurrentIndex(index)
        
        leverage = self.settings.value("leverage", 100, type=float)
        self.leverage_spin.setValue(leverage)
        
        # Aggiorna le impostazioni specifiche del broker
        self.update_broker_settings()
    
    def save_settings(self):
        """Salva le impostazioni correnti."""
        # Broker attivo
        self.settings.setValue("active_broker", self.broker_combo.currentText())
        
        # Account
        self.settings.setValue("account_name", self.account_name_edit.text())
        self.settings.setValue("base_currency", self.base_currency_combo.currentText())
        self.settings.setValue("leverage", self.leverage_spin.value())
        
        # Salva le impostazioni specifiche del broker
        self.save_broker_settings()
    
    def on_broker_changed(self, broker: str):
        """
        Gestisce il cambio di broker selezionato.
        
        Args:
            broker: Nome del broker selezionato
        """
        self.update_broker_settings()
    
    def update_broker_settings(self):
        """Aggiorna i widget con le impostazioni specifiche del broker selezionato."""
        # Pulisci i widget esistenti
        while self.broker_settings_layout.count():
            item = self.broker_settings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        broker = self.broker_combo.currentText()
        
        if broker == "Demo":
            # Non sono necessarie impostazioni speciali per il broker demo
            self.broker_settings_layout.addWidget(QLabel("Non sono necessarie configurazioni per il broker demo."), 0, 0, 1, 2)
            return
        
        elif broker == "MetaTrader 5":
            # Impostazioni per MetaTrader 5
            self.broker_settings_layout.addWidget(QLabel("Percorso MT5:"), 0, 0)
            self.mt5_path_edit = QLineEdit()
            self.mt5_path_edit.setText(self.settings.value("mt5_path", ""))
            self.broker_settings_layout.addWidget(self.mt5_path_edit, 0, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Login:"), 1, 0)
            self.mt5_login_edit = QLineEdit()
            self.mt5_login_edit.setText(self.settings.value("mt5_login", ""))
            self.broker_settings_layout.addWidget(self.mt5_login_edit, 1, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Password:"), 2, 0)
            self.mt5_password_edit = QLineEdit()
            self.mt5_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.mt5_password_edit.setText(self.settings.value("mt5_password", ""))
            self.broker_settings_layout.addWidget(self.mt5_password_edit, 2, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Server:"), 3, 0)
            self.mt5_server_edit = QLineEdit()
            self.mt5_server_edit.setText(self.settings.value("mt5_server", ""))
            self.broker_settings_layout.addWidget(self.mt5_server_edit, 3, 1)
            
        elif broker == "Interactive Brokers":
            # Impostazioni per Interactive Brokers
            self.broker_settings_layout.addWidget(QLabel("Porta TWS:"), 0, 0)
            self.ib_port_spin = QSpinBox()
            self.ib_port_spin.setRange(1000, 9999)
            self.ib_port_spin.setValue(self.settings.value("ib_port", 7496, type=int))
            self.broker_settings_layout.addWidget(self.ib_port_spin, 0, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Client ID:"), 1, 0)
            self.ib_client_id_spin = QSpinBox()
            self.ib_client_id_spin.setRange(1, 9999)
            self.ib_client_id_spin.setValue(self.settings.value("ib_client_id", 1, type=int))
            self.broker_settings_layout.addWidget(self.ib_client_id_spin, 1, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Host:"), 2, 0)
            self.ib_host_edit = QLineEdit()
            self.ib_host_edit.setText(self.settings.value("ib_host", "127.0.0.1"))
            self.broker_settings_layout.addWidget(self.ib_host_edit, 2, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Timeout (s):"), 3, 0)
            self.ib_timeout_spin = QSpinBox()
            self.ib_timeout_spin.setRange(1, 60)
            self.ib_timeout_spin.setValue(self.settings.value("ib_timeout", 20, type=int))
            self.broker_settings_layout.addWidget(self.ib_timeout_spin, 3, 1)
            
        elif broker == "cTrader":
            # Impostazioni per cTrader
            self.broker_settings_layout.addWidget(QLabel("API Token:"), 0, 0)
            self.ctrader_token_edit = QLineEdit()
            self.ctrader_token_edit.setText(self.settings.value("ctrader_token", ""))
            self.broker_settings_layout.addWidget(self.ctrader_token_edit, 0, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Account ID:"), 1, 0)
            self.ctrader_account_edit = QLineEdit()
            self.ctrader_account_edit.setText(self.settings.value("ctrader_account", ""))
            self.broker_settings_layout.addWidget(self.ctrader_account_edit, 1, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Ambiente:"), 2, 0)
            self.ctrader_env_combo = QComboBox()
            self.ctrader_env_combo.addItem("Demo")
            self.ctrader_env_combo.addItem("Live")
            index = self.ctrader_env_combo.findText(self.settings.value("ctrader_env", "Demo"))
            if index >= 0:
                self.ctrader_env_combo.setCurrentIndex(index)
            self.broker_settings_layout.addWidget(self.ctrader_env_combo, 2, 1)
            
        elif broker == "OANDA":
            # Impostazioni per OANDA
            self.broker_settings_layout.addWidget(QLabel("API Key:"), 0, 0)
            self.oanda_api_key_edit = QLineEdit()
            self.oanda_api_key_edit.setText(self.settings.value("oanda_api_key", ""))
            self.broker_settings_layout.addWidget(self.oanda_api_key_edit, 0, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Account ID:"), 1, 0)
            self.oanda_account_edit = QLineEdit()
            self.oanda_account_edit.setText(self.settings.value("oanda_account", ""))
            self.broker_settings_layout.addWidget(self.oanda_account_edit, 1, 1)
            
            self.broker_settings_layout.addWidget(QLabel("Ambiente:"), 2, 0)
            self.oanda_env_combo = QComboBox()
            self.oanda_env_combo.addItem("Practice")
            self.oanda_env_combo.addItem("Live")
            index = self.oanda_env_combo.findText(self.settings.value("oanda_env", "Practice"))
            if index >= 0:
                self.oanda_env_combo.setCurrentIndex(index)
            self.broker_settings_layout.addWidget(self.oanda_env_combo, 2, 1)
    
    def save_broker_settings(self):
        """Salva le impostazioni specifiche del broker selezionato."""
        broker = self.broker_combo.currentText()
        
        if broker == "MetaTrader 5":
            self.settings.setValue("mt5_path", self.mt5_path_edit.text())
            self.settings.setValue("mt5_login", self.mt5_login_edit.text())
            self.settings.setValue("mt5_password", self.mt5_password_edit.text())
            self.settings.setValue("mt5_server", self.mt5_server_edit.text())
            
        elif broker == "Interactive Brokers":
            self.settings.setValue("ib_port", self.ib_port_spin.value())
            self.settings.setValue("ib_client_id", self.ib_client_id_spin.value())
            self.settings.setValue("ib_host", self.ib_host_edit.text())
            self.settings.setValue("ib_timeout", self.ib_timeout_spin.value())
            
        elif broker == "cTrader":
            self.settings.setValue("ctrader_token", self.ctrader_token_edit.text())
            self.settings.setValue("ctrader_account", self.ctrader_account_edit.text())
            self.settings.setValue("ctrader_env", self.ctrader_env_combo.currentText())
            
        elif broker == "OANDA":
            self.settings.setValue("oanda_api_key", self.oanda_api_key_edit.text())
            self.settings.setValue("oanda_account", self.oanda_account_edit.text())
            self.settings.setValue("oanda_env", self.oanda_env_combo.currentText())
    
    def test_connection(self):
        """Testa la connessione al broker selezionato."""
        broker = self.broker_combo.currentText()
        
        # In un'implementazione reale, qui verificheremmo la connessione
        # al broker. Per ora, mostriamo un messaggio fittizio.
        
        if broker == "Demo":
            QMessageBox.information(
                self,
                "Test Connessione",
                "Connessione al broker demo riuscita."
            )
            return
        
        # Simula una connessione riuscita
        QMessageBox.information(
            self,
            "Test Connessione",
            f"Connessione a {broker} riuscita."
        )


class SystemTab(QWidget):
    """Tab per le impostazioni di sistema."""
    
    def __init__(self, parent=None):
        """
        Inizializza il tab delle impostazioni di sistema.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.settings = QSettings(APP_NAME, "settings")
        self.initUI()
        self.load_settings()
    
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
        
        # Performance
        performance_group = QGroupBox("Performance")
        performance_layout = QGridLayout()
        
        # Numero di thread
        performance_layout.addWidget(QLabel("Numero thread:"), 0, 0)
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 16)
        self.threads_spin.setValue(4)
        performance_layout.addWidget(self.threads_spin, 0, 1)
        
        # Cache dati
        performance_layout.addWidget(QLabel("Dimensione cache (MB):"), 1, 0)
        self.cache_spin = QSpinBox()
        self.cache_spin.setRange(100, 10000)
        self.cache_spin.setValue(1000)
        self.cache_spin.setSingleStep(100)
        performance_layout.addWidget(self.cache_spin, 1, 1)
        
        # Modalità batch
        performance_layout.addWidget(QLabel("Modalità batch:"), 2, 0)
        self.batch_mode_check = QCheckBox()
        self.batch_mode_check.setChecked(False)
        performance_layout.addWidget(self.batch_mode_check, 2, 1)
        
        performance_group.setLayout(performance_layout)
        container_layout.addWidget(performance_group)
        
        # Logging
        logging_group = QGroupBox("Logging")
        logging_layout = QGridLayout()
        
        # Livello di log
        logging_layout.addWidget(QLabel("Livello di log:"), 0, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItem("DEBUG")
        self.log_level_combo.addItem("INFO")
        self.log_level_combo.addItem("WARNING")
        self.log_level_combo.addItem("ERROR")
        self.log_level_combo.setCurrentText("INFO")
        logging_layout.addWidget(self.log_level_combo, 0, 1)
        
        # File di log
        logging_layout.addWidget(QLabel("File di log:"), 1, 0)
        self.log_file_edit = QLineEdit()
        self.log_file_edit.setText(os.path.join(DATA_DIR, "logs", "tradinglab.log"))
        logging_layout.addWidget(self.log_file_edit, 1, 1)
        
        self.log_file_button = QPushButton("Sfoglia...")
        self.log_file_button.clicked.connect(self.browse_log_file)
        logging_layout.addWidget(self.log_file_button, 1, 2)
        
        # Rotazione log
        logging_layout.addWidget(QLabel("Rotazione log:"), 2, 0)
        self.log_rotation_combo = QComboBox()
        self.log_rotation_combo.addItem("Giornaliera")
        self.log_rotation_combo.addItem("Settimanale")
        self.log_rotation_combo.addItem("Mensile")
        self.log_rotation_combo.addItem("Mai")
        logging_layout.addWidget(self.log_rotation_combo, 2, 1)
        
        logging_group.setLayout(logging_layout)
        container_layout.addWidget(logging_group)
        
        # Rete
        network_group = QGroupBox("Rete")
        network_layout = QGridLayout()
        
        # Timeout
        network_layout.addWidget(QLabel("Timeout (s):"), 0, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 60)
        self.timeout_spin.setValue(30)
        network_layout.addWidget(self.timeout_spin, 0, 1)
        
        # Retry
        network_layout.addWidget(QLabel("Tentativi:"), 1, 0)
        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(0, 10)
        self.retry_spin.setValue(3)
        network_layout.addWidget(self.retry_spin, 1, 1)
        
        # Proxy
        network_layout.addWidget(QLabel("Usa proxy:"), 2, 0)
        self.proxy_check = QCheckBox()
        self.proxy_check.setChecked(False)
        self.proxy_check.stateChanged.connect(self.on_proxy_changed)
        network_layout.addWidget(self.proxy_check, 2, 1)
        
        # Host proxy
        network_layout.addWidget(QLabel("Host proxy:"), 3, 0)
        self.proxy_host_edit = QLineEdit()
        self.proxy_host_edit.setEnabled(False)
        network_layout.addWidget(self.proxy_host_edit, 3, 1)
        
        # Porta proxy
        network_layout.addWidget(QLabel("Porta proxy:"), 4, 0)
        self.proxy_port_spin = QSpinBox()
        self.proxy_port_spin.setRange(1, 65535)
        self.proxy_port_spin.setValue(8080)
        self.proxy_port_spin.setEnabled(False)
        network_layout.addWidget(self.proxy_port_spin, 4, 1)
        
        network_group.setLayout(network_layout)
        container_layout.addWidget(network_group)
        
        # Pulsanti di manutenzione
        maint_group = QGroupBox("Manutenzione")
        maint_layout = QHBoxLayout()
        
        self.clear_cache_button = QPushButton("Svuota Cache")
        self.clear_cache_button.clicked.connect(self.clear_cache)
        maint_layout.addWidget(self.clear_cache_button)
        
        self.reset_settings_button = QPushButton("Ripristina Default")
        self.reset_settings_button.clicked.connect(self.reset_settings)
        maint_layout.addWidget(self.reset_settings_button)
        
        maint_group.setLayout(maint_layout)
        container_layout.addWidget(maint_group)
        
        # Aggiungi uno stretcher alla fine
        container_layout.addStretch()
        
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)
    
    def load_settings(self):
        """Carica le impostazioni salvate."""
        # Performance
        threads = self.settings.value("threads", 4, type=int)
        self.threads_spin.setValue(threads)
        
        cache_size = self.settings.value("cache_size", 1000, type=int)
        self.cache_spin.setValue(cache_size)
        
        batch_mode = self.settings.value("batch_mode", False, type=bool)
        self.batch_mode_check.setChecked(batch_mode)
        
        # Logging
        log_level = self.settings.value("log_level", "INFO")
        index = self.log_level_combo.findText(log_level)
        if index >= 0:
            self.log_level_combo.setCurrentIndex(index)
        
        log_file = self.settings.value("log_file", os.path.join(DATA_DIR, "logs", "tradinglab.log"))
        self.log_file_edit.setText(log_file)
        
        log_rotation = self.settings.value("log_rotation", "Giornaliera")
        index = self.log_rotation_combo.findText(log_rotation)
        if index >= 0:
            self.log_rotation_combo.setCurrentIndex(index)
        
        # Rete
        timeout = self.settings.value("timeout", 30, type=int)
        self.timeout_spin.setValue(timeout)
        
        retry = self.settings.value("retry", 3, type=int)
        self.retry_spin.setValue(retry)
        
        use_proxy = self.settings.value("use_proxy", False, type=bool)
        self.proxy_check.setChecked(use_proxy)
        
        proxy_host = self.settings.value("proxy_host", "")
        self.proxy_host_edit.setText(proxy_host)
        
        proxy_port = self.settings.value("proxy_port", 8080, type=int)
        self.proxy_port_spin.setValue(proxy_port)
        
        # Aggiorna lo stato dei campi proxy
        self.on_proxy_changed(self.proxy_check.checkState())
    
    def save_settings(self):
        """Salva le impostazioni correnti."""
        # Performance
        self.settings.setValue("threads", self.threads_spin.value())
        self.settings.setValue("cache_size", self.cache_spin.value())
        self.settings.setValue("batch_mode", self.batch_mode_check.isChecked())
        
        # Logging
        self.settings.setValue("log_level", self.log_level_combo.currentText())
        self.settings.setValue("log_file", self.log_file_edit.text())
        self.settings.setValue("log_rotation", self.log_rotation_combo.currentText())
        
        # Rete
        self.settings.setValue("timeout", self.timeout_spin.value())
        self.settings.setValue("retry", self.retry_spin.value())
        self.settings.setValue("use_proxy", self.proxy_check.isChecked())
        self.settings.setValue("proxy_host", self.proxy_host_edit.text())
        self.settings.setValue("proxy_port", self.proxy_port_spin.value())
    
    def browse_log_file(self):
        """Apre una finestra di dialogo per selezionare il file di log."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Seleziona File di Log",
            self.log_file_edit.text(),
            "File di Log (*.log);;Tutti i file (*)"
        )
        
        if filename:
            self.log_file_edit.setText(filename)
    
    def on_proxy_changed(self, state):
        """
        Gestisce il cambio di stato del checkbox proxy.
        
        Args:
            state: Stato del checkbox
        """
        enabled = state == Qt.CheckState.Checked
        self.proxy_host_edit.setEnabled(enabled)
        self.proxy_port_spin.setEnabled(enabled)
    
    def clear_cache(self):
        """Svuota la cache dell'applicazione."""
        reply = QMessageBox.question(
            self,
            "Svuota Cache",
            "Sei sicuro di voler svuotare la cache? Questa operazione non può essere annullata.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # In un'implementazione reale, qui svuoteremmo effettivamente la cache
            # Per ora, mostriamo solo un messaggio
            QMessageBox.information(
                self,
                "Cache Svuotata",
                "La cache è stata svuotata con successo."
            )
    
    def reset_settings(self):
        """Ripristina le impostazioni predefinite."""
        reply = QMessageBox.question(
            self,
            "Ripristina Default",
            "Sei sicuro di voler ripristinare tutte le impostazioni predefinite? Questa operazione non può essere annullata.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Ripristina le impostazioni predefinite
            self.settings.clear()
            
            # Ricarica le impostazioni (ora predefinite)
            self.load_settings()
            
            QMessageBox.information(
                self,
                "Impostazioni Ripristinate",
                "Le impostazioni predefinite sono state ripristinate con successo."
            )


class SettingsDialog(QDialog):
    """Finestra di dialogo per configurare le impostazioni dell'applicazione."""
    
    # Segnale emesso quando cambiano le impostazioni
    settingsChanged = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Inizializza la finestra di dialogo.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia della finestra di dialogo."""
        self.setWindowTitle("Impostazioni")
        self.setMinimumSize(700, 500)
        
        # Layout principale
        layout = QVBoxLayout(self)
        
        # Tab widget per le diverse sezioni
        self.tab_widget = QTabWidget()
        
        # Tab generale
        self.general_tab = GeneralSettingsTab()
        self.tab_widget.addTab(self.general_tab, "Generale")
        
        # Tab broker
        self.brokers_tab = BrokersTab()
        self.tab_widget.addTab(self.brokers_tab, "Broker")
        
        # Tab sistema
        self.system_tab = SystemTab()
        self.tab_widget.addTab(self.system_tab, "Sistema")
        
        # Aggiungi il tab widget al layout
        layout.addWidget(self.tab_widget)
        
        # Pulsanti
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Applica")
        self.apply_button.clicked.connect(self.apply_settings)
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Annulla")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def accept(self):
        """Sovrascrive il metodo accept per salvare le impostazioni."""
        self.apply_settings()
        super().accept()
    
    def apply_settings(self):
        """Applica le impostazioni correnti."""
        # Salva le impostazioni da tutti i tab
        self.general_tab.save_settings()
        self.brokers_tab.save_settings()
        self.system_tab.save_settings()
        
        # Raccogli le impostazioni da inviare con il segnale
        settings_dict = {
            'theme': style_manager.theme.value,
            'language': self.general_tab.language_combo.currentText(),
            'active_broker': self.brokers_tab.broker_combo.currentText()
        }
        
        # Emetti il segnale
        self.settingsChanged.emit(settings_dict)
        
        # Applica il tema
        theme_str = "dark" if self.general_tab.theme_dark_radio.isChecked() else "light"
        theme = Theme.DARK if theme_str == "dark" else Theme.LIGHT
        style_manager.set_theme(theme)
        
        # Se il parent è un QMainWindow, aggiorna il suo foglio di stile
        if self.parent() and hasattr(self.parent(), "setStyleSheet"):
            self.parent().setStyleSheet(style_manager.get_app_stylesheet())