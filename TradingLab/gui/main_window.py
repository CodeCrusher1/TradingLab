# Finestra principale

# Finestra principale

"""
Finestra principale dell'applicazione TradingLab.
Contiene la barra dei menu, la barra degli strumenti e i widget principali.
"""
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QToolBar, QStatusBar, QWidget,
    QVBoxLayout, QHBoxLayout, QSplitter, QMessageBox, QLabel, QMenu, QFileDialog
)
from PyQt6.QtCore import Qt, QSettings, QSize, QTimer
from PyQt6.QtGui import QIcon, QKeySequence, QAction

import os
import logging
from typing import Dict, List, Optional, Union, Tuple, Any

from .dashboard import DashboardWidget
from .charts import ChartWidget
from .controls import (
    SymbolSelector, TimeframeSelector, DateRangeSelector, ModelSelector, 
    IndicatorSelector, TradeSignalView, PredictionResultView, MarketInfoView
)
from .dialogs.backtest import BacktestDialog
from .dialogs.model_training import ModelTrainingDialog
from .dialogs.settings import SettingsDialog
from .styles import style_manager, Theme
from ..utils import app_logger
from ..config import APP_NAME, APP_VERSION, DATA_DIR


class MainWindow(QMainWindow):
    """Finestra principale dell'applicazione TradingLab."""
    
    def __init__(self):
        """Inizializza la finestra principale."""
        super().__init__()
        self.settings = QSettings(APP_NAME, "settings")
        self.init_ui()
        self.load_settings()
        
        # Timer per aggiornamento periodico
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_market_data)
        self.update_timer.start(60000)  # Aggiorna ogni minuto
    
    def init_ui(self):
        """Inizializza l'interfaccia utente."""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1000, 700)
        
        # Imposta il foglio di stile
        self.setStyleSheet(style_manager.get_app_stylesheet())
        
        # Crea la barra dei menu
        self.create_menu_bar()
        
        # Crea la barra degli strumenti
        self.create_toolbar()
        
        # Crea la barra di stato
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Pronto")
        
        # Widget principale
        self.main_tab_widget = QTabWidget()
        self.main_tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.main_tab_widget.setMovable(True)
        
        # Tab Dashboard
        self.dashboard_widget = DashboardWidget()
        self.main_tab_widget.addTab(self.dashboard_widget, "Dashboard")
        
        # Tab Grafico
        self.chart_tab = self.create_chart_tab()
        self.main_tab_widget.addTab(self.chart_tab, "Grafico")
        
        # Tab Backtest
        self.backtest_tab = self.create_backtest_tab()
        self.main_tab_widget.addTab(self.backtest_tab, "Backtest")
        
        # Tab Monitoraggio
        self.monitoring_tab = self.create_monitoring_tab()
        self.main_tab_widget.addTab(self.monitoring_tab, "Monitoraggio")
        
        # Imposta il widget principale
        self.setCentralWidget(self.main_tab_widget)
    
    def create_menu_bar(self):
        """Crea la barra dei menu."""
        menubar = self.menuBar()
        
        # Menu File
        file_menu = menubar.addMenu("File")
        
        # Azioni del menu File
        new_workspace_action = QAction("Nuovo Workspace", self)
        new_workspace_action.setShortcut(QKeySequence.StandardKey.New)
        new_workspace_action.triggered.connect(self.new_workspace)
        file_menu.addAction(new_workspace_action)
        
        open_workspace_action = QAction("Apri Workspace", self)
        open_workspace_action.setShortcut(QKeySequence.StandardKey.Open)
        open_workspace_action.triggered.connect(self.open_workspace)
        file_menu.addAction(open_workspace_action)
        
        save_workspace_action = QAction("Salva Workspace", self)
        save_workspace_action.setShortcut(QKeySequence.StandardKey.Save)
        save_workspace_action.triggered.connect(self.save_workspace)
        file_menu.addAction(save_workspace_action)
        
        file_menu.addSeparator()
        
        import_data_action = QAction("Importa Dati", self)
        import_data_action.triggered.connect(self.import_data)
        file_menu.addAction(import_data_action)
        
        export_data_action = QAction("Esporta Dati", self)
        export_data_action.triggered.connect(self.export_data)
        file_menu.addAction(export_data_action)
        
        file_menu.addSeparator()
        
        settings_action = QAction("Impostazioni", self)
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Esci", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Analisi
        analysis_menu = menubar.addMenu("Analisi")
        
        # Azioni del menu Analisi
        technical_analysis_action = QAction("Analisi Tecnica", self)
        technical_analysis_action.triggered.connect(self.show_technical_analysis)
        analysis_menu.addAction(technical_analysis_action)
        
        pattern_finder_action = QAction("Cerca Pattern", self)
        pattern_finder_action.triggered.connect(self.show_pattern_finder)
        analysis_menu.addAction(pattern_finder_action)
        
        scanner_action = QAction("Scanner Mercato", self)
        scanner_action.triggered.connect(self.show_market_scanner)
        analysis_menu.addAction(scanner_action)
        
        analysis_menu.addSeparator()
        
        correlation_action = QAction("Correlazione", self)
        correlation_action.triggered.connect(self.show_correlation)
        analysis_menu.addAction(correlation_action)
        
        # Menu Trading
        trading_menu = menubar.addMenu("Trading")
        
        # Azioni del menu Trading
        backtest_action = QAction("Backtest", self)
        backtest_action.triggered.connect(self.show_backtest_dialog)
        trading_menu.addAction(backtest_action)
        
        trading_menu.addSeparator()
        
        signals_action = QAction("Segnali", self)
        signals_action.triggered.connect(self.show_signals)
        trading_menu.addAction(signals_action)
        
        performance_action = QAction("Performance", self)
        performance_action.triggered.connect(self.show_performance)
        trading_menu.addAction(performance_action)
        
        # Menu ML
        ml_menu = menubar.addMenu("ML")
        
        # Azioni del menu ML
        train_model_action = QAction("Addestra Modello", self)
        train_model_action.triggered.connect(self.show_model_training)
        ml_menu.addAction(train_model_action)
        
        predict_action = QAction("Previsione", self)
        predict_action.triggered.connect(self.make_prediction)
        ml_menu.addAction(predict_action)
        
        ml_menu.addSeparator()
        
        models_action = QAction("Gestione Modelli", self)
        models_action.triggered.connect(self.show_model_management)
        ml_menu.addAction(models_action)
        
        # Menu Aiuto
        help_menu = menubar.addMenu("Aiuto")
        
        # Azioni del menu Aiuto
        help_action = QAction("Guida", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        about_action = QAction("Informazioni", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Crea la barra degli strumenti."""
        toolbar = QToolBar("Strumenti Principali")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Azioni della barra degli strumenti
        refresh_action = QAction("Aggiorna", self)
        refresh_action.triggered.connect(self.refresh_data)
        toolbar.addAction(refresh_action)
        
        toolbar.addSeparator()
        
        # Selettore simbolo
        self.symbol_selector = SymbolSelector()
        self.symbol_selector.symbolChanged.connect(self.on_symbol_changed)
        toolbar.addWidget(self.symbol_selector)
        
        # Selettore timeframe
        self.timeframe_selector = TimeframeSelector()
        self.timeframe_selector.timeframeChanged.connect(self.on_timeframe_changed)
        toolbar.addWidget(self.timeframe_selector)
        
        toolbar.addSeparator()
        
        # Pulsanti rapidi
        analyze_action = QAction("Analizza", self)
        analyze_action.triggered.connect(self.analyze_current)
        toolbar.addAction(analyze_action)
        
        backtest_action = QAction("Backtest", self)
        backtest_action.triggered.connect(self.show_backtest_dialog)
        toolbar.addAction(backtest_action)
        
        predict_action = QAction("Predici", self)
        predict_action.triggered.connect(self.make_prediction)
        toolbar.addAction(predict_action)
        
        # Tema chiaro/scuro
        theme_action = QAction("Tema", self)
        theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(theme_action)
    
    def create_chart_tab(self) -> QWidget:
        """
        Crea il tab con il grafico principale.
        
        Returns:
            Widget per il tab grafico
        """
        chart_tab = QWidget()
        layout = QVBoxLayout(chart_tab)
        
        # Splitter orizzontale per dividere il grafico e i pannelli laterali
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Panel di controllo a sinistra
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Selettore dell'intervallo di date
        date_selector = DateRangeSelector()
        date_selector.dateRangeChanged.connect(self.on_date_range_changed)
        left_layout.addWidget(date_selector)
        
        # Selettore degli indicatori
        indicator_selector = IndicatorSelector()
        indicator_selector.indicatorsChanged.connect(self.on_indicators_changed)
        left_layout.addWidget(indicator_selector)
        
        # Informazioni di mercato
        market_info = MarketInfoView()
        left_layout.addWidget(market_info)
        
        left_layout.addStretch()
        
        # Widget del grafico al centro
        self.chart_widget = ChartWidget()
        
        # Panel di controllo a destra
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Previsione
        prediction_view = PredictionResultView()
        right_layout.addWidget(prediction_view)
        
        # Segnali di trading
        trade_signals = TradeSignalView()
        right_layout.addWidget(trade_signals)
        
        right_layout.addStretch()
        
        # Aggiungi i pannelli allo splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(self.chart_widget)
        splitter.addWidget(right_panel)
        
        # Imposta le dimensioni iniziali
        splitter.setSizes([200, 600, 200])
        
        layout.addWidget(splitter)
        
        return chart_tab
    
    def create_backtest_tab(self) -> QWidget:
        """
        Crea il tab per il backtest.
        
        Returns:
            Widget per il tab backtest
        """
        backtest_tab = QWidget()
        layout = QVBoxLayout(backtest_tab)
        
        # Placeholder per il tab backtest
        layout.addWidget(QLabel("Tab di Backtest - Implementazione completa nella finestra di dialogo"))
        
        return backtest_tab
    
    def create_monitoring_tab(self) -> QWidget:
        """
        Crea il tab per il monitoraggio di mercato.
        
        Returns:
            Widget per il tab monitoraggio
        """
        monitoring_tab = QWidget()
        layout = QVBoxLayout(monitoring_tab)
        
        # Placeholder per il tab monitoraggio
        layout.addWidget(QLabel("Tab di Monitoraggio - Da implementare"))
        
        return monitoring_tab
    
    def load_settings(self):
        """Carica le impostazioni salvate."""
        # Posizione e dimensione della finestra
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Imposta il tema
        theme_str = self.settings.value("theme", "light")
        theme = Theme.LIGHT if theme_str == "light" else Theme.DARK
        style_manager.set_theme(theme)
        self.setStyleSheet(style_manager.get_app_stylesheet())
        
        # Simbolo e timeframe selezionati
        symbol = self.settings.value("symbol")
        if symbol:
            self.symbol_selector.set_selected_symbol(symbol)
        
        timeframe = self.settings.value("timeframe")
        if timeframe:
            self.timeframe_selector.set_selected_timeframe(timeframe)
    
    def save_settings(self):
        """Salva le impostazioni correnti."""
        # Posizione e dimensione della finestra
        self.settings.setValue("geometry", self.saveGeometry())
        
        # Tema
        theme_str = "light" if style_manager.theme == Theme.LIGHT else "dark"
        self.settings.setValue("theme", theme_str)
        
        # Simbolo e timeframe selezionati
        self.settings.setValue("symbol", self.symbol_selector.get_selected_symbol())
        self.settings.setValue("timeframe", self.timeframe_selector.get_selected_timeframe())
    
    def closeEvent(self, event):
        """
        Gestisce l'evento di chiusura della finestra.
        
        Args:
            event: Evento di chiusura
        """
        # Salva le impostazioni
        self.save_settings()
        
        # Chiede conferma prima di uscire
        reply = QMessageBox.question(
            self, 
            'Conferma Uscita',
            'Sei sicuro di voler uscire?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()
    
    def update_market_data(self):
        """Aggiorna i dati di mercato periodicamente."""
        # Aggiorna i dati solo se l'applicazione è attiva
        if self.isActiveWindow() and self.isVisible():
            symbol = self.symbol_selector.get_selected_symbol()
            timeframe = self.timeframe_selector.get_selected_timeframe()
            
            self.statusbar.showMessage(f"Aggiornamento dati per {symbol} ({timeframe})...")
            
            # Qui dovremmo inserire la logica per aggiornare i dati
            # Per ora è solo un esempio
            
            self.statusbar.showMessage(f"Dati aggiornati: {symbol} ({timeframe})")
    
    def refresh_data(self):
        """Aggiorna manualmente i dati."""
        self.update_market_data()
    
    def on_symbol_changed(self, symbol: str):
        """
        Gestisce il cambio di simbolo.
        
        Args:
            symbol: Nuovo simbolo selezionato
        """
        self.statusbar.showMessage(f"Simbolo selezionato: {symbol}")
        
        # Aggiorna i dati
        self.update_market_data()
    
    def on_timeframe_changed(self, timeframe: str):
        """
        Gestisce il cambio di timeframe.
        
        Args:
            timeframe: Nuovo timeframe selezionato
        """
        self.statusbar.showMessage(f"Timeframe selezionato: {timeframe}")
        
        # Aggiorna i dati
        self.update_market_data()
    
    def on_date_range_changed(self, start_date, end_date):
        """
        Gestisce il cambio di intervallo date.
        
        Args:
            start_date: Data di inizio
            end_date: Data di fine
        """
        self.statusbar.showMessage(f"Intervallo date: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")
        
        # Aggiorna i dati
        self.update_market_data()
    
    def on_indicators_changed(self, indicators: dict):
        """
        Gestisce il cambio di indicatori selezionati.
        
        Args:
            indicators: Dizionario con gli indicatori selezionati
        """
        # Aggiorna gli indicatori sul grafico
        pass
    
    def toggle_theme(self):
        """Alterna tra tema chiaro e scuro."""
        current_theme = style_manager.theme
        new_theme = Theme.DARK if current_theme == Theme.LIGHT else Theme.LIGHT
        
        style_manager.set_theme(new_theme)
        self.setStyleSheet(style_manager.get_app_stylesheet())
        
        theme_name = "scuro" if new_theme == Theme.DARK else "chiaro"
        self.statusbar.showMessage(f"Tema {theme_name} applicato")
    
    def analyze_current(self):
        """Esegue l'analisi tecnica sul simbolo corrente."""
        symbol = self.symbol_selector.get_selected_symbol()
        timeframe = self.timeframe_selector.get_selected_timeframe()
        
        self.statusbar.showMessage(f"Analisi di {symbol} ({timeframe})...")
        
        # Implementazione analisi tecnica
        # ...
        
        self.statusbar.showMessage(f"Analisi completata per {symbol} ({timeframe})")
    
    def new_workspace(self):
        """Crea un nuovo workspace."""
        reply = QMessageBox.question(
            self, 
            'Nuovo Workspace',
            'Creare un nuovo workspace? Eventuali modifiche non salvate andranno perse.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Resetta lo stato dell'applicazione
            pass
    
    def open_workspace(self):
        """Apre un workspace esistente."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Apri Workspace",
            "",
            "TradingLab Workspace (*.tlw);;Tutti i file (*)"
        )
        
        if filename:
            # Carica il workspace
            self.statusbar.showMessage(f"Workspace caricato: {filename}")
    
    def save_workspace(self):
        """Salva il workspace corrente."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Salva Workspace",
            "",
            "TradingLab Workspace (*.tlw);;Tutti i file (*)"
        )
        
        if filename:
            # Salva il workspace
            self.statusbar.showMessage(f"Workspace salvato: {filename}")
    
    def import_data(self):
        """Importa dati da file esterni."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Importa Dati",
            "",
            "CSV (*.csv);;Excel (*.xlsx);;Tutti i file (*)"
        )
        
        if filename:
            # Importa i dati
            self.statusbar.showMessage(f"Dati importati: {filename}")
    
    def export_data(self):
        """Esporta dati in file esterni."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Esporta Dati",
            "",
            "CSV (*.csv);;Excel (*.xlsx);;Tutti i file (*)"
        )
        
        if filename:
            # Esporta i dati
            self.statusbar.showMessage(f"Dati esportati: {filename}")
    
    def show_settings(self):
        """Mostra la finestra di dialogo delle impostazioni."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Applica le impostazioni
            self.setStyleSheet(style_manager.get_app_stylesheet())
    
    def show_technical_analysis(self):
        """Mostra la finestra di analisi tecnica."""
        # Implementazione
        pass
    
    def show_pattern_finder(self):
        """Mostra la finestra di ricerca pattern."""
        # Implementazione
        pass
    
    def show_market_scanner(self):
        """Mostra la finestra di scanner di mercato."""
        # Implementazione
        pass
    
    def show_correlation(self):
        """Mostra la finestra di analisi di correlazione."""
        # Implementazione
        pass
    
    def show_backtest_dialog(self):
        """Mostra la finestra di dialogo per il backtest."""
        dialog = BacktestDialog(self)
        dialog.exec()
    
    def show_signals(self):
        """Mostra la finestra dei segnali di trading."""
        # Implementazione
        pass
    
    def show_performance(self):
        """Mostra la finestra delle performance di trading."""
        # Implementazione
        pass
    
    def show_model_training(self):
        """Mostra la finestra di dialogo per l'addestramento dei modelli."""
        dialog = ModelTrainingDialog(self)
        dialog.exec()
    
    def make_prediction(self):
        """Esegue una previsione con il modello selezionato."""
        symbol = self.symbol_selector.get_selected_symbol()
        timeframe = self.timeframe_selector.get_selected_timeframe()
        
        self.statusbar.showMessage(f"Previsione per {symbol} ({timeframe})...")
        
        # Implementazione previsione
        # ...
        
        self.statusbar.showMessage(f"Previsione completata per {symbol} ({timeframe})")
    
    def show_model_management(self):
        """Mostra la finestra di gestione dei modelli."""
        # Implementazione
        pass
    
    def show_help(self):
        """Mostra la guida dell'applicazione."""
        QMessageBox.information(
            self,
            "Guida",
            f"Guida di {APP_NAME} v{APP_VERSION}.\n\n"
            "Consulta la documentazione per ulteriori informazioni."
        )
    
    def show_about(self):
        """Mostra informazioni sull'applicazione."""
        QMessageBox.about(
            self,
            f"Informazioni su {APP_NAME}",
            f"<h1>{APP_NAME} v{APP_VERSION}</h1>"
            "<p>Un'applicazione per analisi tecnica e trading algoritmico.</p>"
            "<p>&copy; 2023-2025 Your Company</p>"
        )