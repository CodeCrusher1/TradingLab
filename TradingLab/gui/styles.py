# Stili dell'interfaccia

"""
Stili e temi dell'interfaccia utente per TradingLab.
Questo modulo definisce colori, font e stili comuni utilizzati nell'applicazione.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# Definizione di colori
class Colors:
    """Palette di colori principale dell'applicazione."""
    PRIMARY = "#2C3E50"        # Blu scuro
    SECONDARY = "#34495E"      # Blu-grigio
    ACCENT = "#3498DB"         # Blu brillante
    WARNING = "#F39C12"        # Arancione
    ERROR = "#E74C3C"          # Rosso
    SUCCESS = "#2ECC71"        # Verde
    INFO = "#3498DB"           # Blu chiaro
    NEUTRAL = "#95A5A6"        # Grigio
    
    # Colori per grafici e indicatori
    CHART_UP = "#2ECC71"       # Verde per candele rialziste
    CHART_DOWN = "#E74C3C"     # Rosso per candele ribassiste
    CHART_VOLUME = "#3498DB"   # Blu per volume
    
    # Colori per indicatori tecnici
    INDICATOR_LINE_1 = "#3498DB"  # Blu
    INDICATOR_LINE_2 = "#E74C3C"  # Rosso
    INDICATOR_LINE_3 = "#2ECC71"  # Verde
    INDICATOR_LINE_4 = "#F39C12"  # Arancione
    INDICATOR_LINE_5 = "#9B59B6"  # Viola
    
    # Varianti di luminosità
    PRIMARY_LIGHT = "#34495E"
    PRIMARY_DARK = "#1A252F"
    
    # Colori di sfondo
    BACKGROUND = "#ECF0F1"     # Bianco sporco
    CARD_BG = "#FFFFFF"        # Bianco
    
    # Colori per il testo
    TEXT_PRIMARY = "#2C3E50"    # Blu scuro
    TEXT_SECONDARY = "#7F8C8D"  # Grigio
    TEXT_DISABLED = "#BDC3C7"   # Grigio chiaro
    TEXT_LIGHT = "#FFFFFF"      # Bianco


class DarkColors(Colors):
    """Palette di colori per tema scuro."""
    PRIMARY = "#1F2933"
    SECONDARY = "#323F4B"
    
    # Varianti di luminosità
    PRIMARY_LIGHT = "#323F4B"
    PRIMARY_DARK = "#1A1E24"
    
    # Colori di sfondo
    BACKGROUND = "#121212"
    CARD_BG = "#1E1E1E"
    
    # Colori per il testo
    TEXT_PRIMARY = "#ECEFF4"
    TEXT_SECONDARY = "#D8DEE9"
    TEXT_DISABLED = "#81858C"


class Theme(Enum):
    """Enum per i temi disponibili."""
    LIGHT = "light"
    DARK = "dark"


@dataclass
class FontConfig:
    """Configurazione dei font dell'applicazione."""
    family: str = "Segoe UI"
    size_small: int = 8
    size_normal: int = 10
    size_large: int = 12
    size_title: int = 14
    size_header: int = 18
    weight_normal: str = "normal"
    weight_bold: str = "bold"


class StyleManager:
    """
    Gestore degli stili dell'applicazione.
    Fornisce metodi per ottenere stili e colori in base al tema attivo.
    """
    
    def __init__(self, theme: Theme = Theme.LIGHT):
        """
        Inizializza il gestore stili.
        
        Args:
            theme: Tema iniziale (default: Theme.LIGHT)
        """
        self._theme = theme
        self._font = FontConfig()
        self._colors = Colors() if theme == Theme.LIGHT else DarkColors()
        
        # Stili predefiniti per widget
        self._button_style = self._create_button_style()
        self._input_style = self._create_input_style()
        self._tooltip_style = self._create_tooltip_style()
        
    def _create_button_style(self) -> str:
        """
        Crea lo stile CSS per i pulsanti.
        
        Returns:
            Stringa CSS per lo stile dei pulsanti
        """
        return f"""
            QPushButton {{
                background-color: {self._colors.ACCENT};
                color: {self._colors.TEXT_LIGHT};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-family: {self._font.family};
                font-size: {self._font.size_normal}pt;
            }}
            
            QPushButton:hover {{
                background-color: {self._colors.PRIMARY_LIGHT};
            }}
            
            QPushButton:pressed {{
                background-color: {self._colors.PRIMARY_DARK};
            }}
            
            QPushButton:disabled {{
                background-color: {self._colors.NEUTRAL};
                color: {self._colors.TEXT_DISABLED};
            }}
        """
    
    def _create_input_style(self) -> str:
        """
        Crea lo stile CSS per gli input.
        
        Returns:
            Stringa CSS per lo stile degli input
        """
        return f"""
            QLineEdit, QTextEdit, QComboBox {{
                background-color: {self._colors.CARD_BG};
                color: {self._colors.TEXT_PRIMARY};
                border: 1px solid {self._colors.NEUTRAL};
                border-radius: 4px;
                padding: 6px;
                font-family: {self._font.family};
                font-size: {self._font.size_normal}pt;
            }}
            
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
                border: 1px solid {self._colors.ACCENT};
            }}
            
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        """
    
    def _create_tooltip_style(self) -> str:
        """
        Crea lo stile CSS per i tooltip.
        
        Returns:
            Stringa CSS per lo stile dei tooltip
        """
        return f"""
            QToolTip {{
                background-color: {self._colors.PRIMARY};
                color: {self._colors.TEXT_LIGHT};
                border: none;
                font-family: {self._font.family};
                font-size: {self._font.size_small}pt;
                padding: 5px;
            }}
        """
    
    def set_theme(self, theme: Theme) -> None:
        """
        Imposta il tema dell'applicazione.
        
        Args:
            theme: Nuovo tema
        """
        if theme != self._theme:
            self._theme = theme
            self._colors = Colors() if theme == Theme.LIGHT else DarkColors()
            
            # Aggiorna gli stili
            self._button_style = self._create_button_style()
            self._input_style = self._create_input_style()
            self._tooltip_style = self._create_tooltip_style()
    
    def get_app_stylesheet(self) -> str:
        """
        Ottiene il foglio di stile completo per l'applicazione.
        
        Returns:
            Stringa CSS con tutti gli stili
        """
        return f"""
            QMainWindow, QDialog {{
                background-color: {self._colors.BACKGROUND};
                color: {self._colors.TEXT_PRIMARY};
                font-family: {self._font.family};
                font-size: {self._font.size_normal}pt;
            }}
            
            QLabel {{
                color: {self._colors.TEXT_PRIMARY};
                font-family: {self._font.family};
                font-size: {self._font.size_normal}pt;
            }}
            
            QGroupBox {{
                font-weight: {self._font.weight_bold};
                border: 1px solid {self._colors.NEUTRAL};
                border-radius: 4px;
                margin-top: 16px;
                padding-top: 16px;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }}
            
            {self._button_style}
            {self._input_style}
            {self._tooltip_style}
            
            QTabWidget::pane {{
                border: 1px solid {self._colors.NEUTRAL};
                border-radius: 4px;
                top: -1px;
            }}
            
            QTabBar::tab {{
                background-color: {self._colors.SECONDARY};
                color: {self._colors.TEXT_LIGHT};
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            
            QTabBar::tab:selected {{
                background-color: {self._colors.PRIMARY};
            }}
            
            QTabBar::tab:hover {{
                background-color: {self._colors.PRIMARY_LIGHT};
            }}
            
            QMenuBar {{
                background-color: {self._colors.PRIMARY};
                color: {self._colors.TEXT_LIGHT};
            }}
            
            QMenuBar::item {{
                padding: 5px 10px;
            }}
            
            QMenuBar::item:selected {{
                background-color: {self._colors.PRIMARY_LIGHT};
            }}
            
            QMenu {{
                background-color: {self._colors.CARD_BG};
                color: {self._colors.TEXT_PRIMARY};
                border: 1px solid {self._colors.NEUTRAL};
            }}
            
            QMenu::item {{
                padding: 5px 20px;
            }}
            
            QMenu::item:selected {{
                background-color: {self._colors.ACCENT};
                color: {self._colors.TEXT_LIGHT};
            }}
            
            QStatusBar {{
                background-color: {self._colors.PRIMARY};
                color: {self._colors.TEXT_LIGHT};
            }}
            
            QCheckBox {{
                color: {self._colors.TEXT_PRIMARY};
            }}
            
            QCheckBox::indicator {{
                width: 15px;
                height: 15px;
            }}
            
            QRadioButton {{
                color: {self._colors.TEXT_PRIMARY};
            }}
            
            QRadioButton::indicator {{
                width: 15px;
                height: 15px;
            }}
            
            QProgressBar {{
                border: 1px solid {self._colors.NEUTRAL};
                border-radius: 4px;
                background-color: {self._colors.CARD_BG};
                text-align: center;
                color: {self._colors.TEXT_PRIMARY};
            }}
            
            QProgressBar::chunk {{
                background-color: {self._colors.ACCENT};
                width: 10px;
            }}
            
            QTableView {{
                background-color: {self._colors.CARD_BG};
                color: {self._colors.TEXT_PRIMARY};
                gridline-color: {self._colors.NEUTRAL};
                selection-background-color: {self._colors.ACCENT};
                selection-color: {self._colors.TEXT_LIGHT};
                border: 1px solid {self._colors.NEUTRAL};
                border-radius: 4px;
            }}
            
            QHeaderView::section {{
                background-color: {self._colors.SECONDARY};
                color: {self._colors.TEXT_LIGHT};
                padding: 5px;
                border: none;
            }}
        """
    
    @property
    def colors(self) -> Colors:
        """Ottiene la palette di colori corrente."""
        return self._colors
    
    @property
    def font(self) -> FontConfig:
        """Ottiene la configurazione font corrente."""
        return self._font
    
    @property
    def theme(self) -> Theme:
        """Ottiene il tema corrente."""
        return self._theme
    
    def get_chart_colors(self) -> Dict[str, str]:
        """
        Ottiene i colori per i grafici.
        
        Returns:
            Dizionario con i colori per i grafici
        """
        return {
            "up": self._colors.CHART_UP,
            "down": self._colors.CHART_DOWN,
            "volume": self._colors.CHART_VOLUME,
            "bg": self._colors.CARD_BG,
            "grid": self._colors.NEUTRAL,
            "text": self._colors.TEXT_PRIMARY,
            "ma1": self._colors.INDICATOR_LINE_1,
            "ma2": self._colors.INDICATOR_LINE_2,
            "ma3": self._colors.INDICATOR_LINE_3,
            "indicators": [
                self._colors.INDICATOR_LINE_1,
                self._colors.INDICATOR_LINE_2,
                self._colors.INDICATOR_LINE_3,
                self._colors.INDICATOR_LINE_4,
                self._colors.INDICATOR_LINE_5
            ]
        }


# Istanza globale del gestore stili
style_manager = StyleManager()