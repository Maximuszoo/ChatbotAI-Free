"""
Google Gemini-Inspired Dark Mode Styles
Modern, minimal, clean design
"""

# Color Palette (Gemini Dark)
COLORS = {
    'background': '#131314',
    'surface': '#1E1F20',
    'surface_variant': '#282A2C',
    'user_bubble': '#004A77',
    'bot_bubble': '#1E1F20',
    'text_primary': '#E3E3E3',
    'text_secondary': '#9AA0A6',
    'accent': '#8AB4F8',
    'mic_button': '#1A73E8',
    'mic_recording': '#EA4335',
    'border': '#3C4043',
}

GEMINI_STYLE = """
/* ===== MAIN WINDOW ===== */
QMainWindow {
    background-color: #131314;
}

QWidget {
    background-color: #131314;
    color: #E3E3E3;
    font-family: 'Google Sans', 'Segoe UI', 'Roboto', Arial, sans-serif;
}

/* ===== SCROLL AREA ===== */
QScrollArea {
    background-color: #131314;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: #131314;
}

QWidget#chatContainer {
    background-color: #131314;
}

/* ===== HEADER ===== */
QWidget#headerWidget {
    background-color: #131314;
    border-bottom: 1px solid #282A2C;
}

/* ===== MODEL SELECTOR - Pill Style ===== */
QComboBox#modelSelector {
    background-color: #282A2C;
    color: #E3E3E3;
    border: 1px solid #3C4043;
    border-radius: 18px;
    padding: 10px 20px;
    font-size: 13px;
    min-width: 200px;
}

QComboBox#modelSelector:hover {
    background-color: #3C4043;
    border-color: #5F6368;
}

QComboBox#modelSelector::drop-down {
    border: none;
    padding-right: 15px;
}

QComboBox#modelSelector::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #9AA0A6;
    margin-right: 10px;
}

QComboBox#modelSelector QAbstractItemView {
    background-color: #282A2C;
    color: #E3E3E3;
    selection-background-color: #3C4043;
    border: 1px solid #3C4043;
    border-radius: 12px;
    padding: 8px;
    outline: none;
}

QComboBox#modelSelector QAbstractItemView::item {
    padding: 10px 16px;
    border-radius: 8px;
    margin: 2px 4px;
}

QComboBox#modelSelector QAbstractItemView::item:hover {
    background-color: #3C4043;
}

/* ===== STATUS LABEL ===== */
QLabel#statusLabel {
    color: #9AA0A6;
    font-size: 13px;
    padding: 8px 16px;
    background: transparent;
}

/* ===== MESSAGE BUBBLES ===== */
QFrame#userBubble {
    background-color: #004A77;
    border-radius: 20px;
    border: none;
}

QFrame#botBubble {
    background-color: #1E1F20;
    border-radius: 20px;
    border: none;
}

QLabel#bubbleText {
    color: #E3E3E3;
    font-size: 15px;
    padding: 14px 18px;
    background: transparent;
}

/* ===== INPUT BAR ===== */
QWidget#inputBar {
    background-color: #1E1F20;
    border-top: 1px solid #282A2C;
}

/* ===== MIC BUTTON - Large Circular ===== */
QPushButton#micButton {
    background-color: #1A73E8;
    color: white;
    border: none;
    border-radius: 32px;
    min-width: 64px;
    max-width: 64px;
    min-height: 64px;
    max-height: 64px;
    font-size: 26px;
}

QPushButton#micButton:hover {
    background-color: #4285F4;
}

QPushButton#micButton:pressed {
    background-color: #1557B0;
}

/* ===== CLEAR BUTTON - Ghost ===== */
QPushButton#clearButton {
    background-color: transparent;
    color: #9AA0A6;
    border: 1px solid #3C4043;
    border-radius: 20px;
    padding: 10px 20px;
    font-size: 13px;
}

QPushButton#clearButton:hover {
    background-color: #282A2C;
    color: #E3E3E3;
    border-color: #5F6368;
}

/* ===== SCROLLBAR - Minimal ===== */
QScrollBar:vertical {
    border: none;
    background-color: #131314;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #3C4043;
    border-radius: 4px;
    min-height: 40px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5F6368;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}

/* ===== RECORDING PULSE ===== */
QLabel#recordingDot {
    background-color: #EA4335;
    border-radius: 6px;
    min-width: 12px;
    max-width: 12px;
    min-height: 12px;
    max-height: 12px;
}

/* ===== LIVE MODE ===== */
QWidget#liveModeWidget {
    background-color: #131314;
}

QWidget#liveIndicatorContainer {
    background-color: transparent;
}

QLabel#liveStateLabel {
    color: #9AA0A6;
    font-size: 16px;
    font-weight: bold;
    background: transparent;
}

QLabel#liveTranscriptLabel {
    color: #E3E3E3;
    font-size: 14px;
    background: transparent;
    padding: 20px;
}

QPushButton#liveMuteButton {
    background-color: #282A2C;
    color: #E3E3E3;
    border: 2px solid #3C4043;
    border-radius: 35px;
    min-width: 70px;
    max-width: 70px;
    min-height: 70px;
    max-height: 70px;
    font-size: 28px;
}

QPushButton#liveMuteButton:hover {
    background-color: #3C4043;
    border-color: #5F6368;
}

QPushButton#liveMuteButton:checked {
    background-color: #EA4335;
    border-color: #EA4335;
}

QPushButton#liveExitButton {
    background-color: #EA4335;
    color: white;
    border: none;
    border-radius: 35px;
    min-width: 70px;
    max-width: 70px;
    min-height: 70px;
    max-height: 70px;
    font-size: 28px;
}

QPushButton#liveExitButton:hover {
    background-color: #F44336;
}

QPushButton#liveStartButton {
    background-color: #1A73E8;
    color: white;
    border: none;
    border-radius: 30px;
    min-width: 60px;
    max-width: 60px;
    min-height: 60px;
    max-height: 60px;
    font-size: 24px;
}

QPushButton#liveStartButton:hover {
    background-color: #4285F4;
}
"""

# For backwards compatibility
DARK_STYLE = GEMINI_STYLE
