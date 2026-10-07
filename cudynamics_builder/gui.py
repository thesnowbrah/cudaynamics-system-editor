from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtGui import QDoubleValidator, QFont, QIntValidator, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from .generator import generate
from .editor import (
    RANGE_KINDS, apply_system_edit_plan, build_system_edit_plan,
    load_editable_system, make_edit_identity_unique, validate_renames,
)
from .installer import (
    apply_plan, apply_removal_plan, build_plan, build_removal_plan,
    ensure_unique_system_identity, list_library_systems, locate_default_project,
)
from .model import (
    AttributeDefaults, ConditionalEvent, Equation, EquationKind, METHODS, SystemSpec,
    time_to_steps,
)
from .ocr import recognize_image
from .parser import (
    ParseError, normalize_reserved_identifiers, parse_assignments, parse_system,
    pretty_equations, safe_identifier, validate_spec,
)


STYLE = """
QWidget { background: #101820; color: #d8e5e9; font-family: "Segoe UI"; font-size: 10pt; }
QMainWindow { background: #0c141b; }
QFrame#sidebar { background: #0b222a; border-right: 1px solid #1f4d59; }
QLabel#brand { color: #57d7dd; font-size: 18pt; font-weight: 700; padding: 18px 12px; }
QLabel#muted { color: #86a1aa; }
QLabel#section { color: #75e0df; font-size: 15pt; font-weight: 600; }
QPushButton { background: #173742; border: 1px solid #286574; border-radius: 4px; padding: 8px 12px; }
QPushButton:hover { background: #20505d; border-color: #48bec5; }
QPushButton:pressed { background: #12303a; }
QPushButton#primary { background: #168b95; border-color: #54dde0; color: white; font-weight: 600; }
QPushButton#nav { text-align: left; border: 0; border-left: 3px solid transparent; background: transparent; padding: 11px 14px; }
QPushButton#nav:checked { background: #163943; border-left-color: #54d8dc; color: #71e3e4; }
QLineEdit, QTextEdit, QComboBox, QTableWidget {
    background: #0b1319; border: 1px solid #294751; border-radius: 3px; selection-background-color: #197f89;
}
QLineEdit, QComboBox { min-height: 28px; padding: 2px 6px; }
QHeaderView::section { background: #17333d; color: #b8d5da; border: 0; border-right: 1px solid #294751; padding: 7px; }
QGroupBox { border: 1px solid #294751; border-radius: 5px; margin-top: 12px; padding-top: 12px; }
QGroupBox::title { color: #63d7d8; subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QScrollBar:vertical { background: #0c151b; width: 10px; }
QScrollBar::handle:vertical { background: #315762; min-height: 24px; border-radius: 4px; }
"""


UI_EN = {
    "1  Ввод и OCR": "1  Input and OCR", "2  Проверка модели": "2  Model review",
    "3  Методы и значения": "3  Methods and values", "4  Добавление": "4  Installation",
    "Изменения CUDAynamics выполняются\nтолько на последнем шаге.": "CUDAynamics is modified only at the final step.",
    "← Вернуться в меню": "← Back to menu", "Готово": "Ready",
    "Источник уравнений": "Equation source",
    "Загрузите изображение — это основной сценарий — либо вставьте уравнения текстом. После OCR текст всегда можно исправить.": "Load an image—the primary workflow—or paste equations as text. OCR output can always be edited.",
    "Изображение": "Image", "Перетащите сюда PNG/JPG/WebP/BMP/PDF\nили выберите файл кнопкой ниже": "Drop PNG/JPG/WebP/BMP/PDF here\nor choose a file below",
    "Выбрать изображение…": "Choose image…", "OpenAI API key (не сохраняется)": "OpenAI API key (not stored)",
    "Распознать изображение": "Recognize image", "Распознанный / введённый текст": "Recognized / entered text",
    "ID папки и ядра": "Folder and kernel ID", "Название в CUDAynamics": "Name in CUDAynamics",
    "Разобрать и показать модель →": "Parse and show model →",
    "Проверка модели": "Model review",
    "Состояния интегрируются, алгебраические переменные подставляются на каждой стадии, сигнал вычисляется по выбранной форме.": "States are integrated, algebraic variables are substituted at every stage, and the signal uses the selected waveform.",
    "Переменная": "Variable", "Тип": "Type", "Правая часть": "Right-hand side",
    "+ Уравнение": "+ Equation", "Удалить строку": "Delete row", "Условные изменения": "Conditional changes",
    "Позиция": "Position", "Условие": "Condition", "Одновременные присваивания через ;": "Simultaneous assignments separated by ;",
    "+ Условие": "+ Condition", "Удалить условие": "Delete condition", "Модель ещё не разобрана": "Model has not been parsed yet",
    "Проверить модель": "Validate model", "Методы и значения по умолчанию": "Methods and default values",
    "Выберите создаваемые схемы. Значения попадут в .txt-описание системы и будут начальными в CUDAynamics.": "Select generated schemes. Values are written to the system .txt file and used as CUDAynamics defaults.",
    "Численные методы": "Numerical methods", "Моделирование": "Simulation", "Шаг h": "Step h",
    "Время моделирования": "Simulation time", "Время переходного процесса": "Transient time",
    "Начальные значения переменных": "Initial variable values", "Имя": "Name", "Роль": "Role", "Режим": "Mode",
    "Минимум / значение": "Minimum / value", "Максимум": "Maximum", "Количество": "Count",
    "Значения параметров": "Parameter values", "Применить настройки и подготовить изменения →": "Apply settings and prepare changes →",
    "Добавление в CUDAynamics": "Install into CUDAynamics",
    "Просмотрите точный diff. При установке исходные файлы регистрации сохраняются в .cudynamics-builder-backups.": "Review the exact diff. Original registration files are backed up during installation.",
    "Папка…": "Browse…", "Обновить предпросмотр": "Refresh preview", "Проект": "Project",
    "Добавить систему в CUDAynamics": "Add system to CUDAynamics",
    "Состояние (интегрировать)": "State (integrate)", "Алгебраическая": "Algebraic", "Входной сигнал": "Input signal",
    "В конце шага": "At end of step", "Отметить строку": "Select row",
    "Fixed (одно значение)": "Fixed (one value)", "Count (диапазон)": "Count (range)",
    "Fixed — одно значение. Count — указанное число равномерных значений от минимума до максимума.": "Fixed is one value. Count creates evenly spaced values from minimum to maximum.",
}

METHODS_EN = {
    "ExplicitEuler": "Explicit Euler", "SemiExplicitEuler": "Semi-explicit Euler",
    "ImplicitEuler": "Implicit Euler (Newton)", "ExplicitMidpoint": "Explicit midpoint",
    "ImplicitMidpoint": "Implicit midpoint (Newton)", "ExplicitRungeKutta4": "Runge–Kutta 4",
    "ExplicitDormandPrince8": "Dormand–Prince 8 (fixed step)", "VariableSymmetryCD": "Variable Symmetry CD",
}

RUNTIME_EN_REPLACEMENTS = (
    ("Не удалось распознать левую часть", "Could not recognize the left-hand side"),
    ("строка", "line"), ("нет знака", "missing sign"),
    ("Система не содержит уравнений", "The system contains no equations"),
    ("Повторно заданы переменные", "Variables defined more than once"),
    ("Циклическая зависимость алгебраических переменных", "Cyclic algebraic-variable dependency"),
    ("ID системы должен быть корректным идентификатором C/C++", "System ID must be a valid C/C++ identifier"),
    ("Нужно хотя бы одно дифференциальное уравнение", "At least one differential equation is required"),
    ("Текущий формат CUDAynamics поддерживает не более одного входного сигнала", "The current CUDAynamics format supports at most one input signal"),
    ("Для входного сигнала требуется состояние времени t", "An input signal requires time state t"),
    ("Выберите хотя бы один метод интегрирования", "Select at least one integration method"),
    ("Метод по умолчанию должен входить в список создаваемых методов", "Default method must be one of the generated methods"),
    ("Неподдерживаемые методы", "Unsupported methods"),
    ("Зарезервированные имена параметров", "Reserved parameter names"),
    ("Зарезервированные имена переменных", "Reserved variable names"),
    ("неизвестные символы", "unknown symbols"), ("Условие", "Condition"),
    ("не содержит присваиваний", "contains no assignments"),
    ("нельзя присваивать входному сигналу", "cannot assign to input signal"),
    ("В присваивании нет", "Assignment is missing"),
    ("присваивается дважды в одном условии", "is assigned twice in one condition"),
    ("Добавьте хотя бы одно присваивание", "Add at least one assignment"),
    ("Переменная", "Variable"), ("Параметр", "Parameter"),
    ("переименована в", "renamed to"), ("переименован в", "renamed to"),
    ("имя занято CUDAynamics", "name is reserved by CUDAynamics"),
    ("Файл изображения не найден", "Image file not found"),
    ("недоступен", "is unavailable"), ("Укажите", "Provide"),
    ("Нет OCR", "No OCR backend"), ("Ошибка OpenAI OCR", "OpenAI OCR error"),
    ("OCR не вернул текст", "OCR returned no text"),
    ("Время и шаг должны быть конечными числами", "Time and step must be finite numbers"),
    ("Шаг h должен быть больше нуля", "Step h must be greater than zero"),
    ("Время должно быть", "Time must be"), ("больше нуля", "greater than zero"),
    ("неотрицательным", "non-negative"),
)

RUNTIME_RU_REPLACEMENTS = (
    ("Selected folder is not a CUDAynamics project: systems directory is missing", "Выбранная папка не является проектом CUDAynamics: папка systems не найдена"),
    ("Selected folder is not a CUDAynamics project. Missing", "Выбранная папка не является проектом CUDAynamics. Не найдены"),
    ("Select at least one system to delete", "Выберите хотя бы одну систему для удаления"),
    ("System directories were not found", "Папки систем не найдены"),
    ("Removal failed and changes were rolled back", "Удаление не удалось, изменения отменены"),
    ("System cannot be edited; required files are missing", "Систему нельзя редактировать: отсутствуют обязательные файлы"),
    ("Could not find enum", "Не удалось найти перечисление"),
    ("System ID must be a valid C/C++ identifier", "ID системы должен быть корректным идентификатором C/C++"),
    ("System ID", "ID системы"),
    ("is not a valid C/C++ identifier", "не является корректным идентификатором C/C++"),
    ("Attribute name", "Имя атрибута"), ("is reserved by CUDAynamics", "зарезервировано CUDAynamics"),
    ("Duplicate attribute names", "Повторяющиеся имена атрибутов"),
    ("Target system directory already exists", "Целевая папка системы уже существует"),
    ("Required registration file is missing", "Отсутствует обязательный регистрационный файл"),
    ("Could not find registration for system", "Не удалось найти регистрацию системы"),
    ("Editing failed and changes were rolled back", "Редактирование не удалось, изменения отменены"),
    ("Default method must be one of the methods in enum method", "Метод по умолчанию должен быть одним из методов enum method"),
    ("already exists and was changed to", "уже существует и было изменено на"),
)


def ui_text(language: str, russian: str) -> str:
    return russian if language == "ru" else UI_EN.get(russian, russian)


def runtime_text(language: str, text: str) -> str:
    if language == "ru":
        for english, russian in RUNTIME_RU_REPLACEMENTS:
            text = text.replace(english, russian)
        return text
    for russian, english in RUNTIME_EN_REPLACEMENTS:
        text = text.replace(russian, english)
    return text


def localize_widget_tree(root: QWidget, language: str) -> None:
    if language == "ru":
        return
    for widget in [root, *root.findChildren(QWidget)]:
        if isinstance(widget, QLabel):
            widget.setText(ui_text(language, widget.text()))
        elif isinstance(widget, QPushButton):
            widget.setText(ui_text(language, widget.text()))
        elif isinstance(widget, QGroupBox):
            widget.setTitle(ui_text(language, widget.title()))
        if isinstance(widget, (QLineEdit, QTextEdit)):
            widget.setPlaceholderText(ui_text(language, widget.placeholderText()))
        if isinstance(widget, QComboBox):
            for index in range(widget.count()):
                widget.setItemText(index, ui_text(language, widget.itemText(index)))
        if isinstance(widget, QTableWidget):
            for column in range(widget.columnCount()):
                item = widget.horizontalHeaderItem(column)
                if item:
                    item.setText(ui_text(language, item.text()))


class NoWheelComboBox(QComboBox):
    """Combo box whose value cannot be changed accidentally by wheel hover."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt virtual method name
        event.ignore()


class ImageDropLabel(QLabel):
    """Image preview that accepts supported files dragged from Explorer."""

    imageDropped = Signal(str)
    SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"}

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.setAcceptDrops(True)

    @classmethod
    def _supported_url(cls, url) -> bool:
        return url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in cls.SUPPORTED_SUFFIXES

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt virtual method name
        if any(self._supported_url(url) for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt virtual method name
        if any(self._supported_url(url) for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt virtual method name
        for url in event.mimeData().urls():
            if self._supported_url(url):
                self.imageDropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
        event.ignore()


class MainWindow(QMainWindow):
    def __init__(self, root: Path, language: str = "en", return_to_menu=None):
        super().__init__()
        self.root = root
        self.language = language
        self.return_to_menu = return_to_menu
        self.spec: SystemSpec | None = None
        self.image_path = ""
        self.plan = None
        self.setWindowTitle("CUDAynamics · " + ("Конструктор систем" if language == "ru" else "System Creator"))
        self.resize(1320, 840)
        self.setStyleSheet(STYLE)
        self._build_ui()
        localize_widget_tree(self, self.language)
        self._set_page(0)

    def _t(self, russian: str) -> str:
        return ui_text(self.language, russian)

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QFrame(objectName="sidebar")
        sidebar.setFixedWidth(245)
        side = QVBoxLayout(sidebar)
        brand = QLabel("CUDAynamics\nSystem Builder", objectName="brand")
        side.addWidget(brand)
        self.nav = []
        for index, title in enumerate(("1  Ввод и OCR", "2  Проверка модели", "3  Методы и значения", "4  Добавление")):
            button = QPushButton(title, objectName="nav", checkable=True)
            button.clicked.connect(lambda _=False, i=index: self._set_page(i))
            side.addWidget(button)
            self.nav.append(button)
        side.addStretch()
        if self.return_to_menu is not None:
            back = QPushButton("← Вернуться в меню")
            back.clicked.connect(self._back_to_menu)
            side.addWidget(back)
        note = QLabel("Изменения CUDAynamics выполняются\nтолько на последнем шаге.", objectName="muted")
        note.setWordWrap(True)
        side.addWidget(note)
        outer.addWidget(sidebar)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._input_page())
        self.pages.addWidget(self._model_page())
        self.pages.addWidget(self._settings_page())
        self.pages.addWidget(self._install_page())
        outer.addWidget(self.pages, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Готово" if self.language == "ru" else "Ready")

    def _back_to_menu(self) -> None:
        callback = self.return_to_menu
        self.return_to_menu = None
        if callback:
            callback()
        self.close()

    def _page_shell(self, title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        header = QLabel(title, objectName="section")
        layout.addWidget(header)
        hint = QLabel(subtitle, objectName="muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return page, layout

    def _input_page(self) -> QWidget:
        page, layout = self._page_shell("Источник уравнений", "Загрузите изображение — это основной сценарий — либо вставьте уравнения текстом. После OCR текст всегда можно исправить.")
        splitter = QSplitter()
        left = QWidget(); left_l = QVBoxLayout(left)
        image_box = QGroupBox("Изображение")
        image_l = QVBoxLayout(image_box)
        self.image_preview = ImageDropLabel("Перетащите сюда PNG/JPG/WebP/BMP/PDF\nили выберите файл кнопкой ниже")
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setMinimumHeight(290)
        self.image_preview.setStyleSheet("border: 1px dashed #3b6872; background:#091116;")
        self.image_preview.imageDropped.connect(self._load_image)
        image_l.addWidget(self.image_preview)
        choose = QPushButton("Выбрать изображение…")
        choose.clicked.connect(self._choose_image)
        image_l.addWidget(choose)
        ocr_row = QHBoxLayout()
        self.ocr_backend = QComboBox(); self.ocr_backend.addItems(["pix2tex", "openai", "auto"])
        self.api_key = QLineEdit(); self.api_key.setPlaceholderText("OpenAI API key (не сохраняется)"); self.api_key.setEchoMode(QLineEdit.Password)
        self.ocr_backend.currentTextChanged.connect(self._update_api_key_visibility)
        ocr_row.addWidget(self.ocr_backend); ocr_row.addWidget(self.api_key, 1)
        self._update_api_key_visibility(self.ocr_backend.currentText())
        image_l.addLayout(ocr_row)
        recognize = QPushButton("Распознать изображение", objectName="primary")
        recognize.clicked.connect(self._recognize)
        image_l.addWidget(recognize)
        left_l.addWidget(image_box)
        splitter.addWidget(left)

        right = QWidget(); right_l = QVBoxLayout(right)
        right_l.addWidget(QLabel("Распознанный / введённый текст"))
        self.source_text = QTextEdit()
        self.source_text.setPlaceholderText(
            "dx/dt = sigma*(y-x)\ndy/dt = x*(rho-z)-y\ndz/dt = x*y-beta*z\n\n"
            + ("Алгебраическая переменная: q = x+y" if self.language == "ru" else "Algebraic variable: q = x+y")
        )
        right_l.addWidget(self.source_text, 1)
        names = QFormLayout()
        self.system_id = QLineEdit("new_system")
        self.display_name = QLineEdit("New system")
        names.addRow("ID папки и ядра", self.system_id)
        names.addRow("Название в CUDAynamics", self.display_name)
        right_l.addLayout(names)
        parse = QPushButton("Разобрать и показать модель →", objectName="primary")
        parse.clicked.connect(self._parse)
        right_l.addWidget(parse)
        splitter.addWidget(right)
        splitter.setSizes([500, 650])
        layout.addWidget(splitter, 1)
        return page

    def _model_page(self) -> QWidget:
        page, layout = self._page_shell("Проверка модели", "Состояния интегрируются, алгебраические переменные подставляются на каждой стадии, сигнал вычисляется по выбранной форме.")
        self.eq_table = QTableWidget(0, 4)
        self.eq_table.setHorizontalHeaderLabels(["✓", "Переменная", "Тип", "Правая часть"])
        self.eq_table.verticalHeader().setVisible(False)
        self.eq_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.eq_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.eq_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.eq_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.eq_table.setColumnWidth(0, 38)
        self.eq_table.setColumnWidth(2, 250)
        self.eq_table.horizontalHeader().sectionClicked.connect(
            lambda section: self._toggle_all_checks(self.eq_table) if section == 0 else None
        )
        layout.addWidget(self.eq_table, 2)
        buttons = QHBoxLayout()
        add = QPushButton("+ Уравнение"); add.clicked.connect(lambda: self._add_equation_row())
        remove = QPushButton("Удалить строку"); remove.clicked.connect(self._remove_equation_row)
        buttons.addWidget(add); buttons.addWidget(remove); buttons.addStretch()
        layout.addLayout(buttons)
        event_box = QGroupBox("Условные изменения")
        event_layout = QVBoxLayout(event_box)
        self.event_table = QTableWidget(0, 4)
        self.event_table.setHorizontalHeaderLabels(
            ["✓", "Позиция", "Условие", "Одновременные присваивания через ;"]
        )
        self.event_table.verticalHeader().setVisible(False)
        self.event_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.event_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.event_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.event_table.setColumnWidth(0, 38)
        self.event_table.horizontalHeader().sectionClicked.connect(
            lambda section: self._toggle_all_checks(self.event_table) if section == 0 else None
        )
        event_layout.addWidget(self.event_table)
        event_buttons = QHBoxLayout()
        add_event = QPushButton("+ Условие"); add_event.clicked.connect(lambda: self._add_event_row())
        remove_event = QPushButton("Удалить условие"); remove_event.clicked.connect(self._remove_event_row)
        event_buttons.addWidget(add_event); event_buttons.addWidget(remove_event); event_buttons.addStretch()
        event_layout.addLayout(event_buttons)
        layout.addWidget(event_box, 1)
        self.model_summary = QLabel("Модель ещё не разобрана", objectName="muted"); self.model_summary.setWordWrap(True)
        layout.addWidget(self.model_summary)
        validation_row = QHBoxLayout()
        validation_row.addStretch()
        validate = QPushButton("Проверить модель", objectName="primary")
        validate.clicked.connect(self._sync_model)
        validation_row.addWidget(validate)
        layout.addLayout(validation_row)
        return page

    def _settings_page(self) -> QWidget:
        page, layout = self._page_shell("Методы и значения по умолчанию", "Выберите создаваемые схемы. Значения попадут в .txt-описание системы и будут начальными в CUDAynamics.")
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        body = QWidget(); body_l = QVBoxLayout(body)
        method_box = QGroupBox("Численные методы"); method_grid = QGridLayout(method_box)
        self.method_checks = {}
        for i, (key, label) in enumerate(METHODS.items()):
            check = QCheckBox(label if self.language == "ru" else METHODS_EN[key]); check.setChecked(key in ("ExplicitEuler", "ExplicitRungeKutta4"))
            check.toggled.connect(self._method_warning)
            check.toggled.connect(self._update_default_method_choices)
            method_grid.addWidget(check, i // 2, i % 2); self.method_checks[key] = check
        method_grid.addWidget(
            QLabel("Метод по умолчанию" if self.language == "ru" else "Default method"),
            (len(METHODS) + 1) // 2, 0,
        )
        self.default_method_combo = NoWheelComboBox()
        method_grid.addWidget(self.default_method_combo, (len(METHODS) + 1) // 2, 1)
        self._update_default_method_choices()
        body_l.addWidget(method_box)
        self.method_note = QLabel("", objectName="muted"); self.method_note.setWordWrap(True); body_l.addWidget(self.method_note)

        simulation = QGroupBox("Моделирование"); form = QFormLayout(simulation)
        self.step_value = self._number_edit("0.01", integer=False)
        self.simulation_time_value = self._number_edit("10.0", integer=False)
        self.transient_time_value = self._number_edit("0.0", integer=False)
        form.addRow("Шаг h", self.step_value)
        form.addRow("Время моделирования", self.simulation_time_value)
        form.addRow("Время переходного процесса", self.transient_time_value)
        self.time_steps_note = QLabel("", objectName="muted")
        self.time_steps_note.setWordWrap(True)
        form.addRow("", self.time_steps_note)
        for editor in (self.step_value, self.simulation_time_value, self.transient_time_value):
            editor.textChanged.connect(self._update_time_steps_note)
        self._update_time_steps_note()
        body_l.addWidget(simulation)

        body_l.addWidget(QLabel("Начальные значения переменных"))
        self.variable_table = QTableWidget(0, 6)
        self.variable_table.setHorizontalHeaderLabels(["Имя", "Роль", "Режим", "Минимум / значение", "Максимум", "Количество"])
        for column in (0, 1, 2):
            self.variable_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for column in (3, 4, 5):
            self.variable_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        body_l.addWidget(self.variable_table)
        body_l.addWidget(QLabel("Значения параметров"))
        self.parameter_table = QTableWidget(0, 5)
        self.parameter_table.setHorizontalHeaderLabels(["Имя", "Режим", "Минимум / значение", "Максимум", "Количество"])
        for column in (0, 1):
            self.parameter_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        for column in (2, 3, 4):
            self.parameter_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        body_l.addWidget(self.parameter_table)
        save = QPushButton("Применить настройки и подготовить изменения →", objectName="primary"); save.clicked.connect(self._prepare)
        body_l.addWidget(save)
        scroll.setWidget(body); layout.addWidget(scroll, 1)
        return page

    @staticmethod
    def _number_edit(value: str, integer: bool) -> QLineEdit:
        """Plain validated input: unlike a spin box it ignores wheel and arrow steps."""
        edit = QLineEdit(value)
        if integer:
            validator = QIntValidator(0, 100000000, edit)
        else:
            validator = QDoubleValidator(-1e100, 1e100, 16, edit)
            validator.setNotation(QDoubleValidator.ScientificNotation)
            validator.setLocale(QLocale.c())
        edit.setValidator(validator)
        return edit

    def _install_page(self) -> QWidget:
        page, layout = self._page_shell("Добавление в CUDAynamics", "Просмотрите точный diff. При установке исходные файлы регистрации сохраняются в .cudynamics-builder-backups.")
        path_row = QHBoxLayout()
        self.project_path = QLineEdit(str(locate_default_project(self.root)))
        browse = QPushButton("Папка…"); browse.clicked.connect(self._choose_project)
        preview = QPushButton("Обновить предпросмотр"); preview.clicked.connect(self._prepare)
        path_row.addWidget(QLabel("Проект")); path_row.addWidget(self.project_path, 1); path_row.addWidget(browse); path_row.addWidget(preview)
        layout.addLayout(path_row)
        self.diff_view = QTextEdit(); self.diff_view.setReadOnly(True); self.diff_view.setFont(QFont("Cascadia Mono", 9)); layout.addWidget(self.diff_view, 1)
        row = QHBoxLayout(); row.addStretch()
        self.install_button = QPushButton("Добавить систему в CUDAynamics", objectName="primary"); self.install_button.clicked.connect(self._install); self.install_button.setEnabled(False)
        row.addWidget(self.install_button); layout.addLayout(row)
        return page

    def _set_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for i, button in enumerate(self.nav): button.setChecked(i == index)

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Система уравнений" if self.language == "ru" else "Equation system", "",
            "Изображения и PDF (*.png *.jpg *.jpeg *.webp *.bmp *.pdf)" if self.language == "ru" else "Images and PDF (*.png *.jpg *.jpeg *.webp *.bmp *.pdf)",
        )
        if not path: return
        self._load_image(path)

    def _load_image(self, path: str) -> None:
        if Path(path).suffix.lower() == ".pdf":
            self.image_path = path
            self.image_preview.setPixmap(QPixmap())
            self.image_preview.setText(
                (f"Выбран PDF:\n{Path(path).name}\n\nФайл будет сохранён вместе с системой."
                 if self.language == "ru" else
                 f"Selected PDF:\n{Path(path).name}\n\nThe file will be saved with the system.")
            )
            self.statusBar().showMessage(Path(path).name)
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(
                self,
                "Не удалось открыть изображение" if self.language == "ru" else "Could not open image",
                "Выбранный файл не является поддерживаемым изображением." if self.language == "ru" else "The selected file is not a supported image.",
            )
            return
        self.image_path = path
        self.image_preview.setPixmap(
            pixmap.scaled(self.image_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.statusBar().showMessage(Path(path).name)

    def _update_api_key_visibility(self, backend: str) -> None:
        self.api_key.setVisible(backend != "pix2tex")

    def _recognize(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "Нет изображения" if self.language == "ru" else "No image", "Сначала выберите изображение." if self.language == "ru" else "Select an image first."); return
        if Path(self.image_path).suffix.lower() == ".pdf":
            QMessageBox.information(
                self,
                "PDF выбран как исходный документ" if self.language == "ru" else "PDF selected as source document",
                "PDF будет сохранён вместе с системой. Для OCR экспортируйте страницу с уравнениями в PNG, JPG или WebP." if self.language == "ru" else "The PDF will be saved with the system. For OCR, export the equation page as PNG, JPG, or WebP.",
            )
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Распознавание…" if self.language == "ru" else "Recognizing…")
        QApplication.processEvents()
        try:
            text = recognize_image(self.image_path, self.ocr_backend.currentText(), self.api_key.text())
            self.source_text.setPlainText(text)
            self.statusBar().showMessage("OCR завершён. Проверьте текст и нажмите «Разобрать»." if self.language == "ru" else "OCR complete. Review the text and click Parse.")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка OCR" if self.language == "ru" else "OCR error", runtime_text(self.language, str(exc)))
        finally:
            QApplication.restoreOverrideCursor()

    def _parse(self) -> None:
        try:
            self.spec = parse_system(self.source_text.toPlainText(), self.system_id.text(), self.display_name.text())
            self._populate_model(); self._set_page(1)
            if self.spec.rename_warnings:
                QMessageBox.warning(self, "Имена автоматически изменены" if self.language == "ru" else "Names changed automatically", runtime_text(self.language, "\n".join(self.spec.rename_warnings)))
            self.statusBar().showMessage("Уравнения разобраны; требуется проверка пользователя" if self.language == "ru" else "Equations parsed; user review required")
        except ParseError as exc:
            QMessageBox.critical(self, "Не удалось разобрать систему" if self.language == "ru" else "Could not parse system", runtime_text(self.language, str(exc)))

    def _add_equation_row(self, equation: Equation | None = None) -> None:
        row = self.eq_table.rowCount(); self.eq_table.insertRow(row)
        equation = equation or Equation("x", "0", EquationKind.STATE)
        self.eq_table.setCellWidget(row, 0, self._selection_checkbox())
        self.eq_table.setItem(row, 1, QTableWidgetItem(equation.target))
        combo = NoWheelComboBox()
        combo.addItem(self._t("Состояние (интегрировать)"), EquationKind.STATE.value)
        combo.addItem(self._t("Алгебраическая"), EquationKind.ALGEBRAIC.value)
        combo.addItem(self._t("Входной сигнал"), EquationKind.SIGNAL.value)
        combo.setCurrentIndex(max(0, combo.findData(equation.kind.value))); self.eq_table.setCellWidget(row, 2, combo)
        self.eq_table.setItem(row, 3, QTableWidgetItem(equation.expression))

    def _remove_equation_row(self) -> None:
        rows = self._checked_rows(self.eq_table)
        if not rows:
            self.statusBar().showMessage("Отметьте галочкой строки, которые нужно удалить" if self.language == "ru" else "Select the rows to delete using the checkboxes")
            return
        for row in reversed(rows):
            self.eq_table.removeRow(row)

    def _add_event_row(self, event: ConditionalEvent | None = None) -> None:
        row = self.event_table.rowCount(); self.event_table.insertRow(row)
        event = event or ConditionalEvent(
            "variable1 >= parameter1",
            {"variable1": "parameter2", "variable2": "parameter2"},
        )
        assignments = "; ".join(f"{target} = {expression}" for target, expression in event.assignments.items())
        self.event_table.setCellWidget(row, 0, self._selection_checkbox())
        position = NoWheelComboBox()
        position.addItem(self._t("В конце шага"), None)
        for equation_row in range(self.eq_table.rowCount()):
            item = self.eq_table.item(equation_row, 1)
            if item and item.text().strip():
                target = item.text().strip()
                position.addItem(f"После {target}" if self.language == "ru" else f"After {target}", target)
        selected = position.findData(event.after_target)
        position.setCurrentIndex(max(0, selected))
        self.event_table.setCellWidget(row, 1, position)
        self.event_table.setItem(row, 2, QTableWidgetItem(event.condition))
        self.event_table.setItem(row, 3, QTableWidgetItem(assignments))

    def _remove_event_row(self) -> None:
        rows = self._checked_rows(self.event_table)
        if not rows:
            self.statusBar().showMessage("Отметьте галочкой условия, которые нужно удалить" if self.language == "ru" else "Select the conditions to delete using the checkboxes")
            return
        for row in reversed(rows):
            self.event_table.removeRow(row)

    def _selection_checkbox(self) -> QCheckBox:
        checkbox = QCheckBox()
        checkbox.setToolTip(self._t("Отметить строку"))
        checkbox.setStyleSheet("QCheckBox { margin-left: 8px; }")
        return checkbox

    @staticmethod
    def _checked_rows(table: QTableWidget) -> list[int]:
        return [
            row for row in range(table.rowCount())
            if isinstance(table.cellWidget(row, 0), QCheckBox)
            and table.cellWidget(row, 0).isChecked()
        ]

    @staticmethod
    def _toggle_all_checks(table: QTableWidget) -> None:
        checkboxes = [
            table.cellWidget(row, 0) for row in range(table.rowCount())
            if isinstance(table.cellWidget(row, 0), QCheckBox)
        ]
        check_all = not checkboxes or not all(checkbox.isChecked() for checkbox in checkboxes)
        for checkbox in checkboxes:
            checkbox.setChecked(check_all)

    def _populate_model(self) -> None:
        self.eq_table.setRowCount(0)
        # Marker variables are an implementation detail of conditional events,
        # not user-authored equations and therefore have no selectable type.
        for equation in self.spec.equations:
            if equation.kind != EquationKind.MARKER:
                self._add_equation_row(equation)
        self.event_table.setRowCount(0)
        for event in self.spec.events: self._add_event_row(event)
        self.model_summary.setText(pretty_equations(self.spec, self.language))

    def _sync_model(self, quiet: bool = False) -> bool:
        if not self.spec: return False
        equations = []
        try:
            for row in range(self.eq_table.rowCount()):
                target = safe_identifier(self.eq_table.item(row, 1).text().strip())
                expression = self.eq_table.item(row, 3).text().strip()
                kind = EquationKind(self.eq_table.cellWidget(row, 2).currentData())
                equations.append(Equation(target, expression, kind))
            self.spec.equations = equations
            self.spec.events = []
            for row in range(self.event_table.rowCount()):
                condition = self.event_table.item(row, 2).text().strip()
                assignments = parse_assignments(self.event_table.item(row, 3).text())
                after_target = self.event_table.cellWidget(row, 1).currentData()
                self.spec.events.append(ConditionalEvent(condition, assignments, after_target))
            rename_warnings = normalize_reserved_identifiers(self.spec)
            model_rows_changed = bool(self.spec.identifier_renames) or len(self.spec.equations) != len(equations)
            problems = validate_spec(self.spec)
            if problems: raise ValueError("\n".join(problems))
            rename_summary = ", ".join(
                f"{old} → {new}" for old, new in self.spec.identifier_renames.items()
            ) or "—"
            self.model_summary.setText(self._model_summary_text(rename_summary))
            self._populate_defaults()
            if model_rows_changed:
                self._populate_model()
                self.model_summary.setText(self._model_summary_text(rename_summary))
            if not quiet:
                if rename_warnings:
                    QMessageBox.warning(self, "Имена автоматически изменены" if self.language == "ru" else "Names changed automatically", runtime_text(self.language, "\n".join(rename_warnings)))
                QMessageBox.information(self, "Модель корректна" if self.language == "ru" else "Model is valid", self.model_summary.text())
            return True
        except Exception as exc:
            if not quiet: QMessageBox.critical(self, "Ошибка модели" if self.language == "ru" else "Model error", runtime_text(self.language, str(exc)))
            return False

    def _model_summary_text(self, rename_summary: str) -> str:
        if self.language == "ru":
            return (
                f"Интегрируются: {', '.join(self.spec.states)}\n"
                f"Алгебраические: {', '.join(self.spec.algebraic) or '—'}\n"
                f"Сигнал: {', '.join(self.spec.signals) or '—'}\n"
                f"Созданы условиями: {', '.join(self.spec.markers) or '—'}\n"
                f"Условий: {len(self.spec.events)}\n"
                f"Параметры: {', '.join(self.spec.parameters) or '—'}\n"
                f"Автопереименования: {rename_summary}"
            )
        return (
            f"Integrated: {', '.join(self.spec.states)}\n"
            f"Algebraic: {', '.join(self.spec.algebraic) or '—'}\n"
            f"Signal: {', '.join(self.spec.signals) or '—'}\n"
            f"Created by conditions: {', '.join(self.spec.markers) or '—'}\n"
            f"Conditions: {len(self.spec.events)}\n"
            f"Parameters: {', '.join(self.spec.parameters) or '—'}\n"
            f"Automatic renames: {rename_summary}"
        )

    def _populate_defaults(self) -> None:
        if not self.spec: return
        self.variable_table.setRowCount(len(self.spec.variables))
        for row, name in enumerate(self.spec.variables):
            if name in self.spec.states:
                role = "интегрируется" if self.language == "ru" else "integrated"
            elif name in self.spec.signals:
                role = "сигнал" if self.language == "ru" else "signal"
            elif name in self.spec.markers:
                role = "задаётся условием" if self.language == "ru" else "set by condition"
            else:
                role = "вычисляется" if self.language == "ru" else "computed"
            defaults = self.spec.variable_defaults[name]
            self.variable_table.setItem(row, 0, QTableWidgetItem(name))
            self.variable_table.setItem(row, 1, QTableWidgetItem(role))
            self.variable_table.setCellWidget(row, 2, self._range_combo(defaults.range_kind))
            self.variable_table.setItem(row, 3, QTableWidgetItem(str(defaults.value)))
            self.variable_table.setItem(row, 4, QTableWidgetItem(str(defaults.maximum)))
            self.variable_table.setItem(row, 5, QTableWidgetItem(str(defaults.count)))
        self.parameter_table.setRowCount(len(self.spec.parameters))
        for row, name in enumerate(self.spec.parameters):
            defaults = self.spec.parameter_defaults[name]
            self.parameter_table.setItem(row, 0, QTableWidgetItem(name))
            self.parameter_table.setCellWidget(row, 1, self._range_combo(defaults.range_kind))
            self.parameter_table.setItem(row, 2, QTableWidgetItem(str(defaults.value)))
            self.parameter_table.setItem(row, 3, QTableWidgetItem(str(defaults.maximum)))
            self.parameter_table.setItem(row, 4, QTableWidgetItem(str(defaults.count)))

    def _range_combo(self, range_kind: str) -> QComboBox:
        combo = NoWheelComboBox()
        combo.addItem(self._t("Fixed (одно значение)"), "Fixed")
        combo.addItem(self._t("Count (диапазон)"), "Linear")
        combo.setCurrentIndex(0 if range_kind == "Fixed" else 1)
        combo.setToolTip(self._t("Fixed — одно значение. Count — указанное число равномерных значений от минимума до максимума."))
        return combo

    def _method_warning(self) -> None:
        notes = []
        if self.method_checks["VariableSymmetryCD"].isChecked(): notes.append("VSCD зависит от порядка строк состояний; линейные неявности решаются формулой, нелинейные — скалярным Ньютоном с аналитической производной." if self.language == "ru" else "VSCD depends on state row order; affine implicit equations are solved exactly and nonlinear ones use scalar Newton with an analytic derivative.")
        if self.method_checks["ExplicitDormandPrince8"].isChecked(): notes.append("DP8 генерируется с фиксированным шагом, как в текущей библиотеке, без адаптивного контроля ошибки." if self.language == "ru" else "DP8 uses the library's fixed-step implementation without adaptive error control.")
        if self.method_checks["ImplicitEuler"].isChecked() or self.method_checks["ImplicitMidpoint"].isChecked(): notes.append("Неявные методы используют до 8 итераций Ньютона и аналитический Якобиан, вычисленный при генерации." if self.language == "ru" else "Implicit methods use up to eight Newton iterations and an analytic Jacobian computed during generation.")
        self.method_note.setText(" ".join(notes))

    def _update_default_method_choices(self) -> None:
        if not hasattr(self, "default_method_combo"):
            return
        previous = self.default_method_combo.currentData()
        preferred = previous or (self.spec.default_method if self.spec else None)
        self.default_method_combo.blockSignals(True)
        self.default_method_combo.clear()
        for key in METHODS:
            if self.method_checks[key].isChecked():
                label = METHODS[key] if self.language == "ru" else METHODS_EN[key]
                self.default_method_combo.addItem(label, key)
        index = self.default_method_combo.findData(preferred)
        self.default_method_combo.setCurrentIndex(index if index >= 0 else 0)
        self.default_method_combo.blockSignals(False)

    def _read_settings(self) -> None:
        self.spec.methods = [key for key, check in self.method_checks.items() if check.isChecked()]
        self.spec.default_method = self.default_method_combo.currentData()
        self.spec.step = float(self.step_value.text().replace(",", "."))
        simulation_time = float(self.simulation_time_value.text().replace(",", "."))
        transient_time = float(self.transient_time_value.text().replace(",", "."))
        self.spec.steps = time_to_steps(simulation_time, self.spec.step)
        self.spec.transient = time_to_steps(transient_time, self.spec.step, allow_zero=True)
        for row, name in enumerate(self.spec.variables):
            defaults = self.spec.variable_defaults[name]
            defaults.range_kind = self.variable_table.cellWidget(row, 2).currentData()
            defaults.value = float(self.variable_table.item(row, 3).text().replace(",", "."))
            defaults.minimum = defaults.value
            defaults.maximum = float(self.variable_table.item(row, 4).text().replace(",", "."))
            defaults.count = int(self.variable_table.item(row, 5).text())
            self._validate_range(name, defaults)
        for row, name in enumerate(self.spec.parameters):
            defaults = self.spec.parameter_defaults[name]
            defaults.range_kind = self.parameter_table.cellWidget(row, 1).currentData()
            defaults.value = float(self.parameter_table.item(row, 2).text().replace(",", "."))
            defaults.minimum = defaults.value
            defaults.maximum = float(self.parameter_table.item(row, 3).text().replace(",", "."))
            defaults.count = int(self.parameter_table.item(row, 4).text())
            self._validate_range(name, defaults)

    def _update_time_steps_note(self) -> None:
        try:
            step = float(self.step_value.text().replace(",", "."))
            duration = float(self.simulation_time_value.text().replace(",", "."))
            transient = float(self.transient_time_value.text().replace(",", "."))
            steps = time_to_steps(duration, step)
            transient_steps = time_to_steps(transient, step, allow_zero=True)
            self.time_steps_note.setText(
                (f"Будет записано: шагов моделирования — {steps}, переходных шагов — {transient_steps}."
                 if self.language == "ru" else
                 f"Generated values: simulation steps — {steps}, transient steps — {transient_steps}.")
            )
        except (TypeError, ValueError):
            self.time_steps_note.setText("Введите положительный шаг и корректное время." if self.language == "ru" else "Enter a positive step and valid times.")

    def _validate_range(self, name: str, defaults: AttributeDefaults) -> None:
        if defaults.range_kind == "Linear":
            if defaults.count < 2:
                raise ValueError((f"Для Count-атрибута {name!r} количество должно быть не меньше 2" if self.language == "ru" else f"Count attribute {name!r} must contain at least 2 values"))
            if defaults.maximum < defaults.value:
                raise ValueError((f"Для Count-атрибута {name!r} максимум не может быть меньше минимума" if self.language == "ru" else f"Count attribute {name!r} cannot have a maximum below its minimum"))

    def _prepare(self) -> None:
        try:
            if self.spec and self.variable_table.rowCount() == len(self.spec.variables):
                self._read_settings()
            if not self._sync_model(quiet=True): raise ValueError("Сначала исправьте модель на шаге 2" if self.language == "ru" else "Fix the model on step 2 first")
            problems = validate_spec(self.spec)
            if problems: raise ValueError("\n".join(problems))
            identity_warnings = ensure_unique_system_identity(
                self.project_path.text(), self.spec, self.language
            )
            if identity_warnings:
                self.system_id.setText(self.spec.system_id)
                self.display_name.setText(self.spec.display_name)
                QMessageBox.warning(
                    self,
                    "ID или название уже заняты" if self.language == "ru" else "ID or name already in use",
                    "\n".join(identity_warnings),
                )
            generated = generate(self.spec)
            self.plan = build_plan(
                self.project_path.text(), self.spec, generated,
                self.image_path or None,
            )
            self.diff_view.setPlainText(self.plan.diff)
            self.install_button.setEnabled(True); self._set_page(3)
            self.statusBar().showMessage("План готов; CUDAynamics ещё не изменён" if self.language == "ru" else "Plan ready; CUDAynamics has not been modified")
        except Exception as exc:
            self.plan = None; self.install_button.setEnabled(False)
            QMessageBox.critical(self, "Нельзя подготовить систему" if self.language == "ru" else "Could not prepare system", runtime_text(self.language, str(exc)))

    def _choose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Корень cudaynamics-master" if self.language == "ru" else "cudaynamics-master root", self.project_path.text())
        if path: self.project_path.setText(path)

    def _install(self) -> None:
        if not self.plan: return
        created_count = 3 + int(bool(self.image_path))
        question = (f"Добавить систему {self.spec.system_id!r} в проект?\nБудут созданы {created_count} файла и изменены 4 файла регистрации." if self.language == "ru" else f"Add system {self.spec.system_id!r} to the project?\n{created_count} files will be created and 4 registration files modified.")
        answer = QMessageBox.question(self, "Подтверждение" if self.language == "ru" else "Confirmation", question)
        if answer != QMessageBox.Yes: return
        try:
            backup = apply_plan(self.project_path.text(), self.plan)
            self.install_button.setEnabled(False)
            QMessageBox.information(self, "Готово" if self.language == "ru" else "Done", (f"Система добавлена.\nРезервная копия: {backup}\n\nТеперь пересоберите CUDAynamics в Visual Studio." if self.language == "ru" else f"System added.\nBackup: {backup}\n\nRebuild CUDAynamics in Visual Studio."))
            self.statusBar().showMessage("Система успешно добавлена" if self.language == "ru" else "System added successfully")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка установки" if self.language == "ru" else "Installation error", runtime_text(self.language, str(exc)))


class EditWindow(QMainWindow):
    def __init__(self, root: Path, language: str, return_to_menu) -> None:
        super().__init__()
        self.root = root
        self.language = language
        self.return_to_menu = return_to_menu
        self.model = None
        self.systems = []
        self.setWindowTitle("CUDAynamics · " + ("Редактор систем" if language == "ru" else "System Editor"))
        self.resize(1180, 780)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._refresh_systems()

    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(28, 24, 28, 24)
        self.steps = QStackedWidget()
        self.steps.addWidget(self._identity_page())
        self.steps.addWidget(self._values_page())
        outer.addWidget(self.steps)
        self.setCentralWidget(central)

    def _identity_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Шаг 1 · Имена системы и атрибутов" if self.language == "ru" else "Step 1 · System and attribute names", objectName="section"))
        hint = QLabel(
            "Выберите систему и измените только идентификаторы. Структура уравнений останется неизменной."
            if self.language == "ru" else
            "Choose a system and edit identifiers only. Equation structure will remain unchanged.",
            objectName="muted",
        )
        layout.addWidget(hint)
        project_row = QHBoxLayout()
        project_row.addWidget(QLabel("Проект" if self.language == "ru" else "Project"))
        self.edit_project_path = QLineEdit(str(locate_default_project(self.root)))
        browse = QPushButton("Папка…" if self.language == "ru" else "Browse…")
        browse.clicked.connect(self._choose_edit_project)
        refresh = QPushButton("Обновить" if self.language == "ru" else "Refresh")
        refresh.clicked.connect(self._refresh_systems)
        project_row.addWidget(self.edit_project_path, 1)
        project_row.addWidget(browse)
        project_row.addWidget(refresh)
        layout.addLayout(project_row)
        system_row = QHBoxLayout()
        system_row.addWidget(QLabel("Система" if self.language == "ru" else "System"))
        self.edit_system_combo = NoWheelComboBox()
        self.edit_system_combo.currentIndexChanged.connect(self._load_selected_system)
        system_row.addWidget(self.edit_system_combo, 1)
        layout.addLayout(system_row)
        form = QFormLayout()
        self.edit_id = QLineEdit()
        self.edit_name = QLineEdit()
        form.addRow("ID системы" if self.language == "ru" else "System ID", self.edit_id)
        form.addRow("Название" if self.language == "ru" else "Name", self.edit_name)
        layout.addLayout(form)
        self.rename_table = QTableWidget(0, 3)
        self.rename_table.setHorizontalHeaderLabels([
            "Роль" if self.language == "ru" else "Role",
            "Текущее имя" if self.language == "ru" else "Current name",
            "Новое имя" if self.language == "ru" else "New name",
        ])
        self.rename_table.verticalHeader().setVisible(False)
        self.rename_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.rename_table, 1)
        row = QHBoxLayout()
        back = QPushButton("← Вернуться в меню" if self.language == "ru" else "← Back to menu")
        back.clicked.connect(self._back_to_menu)
        next_button = QPushButton("Значения атрибутов →" if self.language == "ru" else "Attribute values →", objectName="primary")
        next_button.clicked.connect(self._go_to_values)
        row.addWidget(back)
        row.addStretch()
        row.addWidget(next_button)
        layout.addLayout(row)
        return page

    def _values_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Шаг 2 · Значения атрибутов" if self.language == "ru" else "Step 2 · Attribute values", objectName="section"))
        hint = QLabel(
            "Изменения применяются к строкам var/param файла конфигурации системы."
            if self.language == "ru" else
            "Changes are applied to var/param rows in the system configuration file.",
            objectName="muted",
        )
        layout.addWidget(hint)
        default_row = QHBoxLayout()
        default_row.addWidget(QLabel("Метод по умолчанию" if self.language == "ru" else "Default method"))
        self.edit_default_method_combo = NoWheelComboBox()
        default_row.addWidget(self.edit_default_method_combo, 1)
        layout.addLayout(default_row)
        self.config_table = QTableWidget(0, 9)
        self.config_table.setHorizontalHeaderLabels([
            "Тип" if self.language == "ru" else "Type",
            "Имя" if self.language == "ru" else "Name",
            "Режим" if self.language == "ru" else "Mode",
            "Минимум / значение" if self.language == "ru" else "Minimum / value",
            "Максимум" if self.language == "ru" else "Maximum",
            "Шаг" if self.language == "ru" else "Step",
            "Количество" if self.language == "ru" else "Count",
            "Среднее" if self.language == "ru" else "Mean",
            "Отклонение" if self.language == "ru" else "Deviation",
        ])
        self.config_table.verticalHeader().setVisible(False)
        self.config_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.config_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for column in range(2, 9):
            self.config_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        layout.addWidget(self.config_table, 1)
        row = QHBoxLayout()
        previous = QPushButton("← Имена" if self.language == "ru" else "← Names")
        previous.clicked.connect(lambda: self.steps.setCurrentIndex(0))
        apply_button = QPushButton("Применить изменения" if self.language == "ru" else "Apply changes", objectName="primary")
        apply_button.clicked.connect(self._apply_edit)
        row.addWidget(previous)
        row.addStretch()
        row.addWidget(apply_button)
        layout.addLayout(row)
        return page

    def _choose_edit_project(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Корень cudaynamics-master" if self.language == "ru" else "cudaynamics-master root",
            self.edit_project_path.text(),
        )
        if path:
            self.edit_project_path.setText(path)
            self._refresh_systems()

    def _refresh_systems(self) -> None:
        try:
            self.systems = list_library_systems(self.edit_project_path.text())
            self.edit_system_combo.blockSignals(True)
            self.edit_system_combo.clear()
            for system in self.systems:
                self.edit_system_combo.addItem(f"{system.display_name}  [{system.system_id}]", system.system_id)
            self.edit_system_combo.blockSignals(False)
            if self.systems:
                self.edit_system_combo.setCurrentIndex(0)
                self._load_selected_system()
            else:
                self.model = None
                self.rename_table.setRowCount(0)
        except Exception as exc:
            self.model = None
            self.edit_system_combo.clear()
            QMessageBox.critical(self, "Ошибка проекта" if self.language == "ru" else "Project error", runtime_text(self.language, str(exc)))

    def _load_selected_system(self) -> None:
        system_id = self.edit_system_combo.currentData()
        if not system_id:
            return
        try:
            self.model = load_editable_system(self.edit_project_path.text(), system_id)
            self.edit_id.setText(self.model.system_id)
            self.edit_name.setText(self.model.display_name)
            attributes = [*self.model.variables, *self.model.parameters]
            self.rename_table.setRowCount(len(attributes))
            for row, attribute in enumerate(attributes):
                role = ("Переменная" if attribute.role == "var" else "Параметр") if self.language == "ru" else ("Variable" if attribute.role == "var" else "Parameter")
                role_item = QTableWidgetItem(role)
                role_item.setFlags(role_item.flags() & ~Qt.ItemIsEditable)
                old_item = QTableWidgetItem(attribute.original)
                old_item.setFlags(old_item.flags() & ~Qt.ItemIsEditable)
                self.rename_table.setItem(row, 0, role_item)
                self.rename_table.setItem(row, 1, old_item)
                self.rename_table.setItem(row, 2, QTableWidgetItem(attribute.edited))
            self.steps.setCurrentIndex(0)
        except Exception as exc:
            self.model = None
            QMessageBox.critical(self, "Ошибка системы" if self.language == "ru" else "System error", runtime_text(self.language, str(exc)))

    def _read_identity(self) -> None:
        if self.model is None:
            raise ValueError("Выберите систему" if self.language == "ru" else "Select a system")
        self.model.system_id = self.edit_id.text().strip()
        self.model.display_name = self.edit_name.text().strip()
        attributes = [*self.model.variables, *self.model.parameters]
        for row, attribute in enumerate(attributes):
            attribute.edited = self.rename_table.item(row, 2).text().strip()
        problems = validate_renames(self.model)
        if not self.model.display_name:
            problems.append("Название не может быть пустым" if self.language == "ru" else "Name cannot be empty")
        if problems:
            raise ValueError("\n".join(runtime_text(self.language, problem) for problem in problems))

    def _go_to_values(self) -> None:
        try:
            self._read_identity()
            messages = make_edit_identity_unique(self.model)
            if messages:
                self.edit_id.setText(self.model.system_id)
                self.edit_name.setText(self.model.display_name)
                QMessageBox.warning(
                    self, "Имена изменены" if self.language == "ru" else "Names changed",
                    runtime_text(self.language, "\n".join(messages)),
                )
            mapping = {
                item.original: item.edited for item in [*self.model.variables, *self.model.parameters]
                if item.original != item.edited
            }
            self.edit_default_method_combo.clear()
            if self.model.methods:
                for method in self.model.methods:
                    label = METHODS.get(method, method) if self.language == "ru" else METHODS_EN.get(method, method)
                    self.edit_default_method_combo.addItem(label, method)
                selected_method = self.edit_default_method_combo.findData(self.model.default_method)
                self.edit_default_method_combo.setCurrentIndex(max(0, selected_method))
                self.edit_default_method_combo.setEnabled(True)
                self.edit_default_method_combo.setToolTip("")
            else:
                unavailable = "Не задан в конфигурации" if self.language == "ru" else "Not specified in configuration"
                self.edit_default_method_combo.addItem(unavailable, None)
                self.edit_default_method_combo.setEnabled(False)
                self.edit_default_method_combo.setToolTip(unavailable)
            self.config_table.setRowCount(len(self.model.config_attributes))
            for row, attribute in enumerate(self.model.config_attributes):
                role = ("Переменная" if attribute.role == "var" else "Параметр") if self.language == "ru" else ("Variable" if attribute.role == "var" else "Parameter")
                role_item = QTableWidgetItem(role)
                role_item.setFlags(role_item.flags() & ~Qt.ItemIsEditable)
                name_item = QTableWidgetItem(mapping.get(attribute.source_name, attribute.config_name))
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.config_table.setItem(row, 0, role_item)
                self.config_table.setItem(row, 1, name_item)
                mode = NoWheelComboBox()
                mode.addItems(RANGE_KINDS)
                mode.setCurrentText(attribute.range_kind)
                self.config_table.setCellWidget(row, 2, mode)
                for column, value in enumerate((attribute.minimum, attribute.maximum, attribute.step, attribute.count, attribute.mean, attribute.deviation), 3):
                    self.config_table.setItem(row, column, QTableWidgetItem(value))
            self.steps.setCurrentIndex(1)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка данных" if self.language == "ru" else "Invalid data", runtime_text(self.language, str(exc)))

    def _read_config(self) -> None:
        if self.model.methods:
            self.model.default_method = self.edit_default_method_combo.currentData()
        for row, attribute in enumerate(self.model.config_attributes):
            attribute.range_kind = self.config_table.cellWidget(row, 2).currentText()
            values = [self.config_table.item(row, column).text().strip() for column in range(3, 9)]
            for value in values:
                float(value.replace(",", "."))
            count = int(values[3])
            if count < 0:
                raise ValueError("Количество не может быть отрицательным" if self.language == "ru" else "Count cannot be negative")
            normalized = [value.replace(",", ".") for value in values]
            attribute.minimum, attribute.maximum, attribute.step, attribute.count, attribute.mean, attribute.deviation = normalized

    def _apply_edit(self) -> None:
        try:
            self._read_config()
            question = (
                f"Применить изменения к системе {self.model.original_id!r}? Будет создана резервная копия."
                if self.language == "ru" else
                f"Apply changes to system {self.model.original_id!r}? A backup will be created."
            )
            if QMessageBox.question(self, "Подтверждение" if self.language == "ru" else "Confirmation", question) != QMessageBox.Yes:
                return
            plan = build_system_edit_plan(self.model)
            backup = apply_system_edit_plan(plan)
            QMessageBox.information(
                self, "Готово" if self.language == "ru" else "Done",
                (f"Система обновлена.\nРезервная копия: {backup}" if self.language == "ru" else f"System updated.\nBackup: {backup}"),
            )
            self._refresh_systems()
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка редактирования" if self.language == "ru" else "Editing error", runtime_text(self.language, str(exc)))

    def _back_to_menu(self) -> None:
        callback = self.return_to_menu
        self.return_to_menu = None
        callback()
        self.close()


class DeleteWindow(QMainWindow):
    def __init__(self, root: Path, language: str, return_to_menu) -> None:
        super().__init__()
        self.root = root
        self.language = language
        self.return_to_menu = return_to_menu
        self.setWindowTitle("CUDAynamics · " + ("Удаление систем" if language == "ru" else "Delete systems"))
        self.resize(980, 700)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Удаление систем" if self.language == "ru" else "Delete systems", objectName="section")
        hint = QLabel(
            "Выберите проект CUDAynamics и отметьте системы для удаления. Перед удалением будет создана резервная копия."
            if self.language == "ru" else
            "Choose a CUDAynamics project and select systems to remove. A backup is created before deletion.",
            objectName="muted",
        )
        hint.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(hint)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Проект" if self.language == "ru" else "Project"))
        self.project_path = QLineEdit(str(locate_default_project(self.root)))
        path_row.addWidget(self.project_path, 1)
        browse = QPushButton("Папка…" if self.language == "ru" else "Browse…")
        browse.clicked.connect(self._choose_project)
        refresh = QPushButton("Обновить список" if self.language == "ru" else "Refresh list")
        refresh.clicked.connect(self._refresh)
        path_row.addWidget(browse)
        path_row.addWidget(refresh)
        layout.addLayout(path_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([
            "✓", "ID", "Название" if self.language == "ru" else "Name",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 38)
        self.table.horizontalHeader().sectionClicked.connect(
            lambda section: MainWindow._toggle_all_checks(self.table) if section == 0 else None
        )
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        back = QPushButton("← Вернуться в меню" if self.language == "ru" else "← Back to menu")
        back.clicked.connect(self._back)
        delete = QPushButton("Удалить выбранные системы" if self.language == "ru" else "Delete selected systems", objectName="primary")
        delete.clicked.connect(self._delete_selected)
        buttons.addWidget(back)
        buttons.addStretch()
        buttons.addWidget(delete)
        layout.addLayout(buttons)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Готово" if self.language == "ru" else "Ready")

    def _choose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Корень cudaynamics-master" if self.language == "ru" else "cudaynamics-master root",
            self.project_path.text(),
        )
        if path:
            self.project_path.setText(path)
            self._refresh()

    def _refresh(self) -> None:
        try:
            systems = list_library_systems(self.project_path.text())
            self.table.setRowCount(0)
            for system in systems:
                row = self.table.rowCount()
                self.table.insertRow(row)
                checkbox = QCheckBox()
                checkbox.setToolTip("Отметить систему" if self.language == "ru" else "Select system")
                checkbox.setStyleSheet("QCheckBox { margin-left: 8px; }")
                self.table.setCellWidget(row, 0, checkbox)
                id_item = QTableWidgetItem(system.system_id)
                id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
                name_item = QTableWidgetItem(system.display_name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 1, id_item)
                self.table.setItem(row, 2, name_item)
            self.statusBar().showMessage(
                (f"Найдено систем: {len(systems)}" if self.language == "ru" else f"Systems found: {len(systems)}")
            )
        except Exception as exc:
            self.table.setRowCount(0)
            QMessageBox.critical(
                self, "Ошибка проекта" if self.language == "ru" else "Project error", runtime_text(self.language, str(exc))
            )

    def _selected_ids(self) -> list[str]:
        return [
            self.table.item(row, 1).text()
            for row in range(self.table.rowCount())
            if isinstance(self.table.cellWidget(row, 0), QCheckBox)
            and self.table.cellWidget(row, 0).isChecked()
        ]

    def _delete_selected(self) -> None:
        selected = self._selected_ids()
        if not selected:
            QMessageBox.information(
                self,
                "Нет выбранных систем" if self.language == "ru" else "No systems selected",
                "Отметьте системы галочками." if self.language == "ru" else "Select systems using the checkboxes.",
            )
            return
        names = ", ".join(selected)
        question = (
            f"Удалить системы: {names}?\nПапки и регистрационные файлы будут сохранены в резервной копии."
            if self.language == "ru" else
            f"Delete systems: {names}?\nDirectories and registration files will be backed up."
        )
        if QMessageBox.question(
            self, "Подтверждение удаления" if self.language == "ru" else "Confirm deletion", question
        ) != QMessageBox.Yes:
            return
        try:
            plan = build_removal_plan(self.project_path.text(), selected)
            backup = apply_removal_plan(self.project_path.text(), plan)
            QMessageBox.information(
                self,
                "Готово" if self.language == "ru" else "Done",
                (f"Системы удалены.\nРезервная копия: {backup}" if self.language == "ru" else f"Systems deleted.\nBackup: {backup}"),
            )
            self._refresh()
        except Exception as exc:
            QMessageBox.critical(
                self, "Ошибка удаления" if self.language == "ru" else "Deletion error", runtime_text(self.language, str(exc))
            )

    def _back(self) -> None:
        callback = self.return_to_menu
        self.return_to_menu = None
        callback()
        self.close()


class EditorMenu(QMainWindow):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.language = "en"
        self.child = None
        self.setWindowTitle("CUDAynamics System Editor")
        self.resize(720, 480)
        self.setStyleSheet(STYLE)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(64, 52, 64, 36)
        self.menu_title = QLabel("CUDAynamics System Editor", objectName="brand")
        self.menu_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.menu_title)
        layout.addStretch()
        self.create_button = QPushButton("Create system", objectName="primary")
        self.create_button.setMinimumHeight(54)
        self.create_button.clicked.connect(self._open_creator)
        self.edit_button = QPushButton("Edit System")
        self.edit_button.setMinimumHeight(54)
        self.edit_button.clicked.connect(self._open_edit)
        self.delete_button = QPushButton("Delete systems")
        self.delete_button.setMinimumHeight(54)
        self.delete_button.clicked.connect(self._open_delete)
        layout.addWidget(self.create_button)
        layout.addSpacing(14)
        layout.addWidget(self.edit_button)
        layout.addSpacing(14)
        layout.addWidget(self.delete_button)
        layout.addStretch()
        language_row = QHBoxLayout()
        language_row.addStretch()
        self.language_label = QLabel("Language")
        self.language_combo = NoWheelComboBox()
        self.language_combo.addItem("English", "en")
        self.language_combo.addItem("Русский", "ru")
        self.language_combo.currentIndexChanged.connect(self._language_changed)
        language_row.addWidget(self.language_label)
        language_row.addWidget(self.language_combo)
        layout.addLayout(language_row)
        self.setCentralWidget(central)

    def _language_changed(self) -> None:
        self.language = self.language_combo.currentData()
        if self.language == "ru":
            self.create_button.setText("Создать систему")
            self.edit_button.setText("Редактировать систему")
            self.delete_button.setText("Удалить системы")
            self.language_label.setText("Язык")
        else:
            self.create_button.setText("Create system")
            self.edit_button.setText("Edit System")
            self.delete_button.setText("Delete systems")
            self.language_label.setText("Language")

    def _open_creator(self) -> None:
        self.child = MainWindow(self.root, self.language, self._return_from_child)
        self.hide()
        self.child.show()

    def _open_delete(self) -> None:
        self.child = DeleteWindow(self.root, self.language, self._return_from_child)
        self.hide()
        self.child.show()

    def _open_edit(self) -> None:
        self.child = EditWindow(self.root, self.language, self._return_from_child)
        self.hide()
        self.child.show()

    def _return_from_child(self) -> None:
        self.child = None
        self.show()
        self.raise_()
        self.activateWindow()


def run() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("CUDAynamics System Editor")
    window = EditorMenu(Path(__file__).resolve().parents[1])
    window.show()
    return app.exec()
