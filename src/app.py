from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)










import sys
from pathlib import Path

import pandas as pd

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QMessageBox,
    QFrame,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QDialog,
)

from PySide6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from predict import predict_fault



FAULT_NAMES = {
    "局部放电": "Partial Discharge",
    "低能放电": "Low-Energy Discharge",
    "高能放电": "High-Energy Discharge",
    "低温过热": "Low-Temperature Overheating",
    "中温过热": "Medium-Temperature Overheating",
    "高温过热": "High-Temperature Overheating",
}


GAS_COLUMNS = [
    "H2",
    "CH4",
    "C2H6",
    "C2H4",
    "C2H2",
]


class ProbabilityChart(FigureCanvasQTAgg):

    def __init__(self, parent=None):

        self.figure = Figure(figsize=(8, 5.5))

        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.setParent(parent)

    def update_chart(self, probabilities):

        self.ax.clear()

        sorted_probs = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        names = [
            FAULT_NAMES.get(name, name)
            for name, _ in sorted_probs
        ]

        values = [
            probability * 100
            for _, probability in sorted_probs
        ]

        names = names[::-1]
        values = values[::-1]

        self.ax.barh(
            names,
            values
        )

        self.ax.set_xlabel(
            "Predicted Probability (%)"
        )

        self.ax.set_title(
            "Fault Probability"
        )

        self.ax.set_xlim(
            0,
            100
        )

        self.figure.subplots_adjust(
        left=0.38,
        right=0.95,
        top=0.90,
        bottom=0.15
    )

        self.draw()




class DGAGasChart(FigureCanvasQTAgg):

    def __init__(self, parent=None):

        self.figure = Figure(figsize=(8, 4))
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)

        self.setParent(parent)

    def update_chart(self, values):

        self.ax.clear()

        gases = list(values.keys())
        concentrations = list(values.values())

        # Avoid log(0)
        concentrations = [
            max(value, 0.0001)
            for value in concentrations
        ]

        self.ax.bar(
            gases,
            concentrations
        )

        self.ax.set_yscale("log")

        self.ax.set_ylabel(
            "Concentration (ppm)"
        )

        self.ax.set_xlabel(
            "Dissolved Gas"
        )

        self.ax.set_title(
            "DGA Gas Concentrations"
        )

        self.figure.subplots_adjust(
    left=0.12,
    right=0.97,
    top=0.88,
    bottom=0.18
)

        self.draw()






class BatchResultsDialog(QDialog):

    def __init__(self, results, parent=None):

        super().__init__(parent)

        self.results = results

        self.setWindowTitle(
            "Batch Diagnosis Results"
        )

        self.resize(
            900,
            550
        )

        layout = QVBoxLayout()

        title = QLabel(
            "Batch Diagnosis Results"
        )

        title.setStyleSheet(
            """
            font-size: 22px;
            font-weight: bold;
            padding: 10px;
            """
        )

        layout.addWidget(title)

        # ==============================================
        # RESULTS TABLE
        # ==============================================

        self.table = QTableWidget()

        self.table.setColumnCount(3)

        self.table.setHorizontalHeaderLabels(
            [
                "Sample",
                "Predicted Fault",
                "Model Probability"
            ]
        )

        self.table.setRowCount(
            len(results)
        )

        for row, result in enumerate(results):

            self.table.setItem(
                row,
                0,
                QTableWidgetItem(
                    str(result["sample"])
                )
            )

            self.table.setItem(
                row,
                1,
                QTableWidgetItem(
                    result["fault"]
                )
            )

            self.table.setItem(
                row,
                2,
                QTableWidgetItem(
                    f'{result["probability"]:.2f}%'
                )
            )

        self.table.resizeColumnsToContents()

        layout.addWidget(
            self.table
        )

        # ==============================================
        # BUTTONS
        # ==============================================

        button_layout = QHBoxLayout()

        export_csv = QPushButton(
            "EXPORT CSV"
        )

        export_excel = QPushButton(
            "EXPORT EXCEL"
        )

        close_button = QPushButton(
            "Close"
        )

        export_csv.clicked.connect(
            self.export_csv
        )

        export_excel.clicked.connect(
            self.export_excel
        )

        close_button.clicked.connect(
            self.accept
        )

        button_layout.addWidget(
            export_csv
        )

        button_layout.addWidget(
            export_excel
        )

        button_layout.addStretch()

        button_layout.addWidget(
            close_button
        )

        layout.addLayout(
            button_layout
        )

        self.setLayout(
            layout
        )

    # ==============================================
    # EXPORT CSV
    # ==============================================

    def export_csv(self):

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Diagnosis Results",
            "transformer_diagnosis.csv",
            "CSV Files (*.csv)"
        )

        if not file_path:

            return

        try:

            df = pd.DataFrame(
                self.results
            )

            df = df.rename(
                columns={
                    "sample": "Sample",
                    "fault": "Predicted Fault",
                    "probability":
                        "Model Probability (%)"
                }
            )

            df.to_csv(
                file_path,
                index=False
            )

            QMessageBox.information(
                self,
                "Export Successful",
                f"Results saved to:\n\n{file_path}"
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Export Error",
                str(error)
            )

    # ==============================================
    # EXPORT EXCEL
    # ==============================================

    def export_excel(self):

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Diagnosis Results",
            "transformer_diagnosis.xlsx",
            "Excel Files (*.xlsx)"
        )

        if not file_path:

            return

        try:

            df = pd.DataFrame(
                self.results
            )

            df = df.rename(
                columns={
                    "sample": "Sample",
                    "fault": "Predicted Fault",
                    "probability":
                        "Model Probability (%)"
                }
            )

            df.to_excel(
                file_path,
                index=False
            )

            QMessageBox.information(
                self,
                "Export Successful",
                f"Results saved to:\n\n{file_path}"
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Export Error",
                str(error)
            )






class AnalysisWindow(QDialog):

    def __init__(self, probabilities, gas_values, parent=None):

        super().__init__(parent)

        self.setWindowFlags(
    Qt.Window
    | Qt.WindowMinimizeButtonHint
    | Qt.WindowMaximizeButtonHint
    | Qt.WindowCloseButtonHint
)

        self.setWindowTitle(
            "DGA Analysis"
        )

        self.resize(
            1000,
            850
        )

        self.setMinimumSize(
            850,
            700
        )

        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI";
                font-size: 14px;
                color: #1f2937;
            }

            QLabel.section-title {
                font-size: 17px;
                font-weight: bold;
                padding: 6px;
                color: #1f2937;
            }

            QPushButton {
                border: none;
                border-radius: 7px;
                padding: 10px 20px;
                font-weight: bold;
                background: #4a90e2;
                color: white;
            }

            QPushButton:hover {
                background: #357abd;
            }
        """)

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            20, 20, 20, 20
        )

        layout.setSpacing(10)

        # ==================================================
        # TITLE
        # ==================================================

        title = QLabel(
            "DISSOLVED GAS ANALYSIS"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            padding: 10px;
            color: #1f2937;
        """)

        layout.addWidget(title)

        # ==================================================
        # PROBABILITY SECTION
        # ==================================================

        probability_label = QLabel(
            "Fault Probability"
        )

        probability_label.setProperty(
            "class",
            "section-title"
        )

        layout.addWidget(
            probability_label
        )

        self.probability_chart = ProbabilityChart()

        self.probability_chart.setMinimumHeight(
            300
        )

        self.probability_chart.update_chart(
            probabilities
        )

        layout.addWidget(
            self.probability_chart,
            1
        )

        # ==================================================
        # GAS SECTION
        # ==================================================

        gas_label = QLabel(
            "DGA Gas Concentrations"
        )

        gas_label.setProperty(
            "class",
            "section-title"
        )

        layout.addWidget(
            gas_label
        )

        self.dga_chart = DGAGasChart()

        self.dga_chart.setMinimumHeight(
            280
        )

        self.dga_chart.update_chart(
            gas_values
        )

        layout.addWidget(
            self.dga_chart,
            1
        )

        # ==================================================
        # CLOSE BUTTON
        # ==================================================

        button_layout = QHBoxLayout()

        button_layout.addStretch()

        close_button = QPushButton(
            "CLOSE"
        )

        close_button.setMinimumWidth(
            120
        )

        close_button.clicked.connect(
            self.close
        )

        button_layout.addWidget(
            close_button
        )

        layout.addLayout(
            button_layout
        )






class TransformerFaultApp(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Transformer Fault Diagnosis"
        )

        self.resize(
            900,
            650
        )

        self.setMinimumSize(
            800,
            600
        )

        self.inputs = {}

        self.create_ui()


    def show_analysis(self):



        if not hasattr(self, "last_probabilities"):
            return

        window = AnalysisWindow(
            self.last_probabilities,
            self.last_gas_values,
            self
        )

        window.exec()







    def export_pdf_report(self):

        if not hasattr(self, "last_probabilities"):

            QMessageBox.warning(
                self,
                "No Analysis",
                "Please analyze a transformer first."
            )

            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF Report",
            "transformer_diagnosis_report.pdf",
            "PDF Files (*.pdf)"
        )

        if not file_path:
            return

        try:

            probabilities = self.last_probabilities
            gas_values = self.last_gas_values

            predicted_fault = max(
                probabilities,
                key=probabilities.get
            )

            predicted_fault_display = FAULT_NAMES.get(
                predicted_fault,
                predicted_fault
            )

            predicted_probability = (
                probabilities[predicted_fault] * 100
            )

            document = SimpleDocTemplate(
                file_path,
                pagesize=A4,
                rightMargin=18 * mm,
                leftMargin=18 * mm,
                topMargin=18 * mm,
                bottomMargin=18 * mm,
            )

            styles = getSampleStyleSheet()

            story = []

            # ==========================================
            # TITLE
            # ==========================================

            story.append(
                Paragraph(
                    "TRANSFORMER FAULT DIAGNOSIS REPORT",
                    styles["Title"]
                )
            )

            story.append(
                Spacer(1, 8)
            )

            story.append(
                Paragraph(
                    "Dissolved Gas Analysis (DGA)",
                    styles["Heading2"]
                )
            )

            story.append(
                Spacer(1, 10)
            )

            # ==========================================
            # DATE
            # ==========================================

            analysis_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            story.append(
                Paragraph(
                    f"<b>Analysis Date:</b> {analysis_time}",
                    styles["Normal"]
                )
            )

            story.append(
                Spacer(1, 15)
            )

            # ==========================================
            # DGA MEASUREMENTS
            # ==========================================

            story.append(
                Paragraph(
                    "DGA Measurements",
                    styles["Heading2"]
                )
            )

            gas_data = [
                ["Gas", "Concentration (ppm)"]
            ]

            for gas in GAS_COLUMNS:

                gas_data.append([
                    gas,
                    f"{gas_values[gas]:.2f}"
                ])

            gas_table = Table(
                gas_data,
                colWidths=[
                    70 * mm,
                    70 * mm
                ]
            )

            gas_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (1, -1),
                        "RIGHT"
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                ])
            )

            story.append(gas_table)

            story.append(
                Spacer(1, 20)
            )

            # ==========================================
            # DIAGNOSIS
            # ==========================================

            story.append(
                Paragraph(
                    "Diagnosis Result",
                    styles["Heading2"]
                )
            )

            diagnosis_data = [
                ["Parameter", "Result"],
                [
                    "Predicted Fault",
                    predicted_fault_display
                ],
                [
                    "Model Probability",
                    f"{predicted_probability:.2f}%"
                ],
            ]

            diagnosis_table = Table(
                diagnosis_data,
                colWidths=[
                    70 * mm,
                    70 * mm
                ]
            )

            diagnosis_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                ])
            )

            story.append(
                diagnosis_table
            )

            story.append(
                Spacer(1, 20)
            )

            # ==========================================
            # PROBABILITIES
            # ==========================================

            story.append(
                Paragraph(
                    "Fault Probability Distribution",
                    styles["Heading2"]
                )
            )

            probability_data = [
                ["Fault", "Model Probability"]
            ]

            for fault, probability in sorted(
                probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            ):

                probability_data.append([
                    FAULT_NAMES.get(
                        fault,
                        fault
                    ),
                    f"{probability * 100:.2f}%"
                ])

            probability_table = Table(
                probability_data,
                colWidths=[
                    100 * mm,
                    40 * mm
                ]
            )

            probability_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (1, -1),
                        "RIGHT"
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6
                    ),
                ])
            )

            story.append(
                probability_table
            )

            story.append(
                Spacer(1, 20)
            )

            story.append(
                Paragraph(
                    "<b>Note:</b> The diagnosis is generated by a "
                    "machine-learning model trained on the supplied "
                    "DGA dataset. Model probability represents the "
                    "model's predicted class probability.",
                    styles["Normal"]
                )
            )

            document.build(story)

            QMessageBox.information(
                self,
                "Report Generated",
                f"PDF report saved to:\n\n{file_path}"
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Report Error",
                str(error)
            )

















    def create_ui(self):

        main_layout = QVBoxLayout()

        main_layout.setContentsMargins(
            25, 20, 25, 20
        )

        main_layout.setSpacing(15)


        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI";
                font-size: 14px;
                color: #1f2937;
            }

            QFrame {
                border: 1px solid #d0d0d0;
                border-radius: 10px;
                background: #fafafa;
            }

            QLineEdit {
                border: 1px solid #bdbdbd;
                border-radius: 6px;
                padding: 8px;
                background: white;
            }

            QLineEdit:focus {
                border: 2px solid #4a90e2;
            }

            QPushButton {
                border: none;
                border-radius: 7px;
                padding: 10px 16px;
                font-weight: bold;
                background: #4a90e2;
                color: white;
            }

            QPushButton:hover {
                background: #357abd;
            }

            QPushButton:disabled {
                background: #bdbdbd;
                color: #eeeeee;
            }
        """)




        # ==================================================
        # TITLE
        # ==================================================

        title = QLabel(
            "TRANSFORMER FAULT DIAGNOSIS"
        )

        title.setAlignment(Qt.AlignCenter)

        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                padding: 12px;
            }
        """)

        subtitle = QLabel(
            "Dissolved Gas Analysis (DGA) Based "
            "Transformer Fault Detection"
        )

        subtitle.setAlignment(Qt.AlignCenter)

        subtitle.setStyleSheet("""
            QLabel {
                font-size: 14px;
                padding-bottom: 15px;
            }
        """)

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # ==================================================
        # MAIN CONTENT
        # ==================================================

        content_layout = QHBoxLayout()

        # ==================================================
        # INPUT PANEL
        # ==================================================

        input_frame = QFrame()

        input_frame.setFrameShape(
            QFrame.StyledPanel
        )

        input_layout = QVBoxLayout()

        input_title = QLabel(
            "DGA Measurements"
        )

        input_title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            padding-bottom: 10px;
        """)

        input_layout.addWidget(input_title)

        grid = QGridLayout()

        for row, gas in enumerate(GAS_COLUMNS):

            label = QLabel(
                f"{gas} (ppm)"
            )

            field = QLineEdit()

            field.setPlaceholderText(
                "Enter value"
            )

            field.setMinimumHeight(38)

            self.inputs[gas] = field

            grid.addWidget(
                label,
                row,
                0
            )

            grid.addWidget(
                field,
                row,
                1
            )

        input_layout.addLayout(grid)

        # ==================================================
        # ANALYZE BUTTON
        # ==================================================

        analyze_button = QPushButton(
            "ANALYZE TRANSFORMER"
        )

        analyze_button.setMinimumHeight(50)

        analyze_button.clicked.connect(
            self.analyze
        )

        input_layout.addSpacing(15)

        input_layout.addWidget(
            analyze_button
        )

        # ==================================================
        # LOAD FILE BUTTON
        # ==================================================

        file_button = QPushButton(
            "LOAD CSV / EXCEL"
        )

        file_button.setMinimumHeight(45)

        file_button.clicked.connect(
            self.load_file
        )

        input_layout.addWidget(
            file_button
        )

        input_layout.addStretch()

        input_frame.setLayout(
            input_layout
        )

        # ==================================================
        # RESULT PANEL
        # ==================================================

        result_frame = QFrame()

        result_frame.setFrameShape(
            QFrame.StyledPanel
        )

        result_layout = QVBoxLayout()

        result_title = QLabel(
            "Diagnosis Result"
        )

        result_title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            padding-bottom: 10px;
        """)

        result_layout.addWidget(
            result_title
        )

        self.result_label = QLabel(
            "Enter DGA values\n"
            "and click ANALYZE TRANSFORMER"
        )

        self.result_label.setAlignment(
            Qt.AlignCenter
        )

        self.result_label.setWordWrap(True)

        self.result_label.setStyleSheet("""
    font-size: 24px;
    font-weight: bold;
    padding: 30px;
    color: #1f2937;
""")

        result_layout.addWidget(
            self.result_label
        )

        self.probability_label = QLabel("")

        self.probability_label.setAlignment(
            Qt.AlignCenter
        )

        self.probability_label.setStyleSheet("""
            font-size: 17px;
            font-weight: bold;
            padding: 10px;
            color: #1f2937;
        """)

        result_layout.addWidget(
            self.probability_label
        )

        result_layout.addStretch()

        result_frame.setLayout(
            result_layout
        )

        # ==================================================
        # ADD PANELS
        # ==================================================

        content_layout.addWidget(
            input_frame,
            1
        )

        content_layout.addWidget(
            result_frame,
            1
        )

        main_layout.addLayout(
            content_layout
        )

        # ==================================================
        # VIEW ANALYSIS BUTTON
        # ==================================================

        self.analysis_button = QPushButton(
            "VIEW ANALYSIS"
        )

        self.analysis_button.setMinimumHeight(
            45
        )

        self.analysis_button.setEnabled(False)

        self.analysis_button.clicked.connect(
            self.show_analysis
        )

        self.report_button = QPushButton(
            "EXPORT PDF REPORT"
        )

        self.report_button.setMinimumHeight(
            45
        )

        self.report_button.setEnabled(
            False
        )

        self.report_button.clicked.connect(
            self.export_pdf_report
        )

        main_layout.addWidget(
            self.report_button
        )

        main_layout.addWidget(
            self.analysis_button
        )

        # ==================================================
        # SET MAIN LAYOUT
        # ==================================================

        self.setLayout(
            main_layout
        )

    
        

    

    # ======================================================
    # MANUAL ANALYSIS
    # ======================================================

    def analyze(self):

        values = {}

        try:

            for key, field in self.inputs.items():

                text = field.text().strip()

                if not text:

                    raise ValueError(
                        f"Please enter a value for {key}."
                    )

                value = float(text)

                if value < 0:

                    raise ValueError(
                        f"{key} cannot be negative."
                    )

                values[key] = value

            fault, probabilities = predict_fault(
                H2=values["H2"],
                CH4=values["CH4"],
                C2H6=values["C2H6"],
                C2H4=values["C2H4"],
                C2H2=values["C2H2"],
            )

            display_fault = FAULT_NAMES.get(
                fault,
                fault
            )

            predicted_probability = (
                probabilities[fault] * 100
            )

            self.result_label.setText(
                "Predicted Fault\n\n"
                f"{display_fault}"
            )

            self.probability_label.setText(
                "Model Probability: "
                f"{predicted_probability:.2f}%"
            )

            self.last_probabilities = probabilities

            self.last_gas_values = values

            self.analysis_button.setEnabled(True)
            self.report_button.setEnabled(True)

        except ValueError as error:

            QMessageBox.warning(
                self,
                "Invalid Input",
                str(error)
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Prediction Error",
                str(error)
            )

    # ======================================================
    # LOAD CSV / EXCEL
    # ======================================================

    def load_file(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select DGA Dataset",
            "",
            "Data Files (*.csv *.xlsx *.xls)"
        )

        if not file_path:

            return

        try:

            path = Path(file_path)

            if path.suffix.lower() == ".csv":

                df = pd.read_csv(
                    file_path
                )

            else:

                df = pd.read_excel(
                    file_path
                )

            # ----------------------------------------------
            # Check required columns
            # ----------------------------------------------

            missing = [
                gas
                for gas in GAS_COLUMNS
                if gas not in df.columns
            ]

            if missing:

                raise ValueError(
                    "Missing required columns:\n\n"
                    + ", ".join(missing)
                    + "\n\nRequired columns are:\n"
                    + ", ".join(GAS_COLUMNS)
                )

            results = []

            for index, row in df.iterrows():

                values = {}

                valid = True

                for gas in GAS_COLUMNS:

                    value = pd.to_numeric(
                        row[gas],
                        errors="coerce"
                    )

                    if pd.isna(value) or value < 0:

                        valid = False

                        break

                    values[gas] = float(value)

                if not valid:

                    continue

                fault, probabilities = predict_fault(
                    H2=values["H2"],
                    CH4=values["CH4"],
                    C2H6=values["C2H6"],
                    C2H4=values["C2H4"],
                    C2H2=values["C2H2"],
                )

                results.append(
                    {
                        "sample": index + 1,
                        "fault": FAULT_NAMES.get(
                            fault,
                            fault
                        ),
                        "probability":
                            probabilities[fault] * 100
                    }
                )

            if not results:

                raise ValueError(
                    "No valid DGA rows were found."
                )

            dialog = BatchResultsDialog(
                results,
                self
            )

            dialog.exec()

        except Exception as error:

            QMessageBox.critical(
                self,
                "File Error",
                str(error)
            )


def main():

    app = QApplication(
        sys.argv
    )

    window = TransformerFaultApp()

    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":

    main()