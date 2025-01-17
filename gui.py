import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QFileDialog, QProgressBar, QLineEdit, QTextEdit,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QSlider,
                             QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont
from src.enhanced_tracker import EnhancedPlayerTracker


class TrackingThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, video_path, output_path, jersey_color, jersey_number, conf_threshold, min_clip_duration, use_gpu):
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.jersey_color = jersey_color
        self.jersey_number = jersey_number
        self.conf_threshold = conf_threshold
        self.min_clip_duration = min_clip_duration
        self.use_gpu = use_gpu

    def run(self):
        try:
            device = 'cuda' if self.use_gpu else 'cpu'
            self.log.emit(f"Initializing Enhanced Player Tracker on {device.upper()}...")
            tracker = EnhancedPlayerTracker(
                jersey_number=self.jersey_number,
                jersey_color=self.jersey_color,
                conf_threshold=self.conf_threshold,
                min_clip_duration=self.min_clip_duration
            )
            self.log.emit(f"Tracking player {self.jersey_number} with jersey color '{self.jersey_color}'...")
            highlight_path = tracker.track_player(self.video_path, self.output_path, self.progress)
            if highlight_path:
                self.finished.emit(highlight_path)
            else:
                self.log.emit("No players detected or processing failed.")
                self.finished.emit(None)
        except Exception as e:
            self.log.emit(f"Critical Error: {str(e)}")
            self.finished.emit(None)


class HockeyTrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hockey Player Tracker")
        self.setGeometry(100, 100, 800, 600)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # File Selection Group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()

        # Video file selection
        video_layout = QHBoxLayout()
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("Select video file...")
        self.video_path.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_video)
        video_layout.addWidget(self.video_path)
        video_layout.addWidget(browse_btn)
        file_layout.addLayout(video_layout)

        # Output directory selection
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Select output directory...")
        self.output_path.setReadOnly(True)
        output_btn = QPushButton("Output")
        output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(output_btn)
        file_layout.addLayout(output_layout)

        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)

        # Player Detection Settings Group
        settings_group = QGroupBox("Player Detection Settings")
        settings_layout = QVBoxLayout()

        # Jersey color selection
        color_layout = QHBoxLayout()
        color_label = QLabel("Jersey Color:")
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            "black-jersey",
            "white-red-jersey",
            "white-jersey",
            "black-white-numbers",
            "game-red",
            "game-blue",
            "game-black",
            "black-light-blue",
            "white-red-blue"
        ])
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_combo)
        settings_layout.addLayout(color_layout)

        # Player number input
        number_layout = QHBoxLayout()
        number_label = QLabel("Player Number:")
        self.number_input = QSpinBox()
        self.number_input.setRange(0, 99)
        number_layout.addWidget(number_label)
        number_layout.addWidget(self.number_input)
        settings_layout.addLayout(number_layout)

        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # Advanced Settings Group
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QVBoxLayout()

        # GPU toggle
        gpu_layout = QHBoxLayout()
        gpu_label = QLabel("Use GPU:")
        self.gpu_checkbox = QCheckBox()
        gpu_layout.addWidget(gpu_label)
        gpu_layout.addWidget(self.gpu_checkbox)
        advanced_layout.addLayout(gpu_layout)

        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Detection Confidence:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)
        self.conf_value = QLabel("0.50")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_value.setText(f"{v / 100:.2f}")
        )
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value)
        advanced_layout.addLayout(conf_layout)

        # Minimum clip duration
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Min Clip Duration (s):")
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 10.0)
        self.duration_spin.setValue(1.5)
        self.duration_spin.setSingleStep(0.5)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_spin)
        advanced_layout.addLayout(duration_layout)

        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)

        # Process button
        self.process_btn = QPushButton("Start Processing")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        main_layout.addWidget(self.process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress_bar)

        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        main_layout.addWidget(self.log_output)

        # Connect signals
        self.video_path.textChanged.connect(self.check_inputs)
        self.output_path.textChanged.connect(self.check_inputs)
        self.number_input.valueChanged.connect(self.check_inputs)

    def browse_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        if file_name:
            self.video_path.setText(file_name)

    def browse_output(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        if directory:
            self.output_path.setText(directory)

    def check_inputs(self):
        """Enable process button if all inputs are valid"""
        self.process_btn.setEnabled(
            bool(self.video_path.text() and 
                 self.output_path.text() and 
                 self.number_input.value() > 0)
        )

    def start_processing(self):
        """Start video processing"""
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_output.clear()

        video_path = self.video_path.text()
        output_path = self.output_path.text()
        jersey_color = self.color_combo.currentText()
        jersey_number = str(self.number_input.value())
        conf_threshold = self.conf_slider.value() / 100
        min_clip_duration = self.duration_spin.value()
        use_gpu = self.gpu_checkbox.isChecked()

        self.thread = TrackingThread(
            video_path,
            output_path,
            jersey_color,
            jersey_number,
            conf_threshold,
            min_clip_duration,
            use_gpu
        )
        self.thread.progress.connect(self.update_progress)
        self.thread.log.connect(self.log_message)
        self.thread.finished.connect(self.processing_complete)
        self.thread.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def log_message(self, message):
        """Update log output"""
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def processing_complete(self, highlight_path):
        """Handle processing completion"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        if highlight_path:
            self.log_output.append(f"\nProcessing complete! Highlight saved at: {highlight_path}")
        else:
            self.log_output.append("\nProcessing failed. Check the log for errors.")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    window = HockeyTrackerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
