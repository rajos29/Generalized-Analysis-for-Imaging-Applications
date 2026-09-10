from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QGridLayout, QLabel, QSpinBox

class Hdf5DatasetDialog(QDialog):
    def __init__(self, path: Path, catalog: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select HDF5 imaging sample")
        self.catalog = catalog
        self.groups: dict[str, dict[str, dict[str, object]]] = catalog["groups"]  # type: ignore[assignment]
        self.samples: list[str] = catalog["common_samples"] or catalog["all_samples"]  # type: ignore[assignment]
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"File: {path.name}"))

        form = QGridLayout()
        self.sample_combo = QComboBox()
        self.input_combo = QComboBox()
        self.reference_combo = QComboBox()
        self.z_spin = QSpinBox()
        self.z_spin.setMinimum(0)
        for sample in self.samples:
            self.sample_combo.addItem(sample, sample)
        for group in sorted(self.groups):
            self.input_combo.addItem(group, group)
            self.reference_combo.addItem(group, group)
        if "noisy" in self.groups:
            self.input_combo.setCurrentText("noisy")
        elif "noisy_1" in self.groups:
            self.input_combo.setCurrentText("noisy_1")
        if "clean" in self.groups:
            self.reference_combo.setCurrentText("clean")

        form.addWidget(QLabel("Sample"), 0, 0)
        form.addWidget(self.sample_combo, 0, 1)
        form.addWidget(QLabel("Input"), 1, 0)
        form.addWidget(self.input_combo, 1, 1)
        form.addWidget(QLabel("Reference"), 2, 0)
        form.addWidget(self.reference_combo, 2, 1)
        form.addWidget(QLabel("Z slice"), 3, 0)
        form.addWidget(self.z_spin, 3, 1)
        layout.addLayout(form)

        note = QLabel("The app will keep paired samples linked while you navigate the dataset.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.sample_combo.currentTextChanged.connect(self.update_z_range)
        self.input_combo.currentTextChanged.connect(self.update_z_range)
        self.update_z_range()

    def update_z_range(self) -> None:
        sample = str(self.sample_combo.currentData())
        group = str(self.input_combo.currentData())
        info = self.groups.get(group, {}).get(sample)
        shape = tuple(info["shape"]) if info else ()  # type: ignore[index]
        if len(shape) == 3 and shape[-1] not in (3, 4):
            self.z_spin.setEnabled(True)
            self.z_spin.setRange(0, int(shape[0]) - 1)
            self.z_spin.setValue(int(shape[0]) // 2)
        else:
            self.z_spin.setEnabled(False)
            self.z_spin.setRange(0, 0)
            self.z_spin.setValue(0)

    def selected_values(self) -> dict[str, object]:
        return {
            "sample": str(self.sample_combo.currentData()),
            "input_group": str(self.input_combo.currentData()),
            "reference_group": str(self.reference_combo.currentData()),
            "z_index": self.z_spin.value(),
        }


