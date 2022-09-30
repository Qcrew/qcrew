# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'server_widget.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_server_widget(object):
    def setupUi(self, server_widget):
        server_widget.setObjectName("server_widget")
        server_widget.resize(900, 300)
        server_widget.setMinimumSize(QtCore.QSize(900, 300))
        self.verticalLayout = QtWidgets.QVBoxLayout(server_widget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.server_tab_layout = QtWidgets.QHBoxLayout()
        self.server_tab_layout.setContentsMargins(5, 5, 5, 5)
        self.server_tab_layout.setSpacing(5)
        self.server_tab_layout.setObjectName("server_tab_layout")
        self.instrument_types_layout = QtWidgets.QVBoxLayout()
        self.instrument_types_layout.setContentsMargins(5, 5, 5, 5)
        self.instrument_types_layout.setSpacing(5)
        self.instrument_types_layout.setObjectName("instrument_types_layout")
        self.instrument_types_label = QtWidgets.QLabel(server_widget)
        self.instrument_types_label.setAlignment(QtCore.Qt.AlignCenter)
        self.instrument_types_label.setObjectName("instrument_types_label")
        self.instrument_types_layout.addWidget(self.instrument_types_label)
        self.instrument_types_list = QtWidgets.QListWidget(server_widget)
        self.instrument_types_list.setObjectName("instrument_types_list")
        self.instrument_types_layout.addWidget(self.instrument_types_list)
        self.server_tab_layout.addLayout(self.instrument_types_layout)
        self.instrument_ids_layout = QtWidgets.QVBoxLayout()
        self.instrument_ids_layout.setContentsMargins(5, 5, 5, 5)
        self.instrument_ids_layout.setSpacing(5)
        self.instrument_ids_layout.setObjectName("instrument_ids_layout")
        self.instrument_ids_label = QtWidgets.QLabel(server_widget)
        self.instrument_ids_label.setAlignment(QtCore.Qt.AlignCenter)
        self.instrument_ids_label.setObjectName("instrument_ids_label")
        self.instrument_ids_layout.addWidget(self.instrument_ids_label)
        self.instrument_ids_list = QtWidgets.QListWidget(server_widget)
        self.instrument_ids_list.setObjectName("instrument_ids_list")
        self.instrument_ids_layout.addWidget(self.instrument_ids_list)
        self.server_tab_layout.addLayout(self.instrument_ids_layout)
        self.stage_buttons_layout = QtWidgets.QVBoxLayout()
        self.stage_buttons_layout.setContentsMargins(5, 5, 5, 5)
        self.stage_buttons_layout.setSpacing(5)
        self.stage_buttons_layout.setObjectName("stage_buttons_layout")
        self.stage_button = QtWidgets.QPushButton(server_widget)
        self.stage_button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.stage_button.sizePolicy().hasHeightForWidth())
        self.stage_button.setSizePolicy(sizePolicy)
        self.stage_button.setObjectName("stage_button")
        self.stage_buttons_layout.addWidget(self.stage_button)
        self.unstage_button = QtWidgets.QPushButton(server_widget)
        self.unstage_button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.unstage_button.sizePolicy().hasHeightForWidth()
        )
        self.unstage_button.setSizePolicy(sizePolicy)
        self.unstage_button.setObjectName("unstage_button")
        self.stage_buttons_layout.addWidget(self.unstage_button)
        self.server_tab_layout.addLayout(self.stage_buttons_layout)
        self.staged_instruments_layout = QtWidgets.QVBoxLayout()
        self.staged_instruments_layout.setContentsMargins(5, 5, 5, 5)
        self.staged_instruments_layout.setSpacing(5)
        self.staged_instruments_layout.setObjectName("staged_instruments_layout")
        self.staged_instruments_label = QtWidgets.QLabel(server_widget)
        self.staged_instruments_label.setAlignment(QtCore.Qt.AlignCenter)
        self.staged_instruments_label.setObjectName("staged_instruments_label")
        self.staged_instruments_layout.addWidget(self.staged_instruments_label)
        self.staged_instruments_list = QtWidgets.QListWidget(server_widget)
        self.staged_instruments_list.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection
        )
        self.staged_instruments_list.setObjectName("staged_instruments_list")
        self.staged_instruments_layout.addWidget(self.staged_instruments_list)
        self.server_tab_layout.addLayout(self.staged_instruments_layout)
        self.serve_buttons_layout = QtWidgets.QVBoxLayout()
        self.serve_buttons_layout.setContentsMargins(5, 5, 5, 5)
        self.serve_buttons_layout.setSpacing(20)
        self.serve_buttons_layout.setObjectName("serve_buttons_layout")
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.serve_buttons_layout.addItem(spacerItem)
        self.setup_button = QtWidgets.QPushButton(server_widget)
        self.setup_button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.setup_button.sizePolicy().hasHeightForWidth())
        self.setup_button.setSizePolicy(sizePolicy)
        self.setup_button.setObjectName("setup_button")
        self.serve_buttons_layout.addWidget(self.setup_button)
        self.teardown_button = QtWidgets.QPushButton(server_widget)
        self.teardown_button.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.teardown_button.sizePolicy().hasHeightForWidth()
        )
        self.teardown_button.setSizePolicy(sizePolicy)
        self.teardown_button.setObjectName("teardown_button")
        self.serve_buttons_layout.addWidget(self.teardown_button)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.serve_buttons_layout.addItem(spacerItem1)
        self.serve_buttons_layout.setStretch(0, 3)
        self.serve_buttons_layout.setStretch(1, 2)
        self.serve_buttons_layout.setStretch(2, 2)
        self.serve_buttons_layout.setStretch(3, 3)
        self.server_tab_layout.addLayout(self.serve_buttons_layout)
        self.server_tab_layout.setStretch(0, 4)
        self.server_tab_layout.setStretch(1, 4)
        self.server_tab_layout.setStretch(2, 1)
        self.server_tab_layout.setStretch(3, 4)
        self.server_tab_layout.setStretch(4, 3)
        self.verticalLayout.addLayout(self.server_tab_layout)

        self.retranslateUi(server_widget)
        QtCore.QMetaObject.connectSlotsByName(server_widget)

    def retranslateUi(self, server_widget):
        _translate = QtCore.QCoreApplication.translate
        server_widget.setWindowTitle(_translate("server_widget", "Server"))
        self.instrument_types_label.setText(
            _translate("server_widget", "Select instrument type")
        )
        self.instrument_ids_label.setText(
            _translate("server_widget", "Select instrument ID")
        )
        self.stage_button.setText(_translate("server_widget", "Stage"))
        self.unstage_button.setText(_translate("server_widget", "Unstage"))
        self.staged_instruments_label.setText(
            _translate("server_widget", "Staged instrument(s)")
        )
        self.setup_button.setText(_translate("server_widget", "Setup stage"))
        self.teardown_button.setText(_translate("server_widget", "Teardown stage"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    server_widget = QtWidgets.QWidget()
    ui = Ui_server_widget()
    ui.setupUi(server_widget)
    server_widget.show()
    sys.exit(app.exec_())
