import sys
import os
import platform

from PySide6.QtWidgets import QHeaderView

from modules import *
from widgets import *

os.environ["QT_FONT_DPI"] = "96"  # 修复DPI高且比例高于100%的问题

# 设置为全局widgets
widgets = None


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # 设置为全局widgets
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # 使用自定义标题栏 | 在Mac或Linux中改为“False”
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        title = "="
        description = "手写算式识别"
        # 应用文本
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # 切换菜单
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # 设置用户界面定义
        UIFunctions.uiDefinitions(self)

        # buttons点击

        # 左菜单
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)

        # 额外的left box
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)

        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # 额外的right box
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)

        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # 显示app
        self.show()

        # 设置自定义主题
        useCustomTheme = False
        themeFile = r"themes\py_dracula_light.qss"

        # 设置主题和hacks
        if useCustomTheme:
            # 加载并应用style
            UIFunctions.theme(self, themeFile, True)

            # 设置 hacks
            AppFunctions.setThemeHack(self)

        # 设置主页并选择菜单
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))

    # buttons click
    # 在这里添加点击按钮的功能
    def buttonClick(self):
        # 点击按钮
        btn = self.sender()
        btnName = btn.objectName()

        # 显示home页面
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # 显示widgets页面
        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # 打印button名称
        # print(f'Button "{btnName}" pressed!')

    # 调整events
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # 鼠标点击events
    def mousePressEvent(self, event):
        p = event.globalPosition()
        globalPos = p.toPoint()
        self.dragPos = globalPos


if __name__ == "__main__":
    app = QApplication()
    app.setWindowIcon(QIcon("./images/icons/icon.ico"))
    window = MainWindow()
    sys.exit(app.exec())
