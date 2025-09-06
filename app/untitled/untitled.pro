QT += core gui widgets charts

CONFIG += c++23
QMAKE_CXXFLAGS += -std=c++23

# Boost libraries
LIBS += -lboost_log \
        -lboost_log_setup \
        -lboost_system \
        -lboost_thread \
        -lboost_serialization
# TDD
LIBS += -ltbb

# Custom TSK library
LIBS += -L/mnt/masha/projects/tsk-fuzzy-network-cpp/lib/ -lTSK

# Include paths
INCLUDEPATH += /mnt/masha/projects/tsk-fuzzy-network-cpp/include
INCLUDEPATH += /usr/include/boost

# Для поддержки концептов C++20 (если требуется)
DEFINES += BOOST_ALL_NO_LIB
QMAKE_CXXFLAGS += -fconcepts

SOURCES += \
    datasetmodel.cpp \
    main.cpp \
    mainwindow.cpp \
    state.cpp

HEADERS += \
    datasetmodel.h \
    mainwindow.h \
    state.h

FORMS += \
    mainwindow.ui

# Deployment rules
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
