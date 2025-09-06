#include "datasetmodel.h"

#include <QAbstractTableModel>
#include <boost/multi_array.hpp>

DatasetModel::DatasetModel(QObject* parent)
    : QAbstractTableModel(parent), dataset{nullptr} {}

void DatasetModel::setData(
    Dataset *dataset
) {
    beginResetModel();
    this->dataset = dataset;
    endResetModel();
}

int DatasetModel::rowCount(const QModelIndex& modelIndex) const  {
    return dataset ? dataset->getX().shape()[0] : 0;
}

int DatasetModel::columnCount(const QModelIndex& modelIndex) const   {
    return dataset ? dataset->getX().shape()[1] + 2: 0;
}

    // Данные для отображения
QVariant DatasetModel::data(const QModelIndex& index, int role) const {
    if (!index.isValid() || !dataset || role != Qt::DisplayRole)
        return QVariant();

    const auto& x = dataset->getX();
    const auto& d = dataset->getD();
    const auto& paramNames = dataset->getParamNames();
    const auto& classNames = dataset->getClassNames();
    const int classCount = dataset->getClassCount();

    const int row = index.row();
    const int col = index.column();

    if (x.shape()[0] > 0 && x.shape()[1] > 0) {
        if (col == x.shape()[1]) {
            if (row >= static_cast<int>(d.size()))
                return QVariant();

            int classId = static_cast<int>(round(d[row] * (classCount-1)));
            if (classId >= static_cast<int>(classNames.size()))
                return classId;
            return QString::fromStdString(classNames[classId]);
        }
        if (row < static_cast<int>(x.shape()[0]) &&
            col < static_cast<int>(x.shape()[1])) {
            return QString::number(x[row][col], 'f', 4);
        }
    }
    return QVariant();
}

QVariant DatasetModel::headerData(int section, Qt::Orientation orientation, int role) const {
    if (role != Qt::DisplayRole || !dataset)
        return QVariant();

    const auto& x = dataset->getX();
    const auto& paramNames = dataset->getParamNames();

    if (orientation == Qt::Vertical)
    {
        if(section < static_cast<int>(x.shape()[0]) + 1)
        {
            return QString("Row %1").arg(section + 1);
        }
    }
    if (orientation == Qt::Horizontal) {
        // Заголовки для столбцов с данными
        if (section < static_cast<int>(x.shape()[1])) {
            return section < static_cast<int>(paramNames.size())
                       ? QString::fromStdString(paramNames[section])
                       : QString("Feature %1").arg(section - 1);
        }
        // Заголовок для столбца с классом
        else if (section == x.shape()[1]) {
            return "Class";
        }
    }

    return QVariant();
}
