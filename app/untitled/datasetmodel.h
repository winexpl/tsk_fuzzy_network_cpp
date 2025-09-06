#ifndef DATASETMODEL_H
#define DATASETMODEL_H

#include <QAbstractTableModel>
#include "tsk_fuzzy_network/dataset.h"
#include <boost/multi_array.hpp>

class DatasetModel : public QAbstractTableModel {
    Q_OBJECT
public:
    DatasetModel(QObject* parent = nullptr);
    void setData(Dataset *dataset);
    int rowCount(const QModelIndex& = QModelIndex()) const override;
    int columnCount(const QModelIndex& = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role) const;
    QVariant headerData(int section, Qt::Orientation orientation, int role) const;
private:
    Dataset *dataset;
};

#endif
