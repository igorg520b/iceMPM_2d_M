// render_selector_dialog.h
// Dialog for selecting which visualization options to render in batch mode

#ifndef RENDER_SELECTOR_DIALOG_H
#define RENDER_SELECTOR_DIALOG_H

#include <QDialog>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QScrollArea>
#include <QPushButton>
#include <QMap>
#include <vector>
#include "visual_representation.h"

class RenderSelectorDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RenderSelectorDialog(QWidget *parent = nullptr);
    ~RenderSelectorDialog() = default;

    // Get list of checked visualization options
    std::vector<VisualRepresentation::VisOpt> getSelectedOptions() const;

    // Save/restore checkbox states using QSettings
    void saveSelections(const QString& settingsFile);
    void loadSelections(const QString& settingsFile);

    // Programmatically set which options are selected
    void setSelectedOptions(const std::vector<VisualRepresentation::VisOpt>& options);

Q_SIGNALS:
    void selectionsChanged();  // Emitted when user changes checkboxes

private Q_SLOTS:
    void onSelectAll();
    void onDeselectAll();

private:
    // Map from VisOpt enum value to QCheckBox widget
    QMap<VisualRepresentation::VisOpt, QCheckBox*> m_checkboxMap;

    void setupUI();
    void populateCheckboxes();
    void setDefaultSelections();
};

#endif // RENDER_SELECTOR_DIALOG_H
