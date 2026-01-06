// render_selector_dialog.cpp

#include "render_selector_dialog.h"
#include <QMetaEnum>
#include <QSettings>
#include <QGroupBox>
#include <QHBoxLayout>

RenderSelectorDialog::RenderSelectorDialog(QWidget *parent)
    : QDialog(parent)
{
    // Set window properties
    setWindowTitle("Select Visualizations to Render");
    setWindowFlags(Qt::Tool | Qt::WindowStaysOnTopHint);
    resize(400, 600);

    setupUI();
}

void RenderSelectorDialog::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Create button group at top for Select All / Deselect All
    QHBoxLayout *buttonLayout = new QHBoxLayout();
    QPushButton *selectAllBtn = new QPushButton("Select All");
    QPushButton *deselectAllBtn = new QPushButton("Deselect All");

    connect(selectAllBtn, &QPushButton::clicked, this, &RenderSelectorDialog::onSelectAll);
    connect(deselectAllBtn, &QPushButton::clicked, this, &RenderSelectorDialog::onDeselectAll);

    buttonLayout->addWidget(selectAllBtn);
    buttonLayout->addWidget(deselectAllBtn);
    buttonLayout->addStretch();

    mainLayout->addLayout(buttonLayout);

    // Create scroll area for checkboxes
    QScrollArea *scrollArea = new QScrollArea();
    scrollArea->setWidgetResizable(true);
    scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // Create container widget for checkboxes
    QWidget *containerWidget = new QWidget();
    QVBoxLayout *containerLayout = new QVBoxLayout(containerWidget);
    containerLayout->setContentsMargins(5, 5, 5, 5);
    containerLayout->setSpacing(5);

    // Populate checkboxes
    populateCheckboxes();

    // Add checkboxes to container layout
    for (auto it = m_checkboxMap.begin(); it != m_checkboxMap.end(); ++it) {
        containerLayout->addWidget(it.value());
    }

    containerLayout->addStretch();
    containerWidget->setLayout(containerLayout);
    scrollArea->setWidget(containerWidget);

    mainLayout->addWidget(scrollArea);

    // Add close button at bottom
    QHBoxLayout *closeLayout = new QHBoxLayout();
    closeLayout->addStretch();
    QPushButton *closeBtn = new QPushButton("Close");
    connect(closeBtn, &QPushButton::clicked, this, &QDialog::hide);
    closeLayout->addWidget(closeBtn);
    closeLayout->addStretch();

    mainLayout->addLayout(closeLayout);

    setLayout(mainLayout);
}

void RenderSelectorDialog::populateCheckboxes()
{
    QMetaEnum qme = QMetaEnum::fromType<VisualRepresentation::VisOpt>();

    for (int i = 0; i < qme.keyCount(); i++) {
        VisualRepresentation::VisOpt opt =
            static_cast<VisualRepresentation::VisOpt>(qme.value(i));

        QString name = qme.key(i);

        // Skip "none" option - not useful for batch rendering
        if (opt == VisualRepresentation::VisOpt::none) {
            continue;
        }

        QCheckBox* checkbox = new QCheckBox(name);
        m_checkboxMap[opt] = checkbox;

        // Connect to emit signal when state changes
        connect(checkbox, &QCheckBox::stateChanged,
                this, &RenderSelectorDialog::selectionsChanged);
    }
}

std::vector<VisualRepresentation::VisOpt>
RenderSelectorDialog::getSelectedOptions() const
{
    std::vector<VisualRepresentation::VisOpt> selected;

    for (auto it = m_checkboxMap.begin(); it != m_checkboxMap.end(); ++it) {
        if (it.value()->isChecked()) {
            selected.push_back(it.key());
        }
    }

    return selected;
}

void RenderSelectorDialog::setSelectedOptions(const std::vector<VisualRepresentation::VisOpt>& options)
{
    // First, uncheck all
    for (auto it = m_checkboxMap.begin(); it != m_checkboxMap.end(); ++it) {
        it.value()->setChecked(false);
    }

    // Then check the specified ones
    for (const auto& opt : options) {
        if (m_checkboxMap.contains(opt)) {
            m_checkboxMap[opt]->setChecked(true);
        }
    }
}

void RenderSelectorDialog::saveSelections(const QString& settingsFile)
{
    QSettings settings(settingsFile, QSettings::IniFormat);

    // Get list of selected options
    std::vector<VisualRepresentation::VisOpt> selected = getSelectedOptions();

    // Convert to QList<QVariant> for saving
    QList<QVariant> selectedList;
    for (const auto& opt : selected) {
        selectedList.append(static_cast<int>(opt));
    }

    settings.setValue("RenderSelector/SelectedOptions", selectedList);
    settings.sync();
}

void RenderSelectorDialog::loadSelections(const QString& settingsFile)
{
    QSettings settings(settingsFile, QSettings::IniFormat);

    // Check if we have saved selections
    if (!settings.contains("RenderSelector/SelectedOptions")) {
        // First run - set default selections
        setDefaultSelections();
        return;
    }

    // Load saved selections
    QList<QVariant> selectedList =
        settings.value("RenderSelector/SelectedOptions").toList();

    std::vector<VisualRepresentation::VisOpt> selected;
    for (const QVariant& v : selectedList) {
        selected.push_back(
            static_cast<VisualRepresentation::VisOpt>(v.toInt()));
    }

    setSelectedOptions(selected);
}

void RenderSelectorDialog::setDefaultSelections()
{
    // Default options (matches the current hardcoded list in pp_mainwindow.h)
    std::vector<VisualRepresentation::VisOpt> defaults = {
        VisualRepresentation::VisOpt::grid_Jpinv,
        VisualRepresentation::VisOpt::grid_ridges,
        VisualRepresentation::VisOpt::grid_Q,
        VisualRepresentation::VisOpt::grid_P,
        VisualRepresentation::VisOpt::grid_vnorm
    };

    setSelectedOptions(defaults);
}

void RenderSelectorDialog::onSelectAll()
{
    for (auto it = m_checkboxMap.begin(); it != m_checkboxMap.end(); ++it) {
        it.value()->setChecked(true);
    }
}

void RenderSelectorDialog::onDeselectAll()
{
    for (auto it = m_checkboxMap.begin(); it != m_checkboxMap.end(); ++it) {
        it.value()->setChecked(false);
    }
}
