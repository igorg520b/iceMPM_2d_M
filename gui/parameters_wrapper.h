#ifndef PARAMETERS_WRAPPER_H
#define PARAMETERS_WRAPPER_H


#include <QObject>
#include <QString>
#include "simulation/parameters_sim.h"
#include <cmath>

// wrapper for SimParams to display/edit them in GUI
class ParamsWrapper : public QObject
{
    Q_OBJECT

Q_SIGNALS:
    void propertyChanged();

public:
    SimParams *prms;

    Q_PROPERTY(double in_TimeStep READ getTimeStep WRITE setTimeStep NOTIFY propertyChanged)
    double getTimeStep() {return prms->InitialTimeStep;}
    void setTimeStep(double val) { prms->InitialTimeStep = val; prms->ComputeHelperVariables();}

    Q_PROPERTY(QString in_TimeStep_ READ getTimeStep_ NOTIFY propertyChanged)
    QString getTimeStep_() {return QString("%1 s").arg(prms->InitialTimeStep,0,'e',1);}

    Q_PROPERTY(double in_SimulationTime READ getSimulationTime WRITE setSimulationTime NOTIFY propertyChanged)
    double getSimulationTime() {return prms->SimulationEndTime;}
    void setSimulationTime(double val) { prms->SimulationEndTime = val; }

    Q_PROPERTY(int in_UpdateEvery READ getUpdateEveryNthStep NOTIFY propertyChanged)
    int getUpdateEveryNthStep() {return prms->UpdateEveryNthStep;}

    Q_PROPERTY(bool in_SaveSnapshots READ getSaveSnapshots WRITE setSaveSnapshots NOTIFY propertyChanged)
    bool getSaveSnapshots() {return prms->SaveSnapshots;}
    void setSaveSnapshots(bool val) {prms->SaveSnapshots=val; }

    Q_PROPERTY(double in_ParticleViewSize READ getParticleViewSize WRITE setParticleViewSize NOTIFY propertyChanged)
    double getParticleViewSize() {return prms->ParticleViewSize;}
    void setParticleViewSize(double val) {prms->ParticleViewSize=val; Q_EMIT propertyChanged();}


    // parameters
    Q_PROPERTY(double p_YoungsModulus READ getYoungsModulus WRITE setYoungsModulus NOTIFY propertyChanged)
    double getYoungsModulus() {return prms->YoungsModulus;}
    void setYoungsModulus(double val) { prms->YoungsModulus = (float)val; prms->ComputeLame(); }

    Q_PROPERTY(QString p_YM READ getYM NOTIFY propertyChanged)
    QString getYM() {return QString("%1 Pa").arg(prms->YoungsModulus, 0, 'e', 2);}

    Q_PROPERTY(double p_PoissonsRatio READ getPoissonsRatio WRITE setPoissonsRatio NOTIFY propertyChanged)
    double getPoissonsRatio() {return prms->PoissonsRatio;}
    void setPoissonsRatio(double val) { prms->PoissonsRatio = (float)val; prms->ComputeLame(); }

    Q_PROPERTY(double p_LameLambda READ getLambda NOTIFY propertyChanged)
    double getLambda() {return prms->lambda;}

    Q_PROPERTY(double p_LameMu READ getMu NOTIFY propertyChanged)
    double getMu() {return prms->mu;}

    Q_PROPERTY(double p_LameKappa READ getKappa NOTIFY propertyChanged)
    double getKappa() {return prms->kappa;}


    // ice block
    Q_PROPERTY(int b_PtInitial READ getPointCountActual NOTIFY propertyChanged)
    int getPointCountActual() {return prms->nPtsInitial;}

    Q_PROPERTY(QString b_Grid READ getGridDimensions NOTIFY propertyChanged)
    QString getGridDimensions() {return QString("%1 x %2").arg(prms->GridXTotal).arg(prms->GridYTotal);}

    // Drucker-Prager
    Q_PROPERTY(double DP_threshold_p READ getDP_threshold_p WRITE setDP_threshold_p NOTIFY propertyChanged)
    double getDP_threshold_p() {return prms->DP_threshold_p;}
    void setDP_threshold_p(double val) {prms->DP_threshold_p = val;}

    Q_PROPERTY(double DP_phi READ getDPPhi NOTIFY propertyChanged)
    double getDPPhi() {return prms->DP_phi;}

    Q_PROPERTY(double ice_CompressiveStr READ getIce_CompressiveStr NOTIFY propertyChanged)
    double getIce_CompressiveStr() {return prms->IceCompressiveStrength;}

    Q_PROPERTY(double ice_TensileStr READ getIce_TensileStr NOTIFY propertyChanged)
    double getIce_TensileStr() {return prms->IceTensileStrength;}

    Q_PROPERTY(double ice_ShearStr READ getIce_ShearStr NOTIFY propertyChanged)
    double getIce_ShearStr() {return prms->IceShearStrength;}

    Q_PROPERTY(int tpb_P2G READ get_tpb_P2G WRITE set_tpb_P2G NOTIFY propertyChanged)
    int get_tpb_P2G() {return prms->tpb_P2G;}
    void set_tpb_P2G(int val) { prms->tpb_P2G = val; }

    Q_PROPERTY(int tpb_Upd READ get_tpb_Upd WRITE set_tpb_Upd NOTIFY propertyChanged)
    int get_tpb_Upd() {return prms->tpb_Upd;}
    void set_tpb_Upd(int val) { prms->tpb_Upd = val; }

    Q_PROPERTY(int tpb_G2P READ get_tpb_G2P WRITE set_tpb_G2P NOTIFY propertyChanged)
    int get_tpb_G2P() {return prms->tpb_G2P;}
    void set_tpb_G2P(int val) { prms->tpb_G2P = val; }


    Q_PROPERTY(QString span READ getSpan NOTIFY propertyChanged)
    QString getSpan() {
        double x = prms->cellsize*prms->InitializationImageSizeX*1e-3;
        double y = prms->cellsize*prms->InitializationImageSizeY*1e-3;
        return QString("%1 x %2 km").arg(x, 0, 'f', 1).arg(y, 0, 'f', 1);
    }

    Q_PROPERTY(double cellsize READ getCellsize NOTIFY propertyChanged)
    double getCellsize() {return prms->cellsize;}



public:
    ParamsWrapper(SimParams *p)
    {
        this->prms = p;
        Reset();
    }

    void Reset()
    {
        // it is possible to change parameters here
    }


};



#endif // PARAMETERS_WRAPPER_H
