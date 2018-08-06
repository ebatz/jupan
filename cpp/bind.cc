#include <iostream>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Cholesky>

#include <Minuit2/FCNBase.h>
#include <Minuit2/MnMinimize.h>
#include <Minuit2/MnUserParameters.h>
#include <Minuit2/FunctionMinimum.h>


namespace py = pybind11;
namespace Minu = ROOT::Minuit2;

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Eigen::LLT<Eigen::MatrixXd> cholCovBoot(const Eigen::Ref<MatrixXdR> theDat) {
	Eigen::MatrixXd centered = theDat.rowwise() - theDat.row(0);
	return ((centered.adjoint() * centered) / double(theDat.rows() - 2)).llt();

	//return aMat.llt().matrixL();
}

struct singleExp {

  int tMin, tMax;

  singleExp(int _tMin, int _tMax) : tMin(_tMin), tMax(_tMax) {}

  Eigen::VectorXd operator()(const std::vector<double>& fitPars) const {
	Eigen::VectorXd modelDat(tMax-tMin+1);

	for (int iT = tMin; iT <= tMax; ++iT)
		modelDat(iT-tMin) = fitPars[0] * exp(-fitPars[1]*iT);

	return modelDat;
  }

};

class CorrChiSq : public Minu::FCNBase {

  private:

    Eigen::LLT<Eigen::MatrixXd> covMat;
    MatrixXdR theDat;
    std::function<Eigen::VectorXd(const std::vector<double>&)> modelFunc;
    size_t iCfg;

  public:

    CorrChiSq(
	Eigen::Ref<MatrixXdR> _theDat,
        std::function<Eigen::VectorXd(const std::vector<double>&)> _modelFunc,
        std::function<Eigen::LLT<Eigen::MatrixXd>(Eigen::Ref<MatrixXdR>)> _covFunc
    ) : theDat(_theDat), modelFunc(_modelFunc), iCfg(0) {

	covMat = _covFunc(_theDat);

    }

    void setCfg(size_t _iCfg) { iCfg = _iCfg; }

    double Up() const { return 0.5; }

    double operator()(const std::vector<double>& fitPar) const {

	Eigen::VectorXd modelDat = modelFunc(fitPar);

	int dof = theDat.row(iCfg).cols() - fitPar.size();
	Eigen::VectorXd residue = theDat.row(iCfg).transpose() - modelDat;
	Eigen::MatrixXd chol = covMat.matrixL();
	Eigen::VectorXd solVec = chol.triangularView<Eigen::Lower>().solve(residue);
	return solVec.squaredNorm() / dof;
    }

};


std::vector<std::vector<double>> doFit(
	  Eigen::Ref<MatrixXdR> _theDat,
	  std::function<Eigen::VectorXd(const std::vector<double>&)> _modelFunc,
	  std::function<Eigen::LLT<Eigen::MatrixXd>(Eigen::Ref<MatrixXdR>)> _covFunc,
	  const std::vector<double>& _initGuess) {

    // reserve space for return structure
  size_t nCfg = _theDat.rows();
  std::vector< std::vector<double> > retDat;
  retDat.reserve(nCfg);

  std::vector<double> initGuess(_initGuess);
  std::vector<double> initStep(initGuess.size(), 0.03);

  Minu::CombinedMinimizer mnmr;

  CorrChiSq metFunc(_theDat, _modelFunc, _covFunc);

    // now do minimizations
  for (size_t iCfg = 0; iCfg < nCfg; ++iCfg) {
    metFunc.setCfg(iCfg);

    auto fMin = mnmr.Minimize(metFunc, initGuess, initStep);

    std::vector<double> aRet;
    aRet.push_back(fMin.Fval());

    for (size_t iPar = 0; iPar < initGuess.size(); ++iPar) {
      aRet.push_back(fMin.UserParameters().Value(iPar));
      if (iCfg == 0) {
        initGuess[iPar] = fMin.UserParameters().Value(iPar);
	initStep[iPar] = 0.02;
      }
    }

    retDat.push_back(aRet);
  }

  return retDat;
}



PYBIND11_MODULE(cppana, m) {
	    m.doc() = "pybind11 example plugin"; // optional module docstring
	    m.def("bootSingleExp",
	            [](Eigen::Ref<MatrixXdR> _theDat, int tMin, int tMax, const std::vector<double>& initGuess) {
		        return doFit(
			    _theDat,
			    singleExp(tMin,tMax),
			    &cholCovBoot,
			    initGuess
			    );
			});
}
