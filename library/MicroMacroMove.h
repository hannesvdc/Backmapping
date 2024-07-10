#ifndef __mMMoveH__
#define __mMMoveH__

#include <iostream>
#include <random>

#include "MCMCMove.h"
#include "OpenMM.h"

void print_v(const std::vector<double>& x) {
	for(int i = 0; i < x.size(); ++i) {
		std::cout << x[i] << ' ';
	}
	std::cout << std::endl;
}

void print_v3(const std::vector<OpenMM::Vec3>& x) {
	for(int i = 0; i < x.size(); ++i) {
		std::cout << x[i] << ' ';
	}
	std::cout << std::endl;
}

std::vector<OpenMM::Vec3> operator-(const std::vector<OpenMM::Vec3>& a, const std::vector<OpenMM::Vec3>& b) {
    std::vector<OpenMM::Vec3> res(a.size(), OpenMM::Vec3(0., 0., 0.));
    for ( int n = 0; n < a.size(); ++n ) {
        res[n] = a[n] - b[n];
    }

    return res;
}

std::vector<OpenMM::Vec3> operator+(const std::vector<OpenMM::Vec3>& a, const std::vector<OpenMM::Vec3>& b) {
    std::vector<OpenMM::Vec3> res(a.size(), OpenMM::Vec3(0., 0., 0.));
    for ( int n = 0; n < a.size(); ++n ) {
        res[n] = a[n] + b[n];
    }

    return res;
}

std::vector<OpenMM::Vec3> operator*(const double a, const std::vector<OpenMM::Vec3>& b) {
    std::vector<OpenMM::Vec3> res(b.size(), OpenMM::Vec3(0., 0., 0.));
    for ( int n  = 0; n < b.size(); ++n ) {
        res[n] = a*b[n];
    }

    return res;
}

std::vector<OpenMM::Vec3> operator*(const std::vector<OpenMM::Vec3>& a, const double& b) {
    std::vector<OpenMM::Vec3> res(a.size(), OpenMM::Vec3(0., 0., 0.));
    for ( int n  = 0; n < a.size(); ++n ) {
        res[n] = a[n]*b;
    }

    return res;
}

double operator*(const std::vector<OpenMM::Vec3>& a, const std::vector<OpenMM::Vec3>& b)  {
    double sum = 0.;
    for ( int n = 0; n < a.size(); ++n ) {
        sum += a[n].dot(b[n]);
    }

    return sum;
}

std::vector<OpenMM::Vec3> operator/(const std::vector<OpenMM::Vec3>& a, const double &x) {
	std::vector<OpenMM::Vec3> res(a.size(), OpenMM::Vec3(0., 0., 0.));
	for ( int i = 0; i < a.size(); ++i ) {
		res[i] = a[i]/x;
	}
	return res;
}

double norm(const std::vector<OpenMM::Vec3>& x) {
	double s = 0.0;
	for ( int i = 0; i < x.size(); ++i ) {
		s += x[i].dot(x[i]);
	}
	return sqrt(s);
}

struct ReconstructionData {
	std::vector<std::vector<OpenMM::Vec3> > positions;
	std::vector<double> energies;
	OpenMM::State xp;
	OpenMM::State zp;
};

template<class T, class S>
class MicroMacroMove: public MCMCMove {
public:
	MicroMacroMove(OpenMM::Context& _microContext,
		           OpenMM::Context& _macroContext,
		           double temperature,
		           int K,
		           double lambda,
		           double stepSize,
		           OpenMM::ReactionCoordinate* rc) :
				   microContext(_microContext),
				   macroContext(_macroContext),
				   beta(1000./(temperature*8.314459)),
				   K(K),
				   lambda(lambda),
				   stepSize(stepSize),
				   macroIntegrator(dynamic_cast<T&>(macroContext.getIntegrator())),
				   microIntegrator(dynamic_cast<S&>(microContext.getIntegrator())),
				   _rc(rc),
				   nSamples(0),
				   nMacroAccepted(0),
				   nmicroAccepted(0),
				   gen(std::mt19937(rd())),
				   rng(std::uniform_real_distribution<>(0., 1.)){
	}

	virtual std::vector<OpenMM::Vec3> step() {
		nSamples +=1;

		OpenMM::State z = macroContext.getState(OpenMM::State::Positions|OpenMM::State::Energy);
		macroIntegrator.step(1);
		OpenMM::State zp = macroContext.getState(OpenMM::State::Positions|OpenMM::State::Energy);
		
		double u = rng(gen);
		double lnalpha = -beta*(zp.getPotentialEnergy() - z.getPotentialEnergy());
		if( log(u) > lnalpha ) {
			macroContext.setState(z);
			macroIntegrator.accepted(false);
			return microContext.getState(OpenMM::State::Positions).getPositions();
		}
		nMacroAccepted += 1;
		macroIntegrator.accepted(true);

		OpenMM::State x = microContext.getState(OpenMM::State::Positions|OpenMM::State::Energy|OpenMM::State::Forces);
		microIntegrator.setMacroscopicVariable(zp.getPositions());
		ReconstructionData recData = reconstruct(zp.getPositions());
		recData.zp = zp;
		
		nmicroAccepted += 1;
		microIntegrator.accepted(true);
		
		return recData.xp.getPositions();
	}

	ReconstructionData reconstruct(const std::vector<OpenMM::Vec3>& z) {
		OpenMM::State pre_state = microContext.getState(OpenMM::State::Positions | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Velocities);
		std::vector<OpenMM::Vec3> x = pre_state.getPositions();
		std::vector<OpenMM::Vec3> pre_forces = pre_state.getForces();
		double pre_E = pre_state.getPotentialEnergy();
		std::vector<OpenMM::Vec3> pre_val = _rc->value(pre_state.getPositions());
		std::vector<OpenMM::Vec3> pre_grad = _rc->gradMatMul(x, pre_val - z);

		nRecAccepted = 0;
		ReconstructionData recData;
		for(int k = 1; k <= K; ++k) {
			if ( k % 1000 == 0 ) 
				std::cout << "k = " << k << std::endl;
			microIntegrator.step(1);

			OpenMM::State new_state = microContext.getState(OpenMM::State::Positions | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Velocities);
			std::vector<OpenMM::Vec3> xp = new_state.getPositions();
			std::vector<OpenMM::Vec3> new_forces = new_state.getForces();
			double new_E = new_state.getPotentialEnergy();
			std::vector<OpenMM::Vec3> new_val = _rc->value(new_state.getPositions());
			std::vector<OpenMM::Vec3> new_grad = _rc->gradMatMul(xp, new_val - z);

			double lnalpha = -beta*(new_E - pre_E + _rc->getBiasedEnergy(xp, z) - _rc->getBiasedEnergy(x, z))
			                 -beta/4.*((-1)*new_forces -pre_forces + lambda*(new_grad+pre_grad)) * (2.*x - 2.*xp + stepSize*((-1)*new_forces + pre_forces + lambda*(new_grad-pre_grad)));
			double u = rng(gen);
			
			if (log(u) <= lnalpha ) {
				nRecAccepted++;
				x = xp;
				pre_forces = new_forces;
				pre_E = new_E;
				pre_val = new_val;
				pre_grad = new_grad;
				pre_state = new_state;
			} else {
				microContext.setState(pre_state);
			}

			recData.positions.push_back(x);
			recData.energies.push_back(pre_E);
		}
		recData.xp = pre_state;

		return recData;
	}

	void setLambda(double lam ) {
		lambda = lam;
	}

	void setStepSize(double s) {
		stepSize = s;
	}

	void setK(int _k) {
		K = _k;
	}

	double getMacroAcceptanceRate() const {
		return ((double) nMacroAccepted)/((double) nSamples);
	}

	double getMicroAcceptanceRate() const {
		return ((double) nmicroAccepted)/((double) nMacroAccepted);	
	}

	double getReconstructionAcceptanceRate() const {
		return ((double) nRecAccepted) / ((double) K);
	}

	void print(const std::vector<OpenMM::Vec3>& pos) const {
		for (int i = 0; i < pos.size(); ++i) {
			std::cout << "[" << pos[i][0] << " " << pos[i][1] << " " << pos[i][2] << "] ";
		}
		std::cout << std::endl;
	}

	std::vector<OpenMM::Vec3> minus(const std::vector<OpenMM::Vec3>& x, const std::vector<OpenMM::Vec3>& y ) const {
		std::vector<OpenMM::Vec3> w(x.size(), OpenMM::Vec3(0., 0., 0.));
		for ( int i = 0; i < x.size(); ++i ) {
			w[i] = x[i]-y[i];
		}
		return w;
	}
private:
	OpenMM::Context &microContext, &macroContext;
	T& macroIntegrator;
	S& microIntegrator;
	double beta;
	int K;
	double lambda;
	double stepSize;
	OpenMM::ReactionCoordinate *_rc;
	int nSamples, nMacroAccepted, nmicroAccepted, nRecAccepted;

	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> rng;
};

#endif //__mMMoveH__