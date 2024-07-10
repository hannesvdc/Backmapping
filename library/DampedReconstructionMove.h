#ifndef __DampedMoveH__
#define __DampedMoveH__

#include <iostream>
#include <random>
#include <algorithm>

#include "MCMCMove.h"
#include "MicroMacroMove.h"
#include "OpenMM.h"

template<class T, class S>
class DampedReconstructionMove: public MCMCMove {
public:
	DampedReconstructionMove(OpenMM::Context& _microContext,
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
				   rng(std::uniform_real_distribution<>(0., 1.)) {
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
		macroIntegrator.accepted(true);
		nMacroAccepted += 1;

		microIntegrator.setMacroscopicVariable(zp.getPositions());
		OpenMM::State xp = reconstruct(zp.getPositions());
		microIntegrator.accepted(true);
		
		return xp.getPositions();
	}

	OpenMM::State reconstruct(const std::vector<OpenMM::Vec3>& z) {
		OpenMM::State pre_state = microContext.getState(OpenMM::State::Positions | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Velocities);
		std::vector<OpenMM::Vec3> x = pre_state.getPositions();
		std::vector<OpenMM::Vec3> pre_forces = pre_state.getForces();
		double pre_E = pre_state.getPotentialEnergy();
		std::vector<OpenMM::Vec3> pre_val = _rc->value(pre_state.getPositions());
		std::vector<OpenMM::Vec3> pre_grad = _rc->gradMatMul(x, pre_val - z);

		nRecAccepted = 0;
		int nSteps = 2*K;
		for(int k = 0; k < nSteps; ++k) {
			if ( k % 1000 == 0 && k > 0 ) 
				std::cout << "k = " << k << std::endl;
			double gamma = std::min<double>(((double)k)/this->K, 1.0);
			microIntegrator.step(1);

			OpenMM::State new_state = microContext.getState(OpenMM::State::Positions | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Velocities);
			std::vector<OpenMM::Vec3> xp = new_state.getPositions();
			std::vector<OpenMM::Vec3> new_forces = new_state.getForces();
			double new_E = new_state.getPotentialEnergy();
			std::vector<OpenMM::Vec3> new_val = _rc->value(new_state.getPositions());
			std::vector<OpenMM::Vec3> new_grad = _rc->gradMatMul(xp, new_val - z);

			bool keep = true;
			if ( k >= this->K ) {
				double lnalpha = -beta*(new_E - pre_E)
				                 -beta/4.*((-1)*new_forces -pre_forces) * (2.*x - 2.*xp + stepSize*((-1)*new_forces + pre_forces));
				double u = rng(gen);
				
				if (log(u) <= lnalpha ) {
					nRecAccepted++;
				} else {
					microContext.setState(pre_state);
					keep = false;
				}
			}

			if ( keep ) {
				x = xp;
				pre_forces = new_forces;
				pre_E = new_E;
				pre_val = new_val;
				pre_grad = new_grad;
				pre_state = new_state;
			}

		}
		return pre_state;
	}

private:
	OpenMM::Context &microContext, &macroContext;
	T& macroIntegrator;
	S& microIntegrator;
	double beta;
	int K;
	double lambda;
	double stepSize;
	double binsize;
	double Nprev;
	OpenMM::ReactionCoordinate *_rc;
	int nSamples, nMacroAccepted, nmicroAccepted, nRecAccepted;
	bool with_acc, with_gimh;

	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> rng;
};


#endif