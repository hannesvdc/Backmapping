#ifndef __CONSTRAINED_SAMPLER_H__
#define __CONSTRAINED_SAMPLER_H__

#include "OpenMM.h"
#include <vector>

class ConstrainedSampler {
public:
	std::vector<std::vector<OpenMM::Vec3> > sampleManifold(const double& z, const std::vector<OpenMM::Vec3>& init, int K, std::vector<double>& energies ) {
		std::vector<OpenMM::Vec3> x;
		double lam = positionConstraint(init, rc->gradient(init), z, x);
		context->setPositions(x);
		OpenMM::State s = context->getState(OpenMM::State::Forces|OpenMM::State::Energy);

		// The rattle scheme.
		for ( int k = 0; k < K; ++k ) {
			std::vector<OpenMM::Vec3> xp = x + stepSize*s.getForces() + sqrt(2*stepSize/beta)*toVec3Vector(rng(gen));
			lam = positionConstraint(xp, rc->gradient(xp), z, xp);
			context.setPositions(xp);
			OpenMM::State sp = context.getState(OpenMM::State::Forces|OpenMM:::State::Energy);

			double lnalpha = -beta*(sp.getPotentialEnergy() - s.getPotentialEnergy()) -beta/4.0*()
		}
	}

private:
	double positionConstraint(const VectorXd y,
							 const VectorXd& dxi,
							 const double& z,
							 VectorXd& x) {
		x = y;
		double lam = 0.;
		int k = 0;
		double w = _xi(toVec3Vector(x));
		VectorXd temp_grad = dxi;

		while(std::abs(w-z) > _tol && k < _max_iterations){
			double denom = temp_grad.dot(dxi);
			lam = lam - (w-z)/denom;

			// Update x and w
			x = y + dxi*lam;
			w = _xi(toVec3Vector(x));
			temp_grad = toVectorXd(_dxi(toVec3Vector(x)));
			++k;
		}

		if( k == _max_iterations) {
			std::cout << "Position constraint failed! " << z << " " << w << std::endl;
			return std::nan("");
		} else {
			return lam;
		}
	}

	OpenMM::ReactionCoordinate *rc;
};


#endif //__CONSTRAINED_SAMPLER_H__
