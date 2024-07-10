#ifndef __mMGIMHMoveH__
#define __mMGIMHMoveH__

#include <iostream>
#include <random>

#include "MCMCMove.h"
#include "OpenMM.h"

// std::vector<OpenMM::Vec3> operator-(const std::vector<OpenMM::Vec3>& a, const std::vector<OpenMM::Vec3>& b) {
//     std::vector<OpenMM::Vec3> res(a.size(), OpenMM::Vec3(0., 0., 0.));
//     for ( int n = 0; n < a.size(); ++n ) {
//         res[n] = a[n] - b[n];
//     }

//     return res;
// }

/*std::vector<OpenMM::Vec3> operator+(const std::vector<OpenMM::Vec3>& a, const std::vector<OpenMM::Vec3>& b) {
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
}*/

std::vector<double> operator*(const double& a, const std::vector<double>& x) {
	std::vector<double> b(x.size(), 0.0);
	for ( int i = 0; i < x.size(); ++i ) {
		b[i] = a*x[i];
	}
	return b;
}

std::vector<OpenMM::Vec3> operator/(const std::vector<OpenMM::Vec3> &v, const std::vector<double> &m ) {
	std::vector<OpenMM::Vec3> res(v.size(), OpenMM::Vec3(0.0, 0.0, 0.0));
	for ( int i = 0; i < v.size(); ++i ) {
		res[i][0] = v[i][0]/m[i];
		res[i][1] = v[i][1]/m[i];
		res[i][2] = v[i][2]/m[i];
	}
	return res;
}

/*double norm(const std::vector<OpenMM::Vec3>& x) {
	double s = 0.0;
	for ( int i = 0; i < x.size(); ++i ) {
		s += x[i].dot(x[i]);
	}
	return sqrt(s);
}*/

struct ReconstructionDataGIMH {
	std::vector<std::vector<OpenMM::Vec3> > positions;
	std::vector<double> energies;
	OpenMM::State xp;
	OpenMM::State zp;
	double lna;
};

template<class T, class S>
class MicroMacroGIMHMove: public MCMCMove {
public:
	MicroMacroGIMHMove(OpenMM::Context& _microContext,
		           OpenMM::Context& _macroContext,
		           OpenMM::System *system,
		           double temperature,
		           int K,
		           double lambda,
		           double stepSize,
		           OpenMM::ReactionCoordinate* rc,
		           double bins) :
				   microContext(_microContext),
				   macroContext(_macroContext),
				   microSystem(system),
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
				   acceptableRec(0),
				   gen(std::mt19937(rd())),
				   rng(std::uniform_real_distribution<>(0., 1.)),
				   rng_normal(std::normal_distribution<>(0.0, 1.0)),
				   binsize(bins),
				   FEprev(0.0) {
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
		

		OpenMM::State x = microContext.getState(OpenMM::State::Positions|OpenMM::State::Energy|OpenMM::State::Forces);
		microIntegrator.setMacroscopicVariable(zp.getPositions());
		ReconstructionDataGIMH recData = reconstruct(zp.getPositions());
		recData.zp = zp;
		double FE = computeFreeEnergy(recData.positions, recData.energies);

		lnalpha = -beta*(FE - FEprev) + beta*(zp.getPotentialEnergy() - z.getPotentialEnergy());
		std::cout << lnalpha << " " << recData.lna << std::endl;
		if ( recData.lna < 1000.0 && recData.lna > -100.0 ) {
			acceptableRec += 1;
		}
		if ( isnan(lnalpha) ) {
			std::cout << "nan: " << FE << " " << FEprev << std::endl;
		}

		//std::cout << lnalpha << " " << -beta*(FE - FEprev) << " " << zp.getPotentialEnergy() - z.getPotentialEnergy() << std::endl;
		//if ( abs(z.getPositions()[0][0] - zp.getPositions()[0][0]) >= 2.0 ) {
		//	std::cout << z.getPositions()[0][0] << " " << zp.getPositions()[0][0] << " " << lnalpha << std::endl;
		//}
		//u = rng(gen);
		//if ( log(u) <= lnalpha ) {
		nmicroAccepted += 1;
		FEprev = FE;
		//} else {
		//	macroContext.setState(z);
		//	microContext.setState(x);
		//}
		
		return recData.xp.getPositions();
	}

	ReconstructionDataGIMH reconstruct(const std::vector<OpenMM::Vec3>& z) {
		OpenMM::State s = microContext.getState(OpenMM::State::Positions | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Velocities);
		std::vector<OpenMM::Vec3> x = s.getPositions();
		std::vector<OpenMM::Vec3> v = s.getVelocities();
		std::vector<OpenMM::Vec3> f = s.getForces();
		double E = s.getPotentialEnergy();
		double Ke = s.getKineticEnergy();

		nRecAccepted = 0;
		ReconstructionDataGIMH recData;
		//std::vector<double> xivals;
		int nSteps = 2*K;
		double avgLn = 0.0;
		for ( int k = 1; k <= nSteps; ++k ) {
			double alpha = std::min(1.0, (1.0*k)/K);
			// Sample normally distributed velocities
			for ( int i = 0; i < microSystem->getNumParticles(); ++i ) {
				v[i][0] = rng_normal(gen)/sqrt(beta*microSystem->getParticleMass(i));
				v[i][1] = rng_normal(gen)/sqrt(beta*microSystem->getParticleMass(i));
				v[i][2] = rng_normal(gen)/sqrt(beta*microSystem->getParticleMass(i));
			}
			microContext.setVelocities(v);
			OpenMM::State s = microContext.getState(OpenMM::State::Positions | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Velocities);
			Ke = s.getKineticEnergy();
			
			OpenMM::State sp = leapfrog(s, z, alpha);
			double Ep = sp.getPotentialEnergy();
			double Kep = sp.getKineticEnergy();

			// Accept-Reject
			double lnalpha = -beta*(alpha*Ep - alpha*E + Kep - Ke + _rc->getBiasedEnergy(sp.getPositions(), z) - _rc->getBiasedEnergy(s.getPositions(), z));
			avgLn += lnalpha/nSteps;
			//std::cout << "recln " << lnalpha << std::endl;
			if ( log(rng(gen)) <= lnalpha ) {
				nRecAccepted++;
				s = sp;
				E = Ep;
				Ke = Kep;
			} else {
				microContext.setState(s);
			}

			if ( k >= K ) {
				//xivals.push_back(_rc->value(s.getPositions())[0][0]);
				recData.positions.push_back(s.getPositions());
				recData.energies.push_back(E + _rc->getBiasedEnergy(s.getPositions(), z));
			}
		}
		std::vector<OpenMM::Vec3> xi = _rc->value(s.getPositions());
		//std::cout << xi[0][0] << " " << z[0][0] << " | " << xi[0][1] << " " << z[0][1] << std::endl;
		recData.xp = s;
		recData.lna = avgLn;

		//std::cout << "z = " <<  z[0][0] << std::endl;
		//print_v(xivals);

		return recData;
	}

	OpenMM::State leapfrog(const OpenMM::State &s, const std::vector<OpenMM::Vec3> &z, const double &alpha) {
		// Initial velocity update
		std::vector<OpenMM::Vec3> x = s.getPositions();
		std::vector<OpenMM::Vec3> vp(microSystem->getNumParticles(), OpenMM::Vec3(0.0, 0.0, 0.0));
		std::vector<OpenMM::Vec3> dxi = _rc->gradMatMul(x, _rc->value(x)-z);
		for ( int i = 0; i < microSystem->getNumParticles(); ++i ) {
			vp[i] = s.getVelocities()[i] + 0.5*stepSize*(alpha*s.getForces()[i] - dxi[i])/microSystem->getParticleMass(i); //
		}

		// Final position update.
		std::vector<OpenMM::Vec3> xp(microSystem->getNumParticles(), OpenMM::Vec3(0.0, 0.0, 0.0));
		for ( int i = 0; i < microSystem->getNumParticles(); ++i) {
			xp[i] = x[i] + stepSize*vp[i];
		}
		microContext.setPositions(xp);
		OpenMM::State sp = microContext.getState(OpenMM::State::Positions | OpenMM::State::Velocities | OpenMM::State::Forces | OpenMM::State::Energy);

		// Final velocity update.
		dxi = _rc->gradMatMul(xp, _rc->value(xp)-z);
		for ( int i = 0; i < microSystem->getNumParticles(); ++i ) {
			vp[i] = vp[i] + 0.5*stepSize*(alpha*sp.getForces()[i] - dxi[i])/microSystem->getParticleMass(i); //
		}
		microContext.setVelocities(vp);

		return microContext.getState(OpenMM::State::Positions | OpenMM::State::Velocities | OpenMM::State::Forces | OpenMM::State::Energy);
	}

	double computeFreeEnergy(const std::vector<std::vector<OpenMM::Vec3> >& pos, const std::vector<double> &energies) const {
		std::vector<int> counts(pos.size(), 0);
		int c_index=0;
		std::vector<OpenMM::Vec3> c_pos = pos[0];
		int count = 1;
		for ( int i = 1; i < pos.size(); ++i ) {
			if ( pos[i] == c_pos ) {
				count++;
				continue;
			}

			for ( int j = c_index; j < i; ++j ) {
				counts[j] = count;
			}
			c_index = i;
			count = 1;
		}
		for ( int j = c_index; j < pos.size(); ++j ) {
			counts[j] = count;
		}

		double is_sum = 0.0;
		for ( int k = 0; k < K; ++k ) {
			//std::cout << "en " << -beta*energies[k] << std::endl;
			is_sum += exp(-beta*energies[k]+500.0)/(K*counts[k]);
		}

		return -1.0/beta*log(is_sum);
	}


	// ReconstructionData reconstruct(const std::vector<OpenMM::Vec3>& z) {
	// 	OpenMM::State pre_state = microContext.getState(OpenMM::State::Positions | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Velocities);
	// 	std::vector<OpenMM::Vec3> x = pre_state.getPositions();
	// 	std::vector<OpenMM::Vec3> pre_forces = pre_state.getForces();
	// 	double pre_E = pre_state.getPotentialEnergy();
	// 	std::vector<OpenMM::Vec3> pre_val = _rc->value(pre_state.getPositions());
	// 	std::vector<OpenMM::Vec3> pre_grad = _rc->gradMatMul(x, pre_val - z);

	// 	nRecAccepted = 0;
	// 	ReconstructionData recData;
	// 	for(int k = 1; k <= K; ++k) {
	// 		if ( k % 1000 == 0 ) 
	// 			std::cout << "k = " << k << std::endl;
	// 		microIntegrator.step(1);

	// 		OpenMM::State new_state = microContext.getState(OpenMM::State::Positions | OpenMM::State::Energy | OpenMM::State::Forces | OpenMM::State::Velocities);
	// 		std::vector<OpenMM::Vec3> xp = new_state.getPositions();
	// 		std::vector<OpenMM::Vec3> new_forces = new_state.getForces();
	// 		double new_E = new_state.getPotentialEnergy();
	// 		std::vector<OpenMM::Vec3> new_val = _rc->value(new_state.getPositions());
	// 		std::vector<OpenMM::Vec3> new_grad = _rc->gradMatMul(xp, new_val - z);

	// 		double lnalpha = -beta*(new_E - pre_E + _rc->getBiasedEnergy(xp, z) - _rc->getBiasedEnergy(x, z))
	// 		                 -beta/4.*((-1)*new_forces -pre_forces + lambda*(new_grad+pre_grad)) * (2.*x - 2.*xp + stepSize*((-1)*new_forces + pre_forces + lambda*(new_grad-pre_grad)));
	// 		double u = rng(gen);
	// 		//std::cout << lnalpha << std::endl;
	// 		if (log(u) <= lnalpha ) {
	// 			nRecAccepted++;
	// 			x = xp;
	// 			pre_forces = new_forces;
	// 			pre_E = new_E;
	// 			pre_val = new_val;
	// 			pre_grad = new_grad;
	// 			pre_state = new_state;
	// 		} else {
	// 			microContext.setState(pre_state);
	// 		}

	// 		recData.positions.push_back(x);
	// 		recData.energies.push_back(pre_E + _rc->getBiasedEnergy(x, z));
	// 	}
	// 	recData.xp = pre_state;

	// 	return recData;
	// }

	/*double computeFreeEnergy(const std::vector<std::vector<OpenMM::Vec3> > &positions, const std::vector<double> &energies) const {
		std::vector<double> exp_energies = (-beta)*energies;
		int ndim = positions[0].size();

		for ( int dim = 0; dim < ndim; ++dim ) {
			std::vector<double> visited(positions.size(), false);

			for ( int i = 0; i < positions.size(); ++i ) {
				if ( visited[i] ) continue;
				visited[i] = true;
				OpenMM::Vec3 y = grid_point(positions[i][dim]);
				std::vector<size_t> indices(1, i);

				for ( int j = i+1; j < positions.size(); ++j ) {
					if ( !visited[j] && in_3dbox(y, positions[j][dim]) ) {
						visited[j] = true;
						indices.push_back(j);
					}
				}

				for ( int k = 0; k < indices.size(); ++k ) {
					size_t idx = indices[k];
					exp_energies[idx] = exp_energies[idx] - (log((1.0*indices.size())/K) - 3*log(binsize));
				}
			}
		}

		double is_sum = 0.0;
		for ( int k = 0; k < K; ++k ) {
			//std::cout << "exp " << exp_energies[k]+160. << std::endl;
			is_sum += exp(exp_energies[k]+160.0)/K;
		}

	 	return -1.0/beta*log(is_sum);
	}*/


	OpenMM::Vec3 grid_point(const OpenMM::Vec3 &x) const {
		return OpenMM::Vec3(x[0] - std::fmod(x[0], binsize), x[1]-std::fmod(x[1], binsize), x[2]-std::fmod(x[2], binsize));
	}

	bool in_3dbox(const OpenMM::Vec3 &grid, const OpenMM::Vec3 &x) const {
		if ( x[0] < grid[0] || x[0] > grid[0]+binsize ) return false;
		if ( x[1] < grid[1] || x[1] > grid[1]+binsize ) return false;
		if ( x[2] < grid[2] || x[2] > grid[2]+binsize ) return false;
		return true;
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

	double getAcceptableReconstructionRate() const {
		return ((double) acceptableRec) / ((double) nSamples);
	}

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

private:
	OpenMM::Context &microContext, &macroContext;
	OpenMM::System *microSystem;
	T& macroIntegrator;
	S& microIntegrator;
	double beta;
	int K;
	double lambda;
	double stepSize;
	OpenMM::ReactionCoordinate *_rc;
	int nSamples, nMacroAccepted, nmicroAccepted, nRecAccepted, acceptableRec;
	double binsize;
	double FEprev;

	std::random_device rd;
	std::mt19937 gen;
	std::uniform_real_distribution<> rng;
	std::normal_distribution<> rng_normal;
};

#endif //__mMGIMHMoveH_