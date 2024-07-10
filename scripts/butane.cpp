#include "OpenMM.h"

#include <vector>
#include <cmath>

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

double operator*(const std::vector<OpenMM::Vec3>& a, const std::vector<OpenMM::Vec3>& b) {
	double sum = 0.;
	for ( int n = 0; n < a.size(); ++n ) {
		sum += a[n].dot(b[n]);
	}

	return sum;
}

class ButaneTorsion : public OpenMM::ReactionCoordinate {
public:
	ButaneTorsion() {}

	virtual std::vector<OpenMM::Vec3> value(const std::vector<OpenMM::Vec3>& x) {
		OpenMM::Vec3 b1 = x[1] - x[0];
		OpenMM::Vec3 b2 = x[2] - x[1];
		OpenMM::Vec3 b3 = x[3] - x[2];

		OpenMM::Vec3 n1 = b1.cross(b2);
		n1 = n1/sqrt(n1.dot(n1));
		OpenMM::Vec3 n2 = b2.cross(b3);
		n2 = n2/sqrt(n2.dot(n2));
		OpenMM::Vec3 m = n1.cross(b2/sqrt(b2.dot(b2)));

		double x = n1.dot(n2);
		double y = m.dot(n2);
		double phi = atan2(-y, -x);

		return std::vector<OpenMM::Vec3>(1, OpenMM::Vec3(phi, 0., 0.));
	}
};

OpenMM::System createButaneSystem() {
	double kr = 1.17*10^6;
	double r0 = 1.53;
	double kt = 62500.;
	double theta0 = 112*3.1415/180.;
	double c0 = 1031.36;
	double c1 = 2037.82;
	double c2 = 158.52;
	double c3 = -3227.7;

	OpenMM::System system;
	OpenMM::CustomBondForce *bondForce = new OpenMM::CustomBondForce("0.5*k*(r - r_0)^2");
	bondForce->addPerBondParameter("k");
	bondForce->addPerBondParameter("r_0");
	bondForce->addBond(0, 1, std::vector<double>({kr, r0}));
    bondForce->addBond(1, 2, std::vector<double>({kr, r0}));
    bondForce->addBond(2, 3, std::vector<double>({kr, r0}));

    OpenMM::CustomAngleForce *angleForce = new OpenMM::CustomAngleForce("0.5*k*(theta - theta0)^2");
    angleForce->addPerAngleParameter("k");
    angleForce->addPerAngleParameter("theta0");
    angleForce->addAngle(0, 1, 2, std::vector<double>({kt, theta0}));
    angleForce->addAngle(1, 2, 3, std::vector<double>({kt, theta0}));

    OpenMM::CustomTorsionForce *torsionForce = new CustomTorsionForce("c0 + c1*cos(theta) + c2*cos(theta)^2 + c3*cos(theta)^3");
    torsionForce->addPerTorsionParameter("c0");
    torsionForce->addPerTorsionParameter("c1");
    torsionForce->addPerTorsionParameter("c2");
    torsionForce->addPerTorsionParameter("c3");
    torsionForce->addTorsion(0, 1, 2, 3, std::vector<double>({c0, c1, c2, c3}));

    system.addParticle(1.);
    system.addParticle(1.);
    system.addParticle(1.);
    system.addParticle(1.);

    return system;
}

void simulateButaneMicroMacro() {
	// Load any shared libraries containing GPU implementations.
	OpenMM::Platform::loadPluginsFromDirectory(
	    OpenMM::Platform::getDefaultPluginsDirectory());

	OpenMM::System microSystem = createButaneSystem();
	ButaneTorsion *rc = new ButaneTorsion();
	OpenMM::System macroSystem = rc->getTorsionSystem();

	double temperature = 100.;
	std::vector<OpenMM::Vec3> initPos{OpenMM::Vec3(0., 0., 0.), OpenMM::Vec3(1.5, 0., 0.), OpenMM::Vec3(3., 0., 0.), OpenMM::Vec3(4.5, 0., 0.)};
    std::vector<OpenMM::Vec3> initPosTorsion = rc->value(initPos);

    double macroStepSize = 0.0005;
    double microStepSize = 1.e-06;
    double lambda = 1.e6;
    OpenMM::RandomWalkIntegrator macroIntegrator(temperature, macroStepSize);
    OpenMM::IndirectReconstructionIntegrator microIntegrator(temperature, lambda, stepSize, rc);
    OpenMM::Context macroContext(macroSystem, macroIntegrator, OpenMM::Platform::getPlatformByName("Reference"));
    OpenMM::Context microContext(microSystem, microIntegrator, OpenMM::Platform::getPlatformByName("Reference"));
    macroContext.setPositions(initPosTorsion);
    microContext.setPositions(initPos);
    microIntegrator.setMacroscopicVariable(initPosTorsion);
    macroIntegrator.setupSampler();
    microIntegrator.setupSampler();

    printf( "REMARK  Using OpenMM platform %s\n", microContext.getPlatform().getName().c_str() );
	std::cout << "Setup Done" << std::endl;

	MicroMacroMove<OpenMM::RandomWalkIntegrator, OpenMM::IndirectReconstructionIntegrator> move(microContext, macroContext, temperature, K, lambda, stepSize, rc);
	MCMCSampler sampler(move);
	int N = 1000000;
	std::vector<std::vector<OpenMM::Vec3> > samples = sampler.sample(N);
    std::cout << "Sampling Done" << std::endl;
}

int main() {
	try {
		simulateButaneMicroMacro();
		return 0;
	} catch (const std::exception& e) {
		printf("EXCEPTION: %s\n", e.what());
		return 1.;
	}
}