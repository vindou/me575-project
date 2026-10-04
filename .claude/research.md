
Attitude-Control Design Basis for a Precision-Pointing 6U CubeSat Simulation
Research landscape

The literature relevant to your project falls into four fairly distinct groups: small-spacecraft GNC surveys and agency state-of-the-art reports; flight papers from precision-pointing missions such as ASTERIA; detailed design/simulation studies; and controller-specific papers. The most useful survey-level sources are Burton et al.'s small-satellite GNC survey and NASA's continuously updated State-of-the-Art of Small Spacecraft Technology. The latter is particularly valuable because its 2026 GNC chapter tabulates flight-heritage reaction wheels, star trackers, gyros, and integrated ADCS units on a common basis; it reports small-spacecraft reaction-wheel capabilities spanning (2.3\times10^{-4})–0.3 N·m peak torque and (5\times10^{-4})–8 N·m·s momentum storage, while noting that actual selection depends strongly on spacecraft inertia, maneuver rate, and wheel saturation requirements.

For a 6U precision-pointing mission, ASTERIA is the strongest flight reference. It was a 6U CubeSat whose payload requirement was 5 arcsec RMS pointing stability over a 20-minute observation and 1 arcsec RMS repeatability between observations. On orbit it demonstrated about 0.5 arcsec RMS stability over 20 minutes and roughly 1 milliarcsecond RMS repeatability. The crucial caveat is that this was a two-stage pointing system: reaction wheels controlled the spacecraft bus, while a piezoelectric focal-plane stage supplied fine payload pointing. Therefore, 0.5 arcsec should not be adopted as a reaction-wheel-only bus requirement. ASTERIA is nevertheless highly relevant because its flight paper also reports problems associated with reaction-wheel zero crossings and momentum management caused by a large spacecraft residual magnetic dipole.

For a reaction-wheel-only bus, the achievable numbers are less spectacular but more appropriate for your model. NASA's 2026 survey lists Blue Canyon integrated small-spacecraft ADCS units at about 0.002–0.007° advertised pointing capability, corresponding to roughly 7–25 arcsec, while CubeSpace's four-wheel-capable CubeADCS is listed near 70 arcsec at 3σ. An independent high-fidelity 12 kg, 6U simulation by Franquiz obtained 0.7 arcmin, or 42 arcsec at 3σ, with reaction wheels during target tracking in a model containing aerodynamic, residual-magnetic, gravity-gradient, and solar-radiation-pressure disturbances. These numbers suggest that tens of arcseconds is the right order of magnitude for a good 6U bus-level course-project requirement, whereas few-arcsecond or subarcsecond performance should be identified as payload/fine-steering territory.

The Franquiz/ARAPAIMA study is unusually useful for your numerical project because it models essentially the same mass and orbital regime: a 12 kg 6U spacecraft near 500 km. It explicitly models all four environmental torques you asked for, a star tracker, gyro errors, reaction wheels, quaternion rigid-body dynamics, estimation, and several controllers. At 500 km its modeled peak external torques are approximately (3.2\times10^{-6}) N·m residual magnetic, (1.2\times10^{-6}) N·m aerodynamic, (1.5)–(1.6\times10^{-7}) N·m gravity-gradient, and (0.9\times10^{-7}) N·m solar radiation pressure. Thus residual magnetic torque is dominant in that particular 6U case, with aerodynamic drag second. The conclusion depends strongly on magnetic cleanliness: ARAPAIMA assumed a rather large 0.1 A·m² residual dipole, while aerodynamic torque also varies greatly with altitude, solar activity, projected area, attitude, and center-of-pressure offset.

The control literature has a clear practical baseline. Quaternion/eigenaxis PD or PID feedback is the recurring spacecraft solution because it handles large rotations directly, has very low implementation cost, and is straightforward to combine with reaction-wheel allocation. Franquiz uses quaternion eigenaxis feedback and explicitly treats it as analogous to PD control; the study compares fixed and scheduled eigenaxis laws and PID variants. LQR remains useful as a linearized near-target benchmark and gives systematic gain selection, but it does not inherently enforce wheel torque or momentum limits. Sliding-mode/adaptive approaches are attractive when robustness to inertia errors and disturbances is the research question, although chattering/high-frequency control is undesirable around real reaction wheels. MPC is especially attractive for this project if you want an advanced comparison because reaction-wheel torque, speed, and momentum constraints can be built directly into the optimization; published constrained-MPC spacecraft work explicitly exploits this feature.

My recommendation is therefore to make a quaternion PD controller your primary controller, use LQR as the clean academic comparison, and treat MPC or sliding mode as a stretch goal rather than trying to implement four sophisticated controllers. The intellectually interesting part of this project is not obtaining yet another nominal slew plot; it is showing how performance changes once realistic disturbances, star-tracker/gyro noise, wheel momentum limits, inertia errors, and possibly a failed wheel are introduced. That direction is strongly aligned with the issues emphasized in both flight experience and the small-spacecraft state of the art.
Quantitative design data

The table below separates published/advertised numbers from recommended modeling values. That distinction matters particularly for settling time and inertia uncertainty, where there is no single accepted “6U standard” in the verified sources.
Quantity	Verified value or useful range	Units	Interpretation for your simulation	Source
Precision pointing — payload stability	Requirement: 5; achieved 0.5	arcsec RMS over 20 min	ASTERIA flight benchmark. Not reaction-wheel-only: piezo focal-plane fine steering was part of the loop.	Pong/ASTERIA.
Precision pointing — repeatability	Requirement 1 arcsec RMS; achieved about 1 mas RMS	arcsec or mas RMS	ASTERIA payload observation-to-observation metric; do not confuse with absolute bus pointing accuracy.	Pong/ASTERIA.
Bus pointing capability, current small-spacecraft ADCS	BCT units approximately 0.002–0.007° = 7.2–25.2 arcsec; CubeSpace approximately 70 arcsec, 3σ	deg / arcsec	Useful indication that tens-of-arcseconds bus pointing is realistic for high-performance CubeSat-class hardware. Definitions/configurations differ by vendor.	NASA 2026 SoA.
6U reaction-wheel pointing simulation	0.7 arcmin = 42 arcsec, 3σ	arcsec	Particularly relevant reaction-wheel-only figure from a 12 kg 6U simulation with modeled environmental disturbances.	Franquiz.
Slew-rate constraint while using star tracker	Low-cost units may track around 0.3°/s; high-end units up to roughly 3°/s; modeled BCT Nano Star Tracker nominal <1°/s, off-nominal <4°/s	deg/s	This is more defensible than claiming a universal 6U mission slew rate. A project cap of 0.5–1°/s keeps the maneuver compatible with precision optical attitude sensing.	NASA SoA; Franquiz/BCT model.
Settling time	No universal 6U precision-pointing value found in the verified sources	s	The flight/design literature more commonly specifies pointing error and observation duration. For your project, define it yourselves; 30–60 s after completion of a representative slew is a reasonable explicit course-project requirement, not a claimed mission standard.	The ARAPAIMA study contains large-angle slew simulations but does not state a classical settling-time requirement.
BCT RWp015 peak torque	0.004	N·m	Very plausible conservative 6U-class wheel model.	NASA 2026 SoA / BCT data.
BCT RWp015 momentum capacity	0.015	N·m·s	Good choice if you want saturation/momentum accumulation to become visible in the simulation.	NASA 2026 SoA / BCT.
BCT RWp050 peak torque / capacity	0.007 / 0.050	N·m / N·m·s	Higher-margin alternative appropriate when RWp015 storage proves too restrictive.	NASA 2026 SoA / BCT.
Rocket Lab/Sinclair RW-0.03 momentum	0.03 nominal; 0.04 peak	N·m·s	Another very relevant wheel size.	Official Rocket Lab product sheet.
Rocket Lab/Sinclair RW-0.03 torque	approximately ±2 mN·m at 0.02 N·m·s, falling to ±0.5 mN·m near 0.04 N·m·s	N·m	Important modeling point: available torque can decrease near the wheel's momentum/speed limit.	Official Rocket Lab product sheet.
Representative maximum wheel speed	approximately 8000 rpm, derived from (h_{\max}/J_w)	rpm	Franquiz models an RW-0.03-4 with (J_w=5.109\times10^{-5}) kg·m² and (h_{\max}=0.0428) N·m·s; these imply 837.7 rad/s ≈ 8000 rpm. This is model-derived from the legacy wheel parameters; the current Rocket Lab sheet available here did not explicitly state max rpm.	Franquiz reaction-wheel model.
Aerodynamic disturbance, 500 km 6U	peak approximately (1.2\times10^{-6})	N·m	ARAPAIMA, 12 kg 6U, 500 km, 28.5°. Strongly altitude/attitude/density dependent.	Franquiz Fig. 2.4.
Residual-magnetic disturbance, 500 km 6U	peak approximately (3.2\times10^{-6})	N·m	Based on 0.1 A·m² residual dipole. Largest of the four disturbances in this case.	Franquiz Fig. 2.5 and model.
Gravity-gradient disturbance, 500 km 6U	peak approximately (1.5)–(1.6\times10^{-7})	N·m	About an order of magnitude below drag and ~20× below magnetic torque in the cited case.	Franquiz Fig. 2.6.
Solar-radiation-pressure disturbance, 500 km 6U	peak approximately (0.9\times10^{-7})	N·m	Uses 0.1221 m² illuminated area, 0.1 m moment arm, reflectance factor 0.6, 1366 W/m² solar flux; goes to zero in eclipse.	Franquiz Fig. 2.7/Table 2.2.
Disturbance ranking at representative 500 km	magnetic > aerodynamic ≫ gravity gradient ≳ SRP	—	For the cited 0.1 A·m² residual dipole case. A magnetically cleaner satellite can make aerodynamic drag dominant, especially toward 400 km.	Derived directly from the four cited disturbance plots.
Wheel momentum accumulated in one 500-km orbit	worst modeled axis 19.22% of 0.0428 N·m·s = 0.00823 N·m·s/orbit	N·m·s/orbit	Extremely useful no-dumping benchmark. It corresponds to an orbit-averaged equivalent bias torque of roughly 1.45 μN·m.	Published percentage and capacity; arithmetic here.
Illustrative no-dump saturation time, RWp015	approximately 2.9 h from zero, or 2.3 h to 80% capacity if the ARAPAIMA net accumulation repeated linearly	h	This is deliberately conservative extrapolation, not a flight prediction; orbital disturbance torques are periodic and their vector averages depend on attitude/orbit.	Calculation using 0.015 N·m·s capacity and cited accumulation.
Illustrative no-dump saturation time, 0.03 N·m·s wheel	approximately 5.7 h from zero, 4.6 h to 80%	h	Useful sensitivity case for your simulation.	Calculation from cited data.
CubeSat star tracker noise/accuracy	BCT Standard NST: 6 arcsec cross-boresight, 40 arcsec twist/roll, 1σ	arcsec	Good direct measurement-noise model. ARAPAIMA uses 5 Hz update rate for this sensor model.	BCT/NASA and Franquiz.
General small-spacecraft star-tracker capability	about 8 arcsec state-of-the-art pointing knowledge	arcsec	Useful check on the BCT values.	NASA 2026 SoA.
Small-spacecraft gyro state of art	bias stability about 0.15°/h, ARW about 0.02°/√h	°/h; °/√h	Good high-performance simulation target.	NASA 2026 SoA.
Compact commercial gyro example	STIM300-family bias instability about 0.4°/h, noise/ARW about 0.10°/√h	°/h; °/√h	A realistic small-SWaP IMU example; not exclusively a CubeSat product.	Safran/Sensonor official specification.
Representative 6U mass	ASTERIA about 10 kg; ARAPAIMA simulation 12 kg	kg	Directly brackets your 10–12 kg design point.	ASTERIA and Franquiz.
Idealized 6U principal inertia, 10 kg	(\mathrm{diag}(0.1083,,0.0833,,0.0417)) for dimensions (x,y,z=(0.10,0.20,0.30)) m	kg·m²	Uniform rectangular-prism estimate: (I_{xx}=m(y^2+z^2)/12), etc. Use as a clean nominal plant, not as ASTERIA's measured inertia. NASA confirms the CubeSat unit form factor; the tensor is calculated here.	NASA form-factor definition.
Idealized 6U principal inertia, 12 kg	(\mathrm{diag}(0.1300,,0.1000,,0.0500))	kg·m²	Same uniform-prism assumption. Thus your nominal inertia can reasonably sit near (\mathrm{diag}(0.12,0.092,0.046)) kg·m² for an 11 kg vehicle.	Calculated from your 10–12 kg project range and standard envelope.
Inertia uncertainty	No verified universal CubeSat percentage found	%	Do not claim that ±20% is an industry standard. For your robustness study, I recommend imposing ±20% independently on principal inertias as an explicit project assumption. Franquiz explicitly discusses robustness to inertia-matrix uncertainty but does not establish a universal percentage.	Franquiz.
Four-wheel pyramid angle	isotropic choice 54.7356° from +body (z), equivalently 35.2644° above the (xy) plane	degrees	This is the symmetric angle that makes (AA^T\propto I). Angle conventions vary in papers, so always state yours. NASA confirms four-wheel skew/pyramid arrangements are standard for small-spacecraft fault tolerance.	Geometry derived below; configuration context from NASA.
Wheel geometry and allocation

For your Python model, a particularly convenient symmetric four-wheel pyramid uses wheel-axis azimuths

[ \phi_i={0^\circ,90^\circ,180^\circ,270^\circ} ]

and a common angle (\beta) measured from (+z). Define each column of (A) as a reaction-wheel spin-axis unit vector expressed in body coordinates:

[ A = \begin{bmatrix} \sin\beta & 0 &-\sin\beta&0\ 0&\sin\beta&0&-\sin\beta\ \cos\beta&\cos\beta&\cos\beta&\cos\beta \end{bmatrix}. ]

Choosing

[ \beta=\tan^{-1}\sqrt 2=54.7356^\circ ]

gives

[ \sin\beta=\sqrt{\frac23}=0.8164966,\qquad \cos\beta=\frac1{\sqrt3}=0.5773503, ]

so

[ A= \begin{bmatrix} 0.8165&0&-0.8165&0\ 0&0.8165&0&-0.8165\ 0.5774&0.5774&0.5774&0.5774 \end{bmatrix}, ]

and

[ AA^T=\frac43 I_3. ]

That last relation is why this angle is attractive: the idealized cluster is isotropic in the least-squares sense. Four-wheel skewed configurations are widely used in small spacecraft specifically because they retain three-axis actuation after one wheel failure, albeit with reduced control authority. NASA's current state-of-the-art review explicitly identifies a four-wheel skewed configuration as a common small-spacecraft fault-tolerance approach.

If (u_w=[u_1,u_2,u_3,u_4]^T) denotes torque applied to the wheel rotors, conservation of angular momentum gives the body torque

[ \tau_b=-A,u_w. ]

The nominal minimum-norm allocation is therefore

[ u_w=-A^\dagger \tau_c =-A^T(AA^T)^{-1}\tau_c =-\frac34A^T\tau_c. ]

This is a useful place to apply wheel torque saturation individually:

[ |u_i|\le u_{\max}, ]

and momentum/speed saturation individually:

[ |h_i|\le h_{\max},\qquad \dot h_i=u_i. ]

When one wheel fails, simply delete its column from (A) and solve with the resulting (3\times3) matrix. For the isotropic geometry above, every single-wheel-out matrix remains full rank; its singular values are approximately (1.155,\ 1.155,\ 0.577), giving condition number 2 compared with 1 for the intact isotropic array. Thus a one-wheel failure does not destroy three-axis controllability in the ideal model, but it removes redundancy and reduces the worst-direction torque/momentum margin. This is precisely the failure mode that is worth demonstrating numerically. NASA likewise notes that the price of skew-wheel fault tolerance is reduced torque capability relative to a dedicated orthogonal wheel on each axis.
Control approaches and primary challenges

Quaternion feedback PD/eigenaxis control is the best baseline for this project. A typical law is

[ \tau_c=-K_p,\mathbf q_e-K_d,\boldsymbol\omega_e ]

with optional gyroscopic cancellation

[ +\boldsymbol\omega\times J\boldsymbol\omega. ]

Franquiz uses essentially this architecture for large-angle slew and target-tracking modes and compares it with PID/scheduled variants. Its advantages are a genuinely nonlinear attitude representation, simple gain tuning, computational economy, and easy compatibility with reaction-wheel allocation. Its weaknesses are that integral action or disturbance estimation may be needed for persistent bias torques, and the quaternion sign ambiguity has to be handled correctly.

LQR is an excellent second controller for a controls course because it gives a clean contrast: linearize the quaternion-error/attitude-rate dynamics near the commanded orientation,

[ \dot x=Ax+Bu, ]

then obtain (K) from the Riccati equation and use (u=-Kx). LQR provides systematic state-versus-effort tuning and is very easy to reproduce, but its guarantee is tied to the linearized operating region and ordinary unconstrained LQR does not know that a wheel has finite torque, finite momentum, or a maximum speed. The standard spacecraft attitude-control treatment and quaternion dynamics are covered comprehensively by Markley and Crassidis.

Sliding-mode control is most attractive when your research question is robustness rather than nominal performance. Spacecraft/CubeSat research uses sliding-mode and higher-order/adaptive variants to tolerate disturbance and model errors; a representative spacecraft-control comparison treats quaternion PD as the conventional baseline and adaptive continuous-twisting sliding mode as the robust alternative. The trade is that the switching structure can produce chattering or high-frequency command activity unless smoothed, which is awkward when wheel torque quantization, wheel imbalance, and pointing jitter matter.

Model-predictive control has the strongest conceptual fit to reaction-wheel constraints. Published constrained spacecraft MPC work incorporates reaction-wheel limitations and momentum management directly in the optimization rather than applying saturation after an unconstrained controller has already requested an infeasible torque. Its drawbacks for your project are implementation effort, computation, and the need to choose between local linear MPC and nonlinear/quaternion formulations. It is an excellent stretch comparison, but PD versus LQR will probably give more insight per hour of a three-student course project.

The literature and flight experience point to several failure mechanisms that deserve explicit simulations:
Challenge	Why it matters in your model	Recommended test
Wheel saturation / momentum accumulation	Reaction wheels absorb external angular momentum; once maximum speed is reached they can no longer store momentum. NASA identifies desaturation as necessary for long-duration operation, and ASTERIA experienced momentum-management problems associated with a large residual dipole. 	Integrate (h_i(t)) for several orbits with no unloading. Report time to 50%, 80%, and 100% capacity.
Residual magnetic bias	In the representative 6U/500-km model it is the largest external torque, peaking near 3.2 μN·m for a 0.1 A·m² dipole. 	Sweep residual dipole from 0.01 to 0.1 A·m² and show saturation time sensitivity. Treat the sweep endpoints as modeling assumptions except for the cited 0.1 case.
Wheel zero crossings	NASA notes that reaction-wheel performance commonly degrades near zero speed, and ASTERIA specifically encountered pointing issues associated with speed reversals/zero crossings. Some missions therefore bias wheel momentum away from zero at the cost of usable storage. 	Compare zero-centered wheels to a nonzero initial wheel-momentum bias; optionally add a small zero-speed torque/jitter penalty.
Quaternion unwinding	(q) and (-q) describe the same physical attitude, so a controller that treats the signs as different states can command an unnecessary long rotation.	Before feedback, enforce a shortest-rotation convention such as changing (q_e\to-q_e) when its scalar component is negative; include a test initialized close to the antipodal quaternion representation.
One failed wheel	A skewed four-wheel array is specifically used to retain three-axis capability after a wheel failure, but with reduced authority and no remaining redundancy. 	Disable each wheel in turn, recompute allocation, and repeat the slew and hold tests. Monitor the surviving wheels for secondary saturation.
Inertia uncertainty	Feedforward cancellation, LQR gains, and aggressive gain selection depend on (J). The detailed 6U study explicitly identifies robustness to inertia-matrix uncertainty as a control issue. 	Monte Carlo or corner cases at nominal (J) and ±20% principal moments. Clearly call ±20% a project robustness assumption, not an industry standard.
Sensor noise / star-tracker update rate	A 6 arcsec cross-axis star tracker at only a few Hz cannot support arbitrarily small bus-error claims without gyro propagation/filtering. Franquiz combines a 5 Hz star tracker with faster gyro measurements for this reason. 	Run ideal-state feedback first, then noisy gyro + star tracker, preferably with an MEKF/EKF or a simpler complementary estimator if estimation is not the course focus.

For momentum buildup in particular, the Franquiz result gives you an unusually defensible benchmark. Its wheels reach as much as 19.22% of a 0.0428 N·m·s saturation level in one orbit, or roughly 0.00823 N·m·s on the most heavily loaded axis. At a roughly 94.5-minute 500-km orbit, this is equivalent to an orbit-average net momentum growth of about (1.45\times10^{-6}) N·m. If that same net growth were applied without dumping to a 0.015 N·m·s wheel, an illustrative linear extrapolation reaches saturation in only about 2.9 hours; a 0.03 N·m·s wheel would last about 5.7 hours. Because the true torques are periodic and allocation redistributes momentum among four skewed wheels, your simulation should compute this directly rather than assume linear accumulation—but these numbers provide an excellent sanity check.
Recommended sources
Burton et al. — survey paper

R. Burton, S. Weston, E. Agasid, and R. Shimmin, “State of the Art in Guidance, Navigation and Control: A Survey of Small Satellite GNC Components,” AAS 16-062, 2016. The accessible copy is an author-uploaded preprint explicitly identified as a survey of small-satellite GNC components.

URL: https://www.researchgate.net/publication/292608145_State_of_the_Art_in_Guidance_Navigation_and_Control_A_Survey_of_Small_Satellite_GNC_Components

This is the survey paper I would use to satisfy your assignment's survey requirement. Its value is breadth: it establishes what sensor and actuator technologies were considered state of the art for small satellites and gives historical context for why reaction wheels, star trackers, and compact inertial sensors are the normal precision-pointing architecture.

For current numerical hardware values, I would not use its decade-old component tables alone; pair it with NASA's 2026 update below.
Pong — ASTERIA flight performance

Christopher M. Pong, “On-Orbit Performance & Operation of the Attitude & Pointing Control Subsystems on ASTERIA,” Jet Propulsion Laboratory / Small Satellite Conference, 2018; NASA NTRS record 20210008536.

URL: https://ntrs.nasa.gov/citations/20210008536

ASTERIA is almost exactly the mission class you are trying to motivate: a 6U precision photometry spacecraft. The paper gives unusually clear flight-level performance targets and achieved results—5 arcsec RMS required stability over 20 minutes versus 0.5 arcsec RMS demonstrated—and discusses reaction-wheel zero crossings and momentum-management problems associated with residual magnetic dipole.

This is your best source for explaining why precision pointing is a meaningful 6U requirement, but state explicitly that its best pointing came from reaction-wheel bus control plus a piezoelectric focal-plane fine-steering stage. It should anchor your mission motivation, not be misused as proof that four reaction wheels alone yield 0.5 arcsec.
Franquiz — reproducible high-fidelity 6U simulation

Francisco J. Franquiz, Attitude Determination & Control System Design and Implementation for a 6U CubeSat Proximity Operations Mission, master's thesis, Embry-Riddle Aeronautical University, 2015; archived by NASA's Small Spacecraft Systems Virtual Institute.

URL: https://s3vi.ndc.nasa.gov/ssri-kb/static/resources/Attitude%20Determination%20%26%20Control%20System%20Design%20and%20Implementation.pdf

This is the full-simulation source I recommend for reproducibility. Its modeled spacecraft is a 12 kg 6U vehicle at approximately 500 km, and its plant includes rigid-body quaternion dynamics, aerodynamics, residual magnetism, gravity gradient, solar radiation pressure, reaction-wheel dynamics, star tracker and gyro models, state estimation, and controller comparisons. It supplies equations, numerical parameter tables, simulated disturbance histories, and controller gains/rates rather than only showing a final attitude-error plot.

It is especially useful to you because the disturbance plots establish an internally consistent 500-km benchmark: approximately 3.2 μN·m residual magnetic, 1.2 μN·m aerodynamic, (1.5\times10^{-7}) N·m gravity-gradient, and (0.9\times10^{-7}) N·m SRP, all for the same spacecraft and orbit. It also quantifies reaction-wheel momentum accumulation, which many controller papers ignore.

Its main mismatch with your final model is that the ARAPAIMA design evolved toward an RCS-based architecture and some reaction-wheel discussion uses three body-axis-aligned wheels rather than your four-wheel pyramid. Use it for the plant, environment, sensor models, magnitude checks, and simulation methodology; use your own four-wheel allocation layer.
NASA Small Spacecraft Systems Virtual Institute — current hardware survey

NASA Small Spacecraft Systems Virtual Institute, “Guidance, Navigation, and Control,” Chapter 5 of State-of-the-Art of Small Spacecraft Technology, updated May 7, 2026.

URL: https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/

This is the strongest single current source for hardware constraints. It lists flight-heritage small-spacecraft reaction wheels including BCT RWp015 at 0.004 N·m / 0.015 N·m·s and RWp050 at 0.007 N·m / 0.050 N·m·s; current integrated ADCS pointing figures; star trackers including the BCT Standard NST at 6 arcsec cross-axis and 40 arcsec twist; and small-spacecraft gyro state-of-the-art figures.

It is also a valuable architecture source because NASA explicitly discusses wheel saturation, momentum dumping, loss of usable storage when wheels are momentum-biased, poorer performance around wheel zero crossings, and the use of a fourth skewed wheel for fault tolerance. Those statements map directly onto the constraints you need to put into a numerical controller project.
Markley and Crassidis — theoretical reference

F. Landis Markley and John L. Crassidis, Fundamentals of Spacecraft Attitude Determination and Control, Springer, 2014. DOI: 10.1007/978-1-4939-0802-8.

DOI URL: https://doi.org/10.1007/978-1-4939-0802-8

This should be your theoretical reference for quaternion kinematics, rigid-body dynamics, attitude determination, and spacecraft attitude-control conventions. It is preferable to piecing together sign conventions for quaternions, body/inertial transformations, and actuator mappings from unrelated CubeSat papers because those conventions are a major source of simulation errors.

For your report, these five sources have complementary jobs: Burton = survey paper; Pong = 6U flight requirement/performance; Franquiz = reproducible 6U simulation; NASA 2026 = current hardware constraints; Markley & Crassidis = equations and theory. Current supplier documents such as the Rocket Lab RW-0.03 and BCT wheel/star-tracker specifications should then be cited additionally beside the exact actuator and sensor limits they support.
Recommended project requirements

I would avoid trying to emulate ASTERIA's 0.5-arcsec payload result directly. A better course project is demanding enough to make sensor noise and saturation meaningful while remaining plausible for reaction-wheel bus control.
Proposed requirement	Quantified requirement	Constraint or challenge it exercises	Rationale
Precision pointing accuracy	After settling, maintain boresight attitude error ≤30 arcsec RMS, with ≤60 arcsec 3σ as a secondary acceptance metric, during a 20-min hold.	Star-tracker accuracy, disturbance rejection, wheel torque resolution, estimator error.	Lies in the same order as the 7–70 arcsec small-spacecraft hardware range and slightly exceeds the 42 arcsec 3σ reaction-wheel result from the 12 kg ARAPAIMA simulation without pretending to achieve ASTERIA's piezo-assisted 0.5 arcsec.
Pointing stability / jitter	During the settled 20-min hold, require detrended boresight variation ≤10 arcsec RMS. Report a 5 arcsec RMS stretch goal separately.	Gyro/star-tracker noise, wheel zero crossings, controller bandwidth, wheel jitter.	ASTERIA's payload requirement was 5 arcsec RMS over 20 minutes, but it required a fine-steering piezo stage; 10 arcsec makes the wheel-only simulation ambitious without presenting the fine-stage result as bus capability.
Rest-to-rest slew	Execute a 30° eigenaxis slew in ≤90 s, beginning and ending below 0.01°/s, with commanded body rate ≤0.5°/s.	Torque limit, sensor tracking rate, large-angle quaternion control.	A 0.5°/s rate remains below the modeled BCT NST's <1°/s nominal tracking range; 30°/90 s is demanding enough to expose actuator constraints while staying in a precision-spacecraft regime. The time itself is a proposed project value rather than a sourced mission requirement.
Post-slew settling	Following the planned slew profile, enter and remain inside ±60 arcsec pointing error and ±0.01°/s rate within 30 s.	Damping, gain tuning, overshoot, sensor noise.	The verified 6U sources do not supply a universal classical settling-time specification, so making the error band and rate band explicit is more defensible than claiming an industry-standard number.
Reaction-wheel feasibility and momentum accounting	Nominal model: each wheel obeys **(	u_i	\le4) mN·m** and **(
Robustness / fault tolerance	Repeat the principal slew and hold tests with ±20% principal-inertia perturbations, full sensor noise and all four environmental torques; as a stretch case, disable any one of four wheels and require stable three-axis control without exceeding surviving-wheel limits.	Model uncertainty and redundant wheel allocation.	±20% should be explicitly labeled a project robustness envelope rather than a published standard. A four-wheel skew cluster is specifically motivated by one-wheel-out fault tolerance, but surviving wheels have reduced margin.

A useful way to organize the experiments is consequently:

[ \text{ideal plant} \rightarrow \text{disturbed plant} \rightarrow \text{noisy sensed plant} \rightarrow \text{wheel saturation} \rightarrow \text{inertia uncertainty} \rightarrow \text{one-wheel failure}. ]

For each step, report pointing RMS/3σ error, maximum rate, settling time, maximum wheel torque, maximum wheel momentum, and accumulated wheel momentum per orbit. That gives you a far stronger controls paper than simply comparing four controllers under ideal conditions.

For the nominal simulation I would use

[ m=11\ {\rm kg}, \qquad J_0= \begin{bmatrix} 0.119 & 0 &0\ 0&0.0917&0\ 0&0&0.0458 \end{bmatrix} {\rm kg,m^2}, ]

which is the uniform-prism estimate at the midpoint of your 10–12 kg range. It sits between the calculated 10 kg and 12 kg bounds above; the true inertia should then be varied in the robustness study rather than pretending that this ideal rectangular-prism estimate is known exactly. The 10–12 kg range itself is well supported by the 10 kg ASTERIA and 12 kg ARAPAIMA 6U examples.

For disturbances at the nominal 500 km case, an especially clean baseline is

[ |\tau_{\rm aero}|_{\rm pk}\approx1.2\times10^{-6}\ {\rm N,m}, ]

[ |\tau_{\rm mag}|_{\rm pk}\approx3.2\times10^{-6}\ {\rm N,m}, ]

[ |\tau_{\rm gg}|_{\rm pk}\approx1.6\times10^{-7}\ {\rm N,m}, ]

[ |\tau_{\rm srp}|_{\rm pk}\approx9\times10^{-8}\ {\rm N,m}. ]

Rather than implement these as four constant torques, reproduce their physical dependence: (\tau_{\rm gg}=3\mu r^{-3}\hat r\times J\hat r); (\tau_{\rm mag}=m_{\rm res}\times B); drag from dynamic pressure, projected area, and center-of-pressure offset; and SRP from incident flux, projected area, optical coefficient, and moment arm. The Franquiz source supplies precisely these models and shows their resulting orbital histories.

One particularly worthwhile sensitivity study is residual magnetic dipole. At the cited 0.1 A·m² value, magnetic torque is clearly dominant; reducing the assumed dipole by a factor of ten would reduce that torque by the same factor and move it down to the few-(10^{-7}) N·m scale, at which point aerodynamic drag can dominate. That simple sweep both explains why different CubeSat papers report different “dominant disturbance” rankings and connects directly to ASTERIA's flight experience with residual-dipole-driven momentum management. The linear magnetic scaling follows directly from (\tau_{\rm mag}=m_{\rm res}\times B); the 0.1 A·m² reference case and ASTERIA issue are documented in the cited sources.
Homework-ready system description

The system considered is a rigid 6U CubeSat of approximately 10–12 kg in a 400–600 km low-Earth orbit, with nominal simulations performed at 500 km. The spacecraft attitude is represented by a unit quaternion and its rotational dynamics by Euler's rigid-body equation using a nominal principal-axis inertia near (J=\mathrm{diag}(0.119,,0.0917,,0.0458)\ {\rm kg,m^2}), obtained from an 11 kg uniform 0.1 m × 0.2 m × 0.3 m 6U approximation and subsequently varied to study inertia uncertainty. Attitude control is provided by four reaction wheels in a symmetric pyramid, using an isotropic skew of 54.74° from the body (+z) axis and minimum-norm torque allocation; a representative wheel limit is 4 mN·m torque and 0.015 N·m·s stored momentum per wheel. The primary maneuvers are rest-to-rest slews followed by precision inertial pointing holds. The plant includes gravity-gradient, aerodynamic, residual-magnetic, and solar-radiation-pressure disturbance torques, with representative 500-km peak magnitudes of approximately (1.6\times10^{-7}), (1.2\times10^{-6}), (3.2\times10^{-6}), and (0.9\times10^{-7}) N·m, respectively, for the cited 12 kg 6U reference case; residual magnetic torque is therefore dominant for its assumed 0.1 A·m² residual dipole. Attitude measurements are modeled using a CubeSat-class star tracker with approximately 6 arcsec cross-boresight and 40 arcsec roll/twist 1σ errors together with a higher-rate gyro. Momentum dumping is excluded from the controller, but wheel angular momentum is integrated throughout every simulation so that the time to wheel saturation can be quantified. The baseline controller is quaternion-feedback PD/eigenaxis control, with LQR as the principal comparison and optional constrained MPC as an advanced extension; performance is evaluated in terms of pointing accuracy and stability, slew and settling time, wheel torque and momentum limits, inertia uncertainty, sensor noise, and one-wheel-out operation.
