import os
import tempfile
import time

from ansys.aedt.core import Hfss

AEDT_VERSION = "2025R1"
NUM_CORES = 4
NG_MODE = False  # Open AEDT UI when it is launched.

temp_folder = tempfile.TemporaryDirectory(suffix=".ansys")

project_name = os.path.join(temp_folder.name, "dipole.aedt")
hfss = Hfss(version=AEDT_VERSION,
            non_graphical=NG_MODE,
            project=project_name,
            new_desktop=True,
            solution_type="Modal",
            )

hfss["l_dipole"] = "10.2cm"
component_name = "Dipole_Antenna_DM"
freq_range = ["1GHz", "2GHz"]      # Frequency range for analysis and post-processing.
center_freq = "1.5GHz"             # Center frequency
freq_step = "0.5GHz"

component_fn = hfss.components3d[component_name]          # Full file name.
comp_params = hfss.get_component_variables(component_name)  # Retrieve dipole parameters.
comp_params["dipole_length"] = "l_dipole"                 # Update the dipole length.
hfss.modeler.insert_3d_component(component_fn, geometry_parameters=comp_params)

hfss.create_open_region(frequency=center_freq)

setup = hfss.create_setup(name="MySetup", MultipleAdaptiveFreqsSetup=freq_range, MaximumPasses=2)

disc_sweep = setup.add_sweep(name="DiscreteSweep", sweep_type="Discrete",
                             RangeStart=freq_range[0], RangeEnd=freq_range[1], RangeStep=freq_step,
                             SaveFields=True)

interp_sweep = setup.add_sweep(name="InterpolatingSweep", sweep_type="Interpolating",
                               RangeStart=freq_range[0], RangeEnd=freq_range[1],
                               SaveFields=False)

setup.analyze()

spar_plot = hfss.create_scattering(plot="Return Loss", sweep=interp_sweep.name)

variations = hfss.available_variations.nominal_values
variations["Freq"] = [center_freq]
variations["Theta"] = ["All"]
variations["Phi"] = ["All"]
elevation_ffd_plot = hfss.post.create_report(expressions="db(GainTheta)",
                                             setup_sweep_name=disc_sweep.name,
                                             variations=variations,
                                             primary_sweep_variable="Theta",
                                             context="Elevation",           # Far-field setup is pre-defined.
                                             report_category="Far Fields",
                                             plot_type="Radiation Pattern",
                                             plot_name="Elevation Gain (dB)"
                                            )
elevation_ffd_plot.children["Legend"].properties["Show Trace Name"] = False
elevation_ffd_plot.children["Legend"].properties["Show Solution Name"] = False

report_3d = hfss.post.reports_by_category.far_field("db(RealizedGainTheta)",
                                                      disc_sweep.name,
                                                      sphere_name="3D",
                                                      Freq= [center_freq],)

report_3d.report_type = "3D Polar Plot"
report_3d.create(name="Realized Gain (dB)")

report_3d_data = report_3d.get_solution_data()
new_plot = report_3d_data.plot_3d()

xpol_expressions = ["db(RealizedGainTheta)", "db(RealizedGainPhi)"]
xpol = hfss.post.reports_by_category.far_field(["db(RealizedGainTheta)", "db(RealizedGainPhi)"],
                                                disc_sweep.name,
                                                name="Cross Polarization",
                                                sphere_name="Azimuth",
                                                Freq= [center_freq],)

xpol.report_type = "Radiation Pattern"
xpol.create(name="xpol")
xpol.children["Legend"].properties["Show Solution Name"] = False
xpol.children["Legend"].properties["Show Variation Key"] = False

ff_el_data = elevation_ffd_plot.get_solution_data()
ff_el_data.plot(x_label="Theta", y_label="Gain", is_polar=True)

hfss.save_project()
hfss.release_desktop()

# Wait 3 seconds to allow AEDT to shut down before cleaning the temporary directory.
time.sleep(3)
temp_folder.cleanup()