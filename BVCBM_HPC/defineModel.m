%% About defineModel.mlx
% This file defines the MATLAB interface to the library |Model|.
%
% Commented sections represent C++ functionality that MATLAB cannot automatically define. To include
% functionality, uncomment a section and provide values for &lt;SHAPE&gt;, &lt;DIRECTION&gt;, etc. For more
% information, see <matlab:helpview(fullfile(docroot,'matlab','helptargets.map'),'cpp_define_interface') Define MATLAB Interface for C++ Library>.



%% Setup. Do not edit this section.
function libDef = defineModel()
libDef = clibgen.LibraryDefinition("ModelData.xml");
%% OutputFolder and Libraries 
libDef.OutputFolder = "/mnt/home/n10977341/VCBM";
libDef.Libraries = "";

%% C++ class |Params| with MATLAB name |clib.Model.Params| 
ParamsDefinition = addClass(libDef, "Params", "MATLABName", "clib.Model.Params", ...
    "Description", "clib.Model.Params    Representation of C++ class Params."); % Modify help description values as needed.

%% C++ class constructor for C++ class |Params| 
% C++ Signature: Params::Params(double p_0,double p_psc,double dmax,int gage,int page)
ParamsConstructor1Definition = addConstructor(ParamsDefinition, ...
    "Params::Params(double p_0,double p_psc,double dmax,int gage,int page)", ...
    "Description", "clib.Model.Params    Constructor of C++ class Params."); % Modify help description values as needed.
defineArgument(ParamsConstructor1Definition, "p_0", "double");
defineArgument(ParamsConstructor1Definition, "p_psc", "double");
defineArgument(ParamsConstructor1Definition, "dmax", "double");
defineArgument(ParamsConstructor1Definition, "gage", "int32");
defineArgument(ParamsConstructor1Definition, "page", "int32");
validate(ParamsConstructor1Definition);

%% C++ class method |RandomDouble| for C++ class |Params| 
% C++ Signature: double Params::RandomDouble()
RandomDoubleDefinition = addMethod(ParamsDefinition, ...
    "double Params::RandomDouble()", ...
    "MATLABName", "RandomDouble", ...
    "Description", "RandomDouble    Method of C++ class Params."); % Modify help description values as needed.
defineOutput(RandomDoubleDefinition, "RetVal", "double");
validate(RandomDoubleDefinition);

%% C++ class method |WithProbability| for C++ class |Params| 
% C++ Signature: bool Params::WithProbability(double prob)
WithProbabilityDefinition = addMethod(ParamsDefinition, ...
    "bool Params::WithProbability(double prob)", ...
    "MATLABName", "WithProbability", ...
    "Description", "WithProbability    Method of C++ class Params."); % Modify help description values as needed.
defineArgument(WithProbabilityDefinition, "prob", "double");
defineOutput(WithProbabilityDefinition, "RetVal", "logical");
validate(WithProbabilityDefinition);

%% C++ class public data member |gage| for C++ class |Params| 
% C++ Signature: int Params::gage
addProperty(ParamsDefinition, "gage", "int32", ...
    "Description", "int32    Data member of C++ class Params."); % Modify help description values as needed.

%% C++ class public data member |page| for C++ class |Params| 
% C++ Signature: int Params::page
addProperty(ParamsDefinition, "page", "int32", ...
    "Description", "int32    Data member of C++ class Params."); % Modify help description values as needed.

%% C++ class public data member |p_0| for C++ class |Params| 
% C++ Signature: double Params::p_0
addProperty(ParamsDefinition, "p_0", "double", ...
    "Description", "double    Data member of C++ class Params."); % Modify help description values as needed.

%% C++ class public data member |dmax| for C++ class |Params| 
% C++ Signature: double Params::dmax
addProperty(ParamsDefinition, "dmax", "double", ...
    "Description", "double    Data member of C++ class Params."); % Modify help description values as needed.

%% C++ class public data member |p_psc| for C++ class |Params| 
% C++ Signature: double Params::p_psc
addProperty(ParamsDefinition, "p_psc", "double", ...
    "Description", "double    Data member of C++ class Params."); % Modify help description values as needed.

%% C++ class |Pancreas| with MATLAB name |clib.Model.Pancreas| 
PancreasDefinition = addClass(libDef, "Pancreas", "MATLABName", "clib.Model.Pancreas", ...
    "Description", "clib.Model.Pancreas    Representation of C++ class Pancreas."); % Modify help description values as needed.

%% C++ class method |CreateNewParticle| for C++ class |Pancreas| 
% C++ Signature: Pancreas * Pancreas::CreateNewParticle(Params * parameters)
CreateNewParticleDefinition = addMethod(PancreasDefinition, ...
   "Pancreas * Pancreas::CreateNewParticle(Params * parameters)", ...
   "MATLABName", "CreateNewParticle", ...
   "Description", "CreateNewParticle    Method of C++ class Pancreas."); % Modify help description values as needed.
defineArgument(CreateNewParticleDefinition, "parameters", "clib.Model.Params", "input", 1); % '<MLTYPE>' can be clib.Model.Params, or clib.array.Model.Params
defineOutput(CreateNewParticleDefinition, "RetVal", "clib.Model.Pancreas", 1);
validate(CreateNewParticleDefinition);

%% C++ class method |LoadCellsCoordinates| for C++ class |Pancreas| 
% C++ Signature: double * Pancreas::LoadCellsCoordinates()
LoadCellsCoordinatesDefinition = addMethod(PancreasDefinition, ...
   "double * Pancreas::LoadCellsCoordinates()", ...
   "MATLABName", "LoadCellsCoordinates", ...
   "Description", "LoadCellsCoordinates    Method of C++ class Pancreas."); % Modify help description values as needed.
defineOutput(LoadCellsCoordinatesDefinition, "RetVal", "double", 1);
validate(LoadCellsCoordinatesDefinition);

%% C++ class method |TumourVolume| for C++ class |Pancreas| 
% C++ Signature: double Pancreas::TumourVolume()
TumourVolumeDefinition = addMethod(PancreasDefinition, ...
    "double Pancreas::TumourVolume()", ...
    "MATLABName", "TumourVolume", ...
    "Description", "TumourVolume    Method of C++ class Pancreas."); % Modify help description values as needed.
defineOutput(TumourVolumeDefinition, "RetVal", "double");
validate(TumourVolumeDefinition);

%% C++ class method |CreateInitialTumour| for C++ class |Pancreas| 
% C++ Signature: void Pancreas::CreateInitialTumour()
CreateInitialTumourDefinition = addMethod(PancreasDefinition, ...
    "void Pancreas::CreateInitialTumour()", ...
    "MATLABName", "CreateInitialTumour", ...
    "Description", "CreateInitialTumour    Method of C++ class Pancreas."); % Modify help description values as needed.
validate(CreateInitialTumourDefinition);

%% C++ class method |SimulateOneDay| for C++ class |Pancreas| 
% C++ Signature: double Pancreas::SimulateOneDay(int day)
SimulateOneDayDefinition = addMethod(PancreasDefinition, ...
    "double Pancreas::SimulateOneDay(int day)", ...
    "MATLABName", "SimulateOneDay", ...
    "Description", "SimulateOneDay    Method of C++ class Pancreas."); % Modify help description values as needed.
defineArgument(SimulateOneDayDefinition, "day", "int32");
defineOutput(SimulateOneDayDefinition, "RetVal", "double");
validate(SimulateOneDayDefinition);

%% C++ class method |UpdateParameters| for C++ class |Pancreas| 
% C++ Signature: void Pancreas::UpdateParameters(Params * parameters)
UpdateParametersDefinition = addMethod(PancreasDefinition, ...
   "void Pancreas::UpdateParameters(Params * parameters)", ...
   "MATLABName", "UpdateParameters", ...
   "Description", "UpdateParameters    Method of C++ class Pancreas."); % Modify help description values as needed.
defineArgument(UpdateParametersDefinition, "parameters", "clib.Model.Params", "input", 1); % '<MLTYPE>' can be clib.Model.Params, or clib.array.Model.Params
validate(UpdateParametersDefinition);

%% C++ class method |ReturnCellPositions| for C++ class |Pancreas| 
% C++ Signature: double Pancreas::ReturnCellPositions(int index)
ReturnCellPositionsDefinition = addMethod(PancreasDefinition, ...
    "double Pancreas::ReturnCellPositions(int index)", ...
    "MATLABName", "ReturnCellPositions", ...
    "Description", "ReturnCellPositions    Method of C++ class Pancreas."); % Modify help description values as needed.
defineArgument(ReturnCellPositionsDefinition, "index", "int32");
defineOutput(ReturnCellPositionsDefinition, "RetVal", "double");
validate(ReturnCellPositionsDefinition);

%% C++ class method |ReturnNumberCells| for C++ class |Pancreas| 
% C++ Signature: int Pancreas::ReturnNumberCells()
ReturnNumberCellsDefinition = addMethod(PancreasDefinition, ...
    "int Pancreas::ReturnNumberCells()", ...
    "MATLABName", "ReturnNumberCells", ...
    "Description", "ReturnNumberCells    Method of C++ class Pancreas."); % Modify help description values as needed.
defineOutput(ReturnNumberCellsDefinition, "RetVal", "int32");
validate(ReturnNumberCellsDefinition);

%% C++ class method |ReturnCellType| for C++ class |Pancreas| 
% C++ Signature: int Pancreas::ReturnCellType(int index)
ReturnCellTypeDefinition = addMethod(PancreasDefinition, ...
    "int Pancreas::ReturnCellType(int index)", ...
    "MATLABName", "ReturnCellType", ...
    "Description", "ReturnCellType    Method of C++ class Pancreas."); % Modify help description values as needed.
defineArgument(ReturnCellTypeDefinition, "index", "int32");
defineOutput(ReturnCellTypeDefinition, "RetVal", "int32");
validate(ReturnCellTypeDefinition);

%% C++ class constructor for C++ class |Pancreas| 
% C++ Signature: Pancreas::Pancreas(Pancreas const & input1)
PancreasConstructor1Definition = addConstructor(PancreasDefinition, ...
    "Pancreas::Pancreas(Pancreas const & input1)", ...
    "Description", "clib.Model.Pancreas    Constructor of C++ class Pancreas."); % Modify help description values as needed.
defineArgument(PancreasConstructor1Definition, "input1", "clib.Model.Pancreas", "input");
validate(PancreasConstructor1Definition);

%% C++ function |SeedAndGrowToStartVolume| with MATLAB name |clib.Model.SeedAndGrowToStartVolume|
% C++ Signature: Pancreas * SeedAndGrowToStartVolume(double p0,double psc,int dmax,int gage,int page,double startVolume)
SeedAndGrowToStartVolumeDefinition = addFunction(libDef, ...
    "Pancreas * SeedAndGrowToStartVolume(double p0,double psc,int dmax,int gage,int page,double startVolume)", ...
    "MATLABName", "clib.Model.SeedAndGrowToStartVolume", ...
    "Description", "clib.Model.SeedAndGrowToStartVolume    Representation of C++ function SeedAndGrowToStartVolume."); % Modify help description values as needed.
defineArgument(SeedAndGrowToStartVolumeDefinition, "p0", "double");
defineArgument(SeedAndGrowToStartVolumeDefinition, "psc", "double");
defineArgument(SeedAndGrowToStartVolumeDefinition, "dmax", "int32");
defineArgument(SeedAndGrowToStartVolumeDefinition, "gage", "int32");
defineArgument(SeedAndGrowToStartVolumeDefinition, "page", "int32");
defineArgument(SeedAndGrowToStartVolumeDefinition, "startVolume", "double");
defineOutput(SeedAndGrowToStartVolumeDefinition, "RetVal", "clib.Model.Pancreas", 1);
validate(SeedAndGrowToStartVolumeDefinition);

%% C++ function |CreateNewParticle| with MATLAB name |clib.Model.CreateNewParticle|
% C++ Signature: Pancreas * CreateNewParticle(double p0,double psc,int dmax,int gage,int page,Pancreas * pancreas)
CreateNewParticleDefinition = addFunction(libDef, ...
    "Pancreas * CreateNewParticle(double p0,double psc,int dmax,int gage,int page,Pancreas * pancreas)", ...
    "MATLABName", "clib.Model.CreateNewParticle", ...
    "Description", "clib.Model.CreateNewParticle    Representation of C++ function CreateNewParticle."); % Modify help description values as needed.
defineArgument(CreateNewParticleDefinition, "p0", "double");
defineArgument(CreateNewParticleDefinition, "psc", "double");
defineArgument(CreateNewParticleDefinition, "dmax", "int32");
defineArgument(CreateNewParticleDefinition, "gage", "int32");
defineArgument(CreateNewParticleDefinition, "page", "int32");
defineArgument(CreateNewParticleDefinition, "pancreas", "clib.Model.Pancreas", "input", 1); % '<MLTYPE>' can be clib.Model.Pancreas, or clib.array.Model.Pancreas
defineOutput(CreateNewParticleDefinition, "RetVal", "clib.Model.Pancreas", 1);
validate(CreateNewParticleDefinition);

%% C++ function |UpdateParticle| with MATLAB name |clib.Model.UpdateParticle|
% C++ Signature: void UpdateParticle(double p0,double psc,int dmax,int gage,int page,Pancreas * pancreas)
UpdateParticleDefinition = addFunction(libDef, ...
    "void UpdateParticle(double p0,double psc,int dmax,int gage,int page,Pancreas * pancreas)", ...
    "MATLABName", "clib.Model.UpdateParticle", ...
    "Description", "clib.Model.UpdateParticle    Representation of C++ function UpdateParticle."); % Modify help description values as needed.
defineArgument(UpdateParticleDefinition, "p0", "double");
defineArgument(UpdateParticleDefinition, "psc", "double");
defineArgument(UpdateParticleDefinition, "dmax", "int32");
defineArgument(UpdateParticleDefinition, "gage", "int32");
defineArgument(UpdateParticleDefinition, "page", "int32");
defineArgument(UpdateParticleDefinition, "pancreas", "clib.Model.Pancreas", "input", 1); % '<MLTYPE>' can be clib.Model.Pancreas, or clib.array.Model.Pancreas
validate(UpdateParticleDefinition);

%% C++ function |FullSimulation_biphasic| with MATLAB name |clib.Model.FullSimulation_biphasic|
% C++ Signature: std::vector<double, std::allocator<double>> FullSimulation_biphasic(double p0_1,double psc_1,int dmax_1,int gage_1,int page,double p0_2,double psc_2,int dmax_2,int gage_2,double tau,double startVolume,int simtime)
FullSimulation_biphasicDefinition = addFunction(libDef, ...
    "std::vector<double, std::allocator<double>> FullSimulation_biphasic(double p0_1,double psc_1,int dmax_1,int gage_1,int page,double p0_2,double psc_2,int dmax_2,int gage_2,double tau,double startVolume,int simtime)", ...
    "MATLABName", "clib.Model.FullSimulation_biphasic", ...
    "Description", "clib.Model.FullSimulation_biphasic    Representation of C++ function FullSimulation_biphasic."); % Modify help description values as needed.
defineArgument(FullSimulation_biphasicDefinition, "p0_1", "double");
defineArgument(FullSimulation_biphasicDefinition, "psc_1", "double");
defineArgument(FullSimulation_biphasicDefinition, "dmax_1", "int32");
defineArgument(FullSimulation_biphasicDefinition, "gage_1", "int32");
defineArgument(FullSimulation_biphasicDefinition, "page", "int32");
defineArgument(FullSimulation_biphasicDefinition, "p0_2", "double");
defineArgument(FullSimulation_biphasicDefinition, "psc_2", "double");
defineArgument(FullSimulation_biphasicDefinition, "dmax_2", "int32");
defineArgument(FullSimulation_biphasicDefinition, "gage_2", "int32");
defineArgument(FullSimulation_biphasicDefinition, "tau", "double");
defineArgument(FullSimulation_biphasicDefinition, "startVolume", "double");
defineArgument(FullSimulation_biphasicDefinition, "simtime", "int32");
defineOutput(FullSimulation_biphasicDefinition, "RetVal", "clib.array.Model.Double");
validate(FullSimulation_biphasicDefinition);

%% C++ function |FullSimulation| with MATLAB name |clib.Model.FullSimulation|
% C++ Signature: std::vector<double, std::allocator<double>> FullSimulation(double p0,double psc,int dmax,int gage,int page,double startVolume,int simtime)
FullSimulationDefinition = addFunction(libDef, ...
    "std::vector<double, std::allocator<double>> FullSimulation(double p0,double psc,int dmax,int gage,int page,double startVolume,int simtime)", ...
    "MATLABName", "clib.Model.FullSimulation", ...
    "Description", "clib.Model.FullSimulation    Representation of C++ function FullSimulation."); % Modify help description values as needed.
defineArgument(FullSimulationDefinition, "p0", "double");
defineArgument(FullSimulationDefinition, "psc", "double");
defineArgument(FullSimulationDefinition, "dmax", "int32");
defineArgument(FullSimulationDefinition, "gage", "int32");
defineArgument(FullSimulationDefinition, "page", "int32");
defineArgument(FullSimulationDefinition, "startVolume", "double");
defineArgument(FullSimulationDefinition, "simtime", "int32");
defineOutput(FullSimulationDefinition, "RetVal", "clib.array.Model.Double");
validate(FullSimulationDefinition);

%% Validate the library definition
validate(libDef);

end
