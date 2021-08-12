call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars32.bat" %1
if not exist buildOneVPL mkdir buildOneVPL
if not exist buildOneVPL\%1 mkdir buildOneVPL\%1
cd buildOneVPL\%1
if not exist oneVPL.sln (
  cmake -G "Visual Studio 16 2019" -A %1 ^
    -DBUILD_DEV=OFF ^
    -DBUILD_TOOLS=OFF ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DBUILD_EXAMPLES=OFF ^
    -DBUILD_PYTHON_BINDING=OFF ^
    -DCMAKE_C_FLAGS_RELEASE="/MT /MP /O1 /Os /Ob1 /Oy /GT /DNDEBUG" ^
    -DCMAKE_CXX_FLAGS_RELEASE="/MT /MP /O1 /Os /Ob1 /Oy /GT /DNDEBUG" ^
    ..\..\oneVPL
)
cd ..\..\
MSBuild "buildOneVPL\%1\dispatcher\VPL.vcxproj" /property:WindowsTargetPlatformVersion=10.0;PlatformToolset=v142;Configuration="Release";Platform=%1;WholeProgramOptimization=true;ConfigurationType=StaticLibrary;ForceImportBeforeCppTargets="BuildVPL.props" /p:BuildProjectReferences=true /p:SpectreMitigation=false