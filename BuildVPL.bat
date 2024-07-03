set BUILD_ARCH=x64
if "%~1" == "Win32" set BUILD_ARCH=x86
set VSVER=Visual Studio 17 2022
if "%~3" == "v142" set VSVER=Visual Studio 16 2019
if "%VSVER%" == "Visual Studio 16 2019" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars32.bat" %BUILD_ARCH%
) else (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars32.bat" %BUILD_ARCH%
)
cd %~dp0
if not exist buildVPL mkdir buildVPL
if not exist buildVPL\%1 mkdir buildVPL\%1
if not exist buildVPL\%1\%2 mkdir buildVPL\%1\%2
cd buildVPL\%1\%2
if not exist libvpl.sln (
  cmake -G "%VSVER%" -A %1 ^
    -DBUILD_DEV=OFF ^
    -DBUILD_TOOLS=OFF ^
    -DBUILD_SHARED_LIBS=OFF ^
    -DBUILD_EXAMPLES=OFF ^
    -DBUILD_PYTHON_BINDING=OFF ^
    -DCMAKE_C_FLAGS_RELEASE="/MT /MP /O1 /Os /Ob1 /Oy /GT /DNDEBUG /Dlibvpl_EXPERIMENTAL" ^
    -DCMAKE_CXX_FLAGS_RELEASE="/MT /MP /O1 /Os /Ob1 /Oy /GT /DNDEBUG /Dlibvpl_EXPERIMENTAL" ^
    -DCMAKE_C_FLAGS_DEBUG="/MTd /MP /O0 /Zi /Ob0 /Od /RTC1" ^
    -DCMAKE_CXX_FLAGS_DEBUG="/MTd /MP /O0 /Zi /Ob0 /Od /RTC1" ^
    ..\..\..\libvpl
)
cd ..\..\..
MSBuild "buildVPL\%1\%2\libvpl\VPL.vcxproj" /property:WindowsTargetPlatformVersion=10.0;PlatformToolset=%~3;Configuration="%2";Platform=%1;WholeProgramOptimization=true;ConfigurationType=StaticLibrary;ForceImportBeforeCppTargets="BuildVPL.props" /p:BuildProjectReferences=true /p:SpectreMitigation=false