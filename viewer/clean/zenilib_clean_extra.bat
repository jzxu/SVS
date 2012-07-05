call "%~dp0zenilib_clean.bat"


DEL /Q /A:H "%~dp0..\Visual Studio 2010\*.suo"

FOR /F "tokens=*" %%G IN ('DIR /B "%~dp0..\Visual Studio 2010\*.user"') DO (
  IF "%%G" NEQ "Application.vcxproj.user" (
    IF "%%G" NEQ "Application_SoarSML.vcxproj.user" (
      IF "%%G" NEQ "Launcher.vcxproj.user" (
        ECHO DEL /Q "%~dp0..\Visual Studio 2010\%%G"
        DEL /Q "%~dp0..\Visual Studio 2010\%%G"
      )
    )
  )
)


DEL /Q "%~dp0..\bin\*_d.dll"
DEL /Q "%~dp0..\bin\*_d.so"
DEL /Q "%~dp0..\bin\x64\*_d.dll"
DEL /Q "%~dp0..\bin\x64\*_d.so"
DEL /Q "%~dp0..\lib_win\*_d.exp"
DEL /Q "%~dp0..\lib_win\*_d.lib"
DEL /Q "%~dp0..\lib_win\x64\*_d.exp"
DEL /Q "%~dp0..\lib_win\x64\*_d.lib"
DEL /Q "%~dp0..\bin\*_d.exe"
DEL /Q "%~dp0..\bin\x64\*_d.exe"
DEL /Q "%~dp0..\*_d.exe"
DEL /Q "%~dp0..\*_x64d.exe"
DEL /Q "%~dp0..\*_d"
DEL /Q "%~dp0..\*_x64d"

FOR /F "tokens=*" %%G IN ('DIR /B /AD /S "%~dp0..\*_d.app"') DO RMDIR /S /Q "%%G"


RMDIR /S /Q "%~dp0..\docs\Zeni"
