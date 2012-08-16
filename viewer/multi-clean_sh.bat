#!/bin/bash

@echo off
GOTO WINDOWS 



function usage {
  echo
  echo "Usage: $0 [options]"
  echo
  echo "  --build=all       game and all dependencies"
  echo "          mine      game only (default)"
  echo
}

function usage_error {
  echo
  echo "Error: $1"
  usage
  exit $2
}

BUILD=mine
CONFIG=release
DIR=build
MACOSX=native

STATE=config
for arg in "$@"; do
  case "$STATE" in
    build)
      case "$arg" in
        all) BUILD=all ;;
        mine) BUILD=mine ;;
        *) usage_error "Invalid Argument '$arg'" 2
      esac
      STATE=config
      ;;
    config)
      case "$arg" in
        --build) STATE=build ;;
          --build=all) BUILD=all ;;
          --build=mine) BUILD=mine ;;
        --dir) STATE=dir ;;
          --dir=*) DIR=$(echo "$arg" | sed 's/--dir=//') ;;
        --macosx) STATE=macosx ;;
          --macosx=10.6) MACOSX=10.6 ;;
          --macosx=10.7) MACOSX=10.7 ;;
          --macosx=10.8) MACOSX=10.8 ;;
          --macosx=native) MACOSX=native ;;
        debug) CONFIG=debug ;;
        release) CONFIG=release ;;
        releaseuniv) CONFIG=release ;;
        *) usage_error "Invalid Argument '$arg'" 3
      esac
      ;;
    dir)
      DIR="$arg"
      STATE=config
      ;;
    macosx)
      case "$arg" in
        10.6) MACOSX=10.6 ;;
        10.7) MACOSX=10.7 ;;
        10.8) MACOSX=10.8 ;;
        native) MACOSX=native ;;
        *) usage_error "Invalid Argument '$arg'" 4
      esac
      STATE=config
      ;;
    *)
      usage_error "Invalid Argument '$arg'" 1
  esac
done
if [ "$STATE" != "config" ]; then
  usage_error "Trailing Argument" 5
fi

pushd $(echo $0 | sed 's/\/[^\/]*$//')
if [ $? -ne 0 ]; then exit -4; fi

rm assets/stderr.txt
rm assets/stdout.txt
rm -r Resources
rm *.plist

rm bin/x32/*.exp
rm bin/x32/*.idb
rm bin/x32/*.ilk
rm bin/x32/*.pdb
rm bin/x64/*.exp
rm bin/x64/*.idb
rm bin/x64/*.ilk
rm bin/x64/*.pdb
rm -r build/vs2010/ipch
rm -r build/vs2010/obj
rm build/vs2010/*.sdf
rm build/*.log
rm build/backupenv.bat
rm *\ Debug.exe
rm *.idb
rm *.ilk
rm *.pdb

rm -r build/linux/obj
rm game_d32
rm game_d64
rm lsbappchk_filtered.txt
rm lsbappchk_full.txt

if [ "$BUILD" == "all" ]; then
  rm -r bin/d32
  rm -r bin/d64
  rm bin/x32/*.lib
  rm bin/x64/*.lib

  rm -r lib/d32
  rm -r lib/d64
  rm lib/x32/liblocal_SDLmain.a
  rm lib/x64/liblocal_SDLmain.a

  rm -r lib/univ_d
else
  rm bin/d32/game.exe
  rm bin/d32/game.idb
  rm bin/d32/game.pdb
  rm bin/d32/*.exp
  rm bin/d32/*.idb
  rm bin/d32/*.ilk
  rm bin/d64/game.exe
  rm bin/d64/game.idb
  rm bin/d64/game.pdb
  rm bin/d64/*.exp
  rm bin/d64/*.idb
  rm bin/d64/*.ilk
fi

rm -r build/macosx/obj
rm -r build/xcode3/build
rm -r build/xcode4/build
rm -r game_d.app

popd
exit



:WINDOWS 



SET DP0=%~dp0
SET BUILD=mine
SET CONFIG=release32
SET DIR=build
SET MACOSX=native

SET STATE=config
:NEXTARG 
IF "%1" == "" GOTO DONEARGS 

IF "%STATE%"=="build" (
  IF "%1"=="all" (
    SET BUILD=all
  ) ELSE ( IF "%1"=="mine" (
    SET BUILD=mine
  ) ELSE (
    ECHO(
    ECHO Error: Invalid Argument '%1'
    GOTO ARGERROR 
  ))
  SET STATE=config
) ELSE ( IF "%STATE%"=="config" (
  IF "%1"=="--build" (
    SET STATE=build
  ) ELSE ( IF "%1"=="--macosx" (
    SET STATE=macosx
  ) ELSE ( IF "%1"=="--dir" (
    SET STATE=dir
  ) ELSE ( IF "%1"=="debug" (
    SET CONFIG=debug
  ) ELSE ( IF "%1"=="debug32" (
    SET CONFIG=debug32
  ) ELSE ( IF "%1"=="debug64" (
    SET CONFIG=debug64
  ) ELSE ( IF "%1"=="release" (
    SET CONFIG=release
  ) ELSE ( IF "%1"=="release32" (
    SET CONFIG=release32
  ) ELSE ( IF "%1"=="release64" (
    SET CONFIG=release64
  ) ELSE (
    ECHO(
    ECHO Error: Invalid Argument '%1'
    GOTO ARGERROR 
  )))))))))
) ELSE ( IF "%STATE%"=="dir" (
  SET DIR=%1
  SET STATE=config
) ELSE ( IF "%STATE%"=="macosx" (
  IF "%1"=="10.6" (
    SET MACOSX=10.6
  ) ELSE ( IF "%1"=="10.7" (
    SET MACOSX=10.7
  ) ELSE ( IF "%1"=="10.8" (
    SET MACOSX=10.8
  ) ELSE ( IF "%1"=="native" (
    SET MACOSX=native
  ) ELSE (
    ECHO(
    ECHO Error: Invalid Argument '%1'
    GOTO ARGERROR 
  ))))
  SET STATE=config
) ELSE (
  ECHO(
  ECHO Error: Invalid Argument '%1'
  GOTO ARGERROR 
))))

SHIFT
GOTO NEXTARG 

:ARGERROR 

ECHO(
ECHO Usage: multi-build_sh.bat [options]
ECHO(
ECHO   --build=all       game and all dependencies
ECHO           mine      game only (default)

EXIT /B 1

:DONEARGS 

IF NOT "%STATE%"=="config" (
  ECHO(
  ECHO Error: Trailing Argument
  GOTO ARGERROR 
)


DEL /Q "%DP0%\assets\stderr.txt"
DEL /Q "%DP0%\assets\stdout.txt"
RMDIR /S /Q "%DP0%\Resources"
DEL /Q "%DP0%\*.plist"

DEL /Q "%DP0%\bin\x32\*.exp"
DEL /Q "%DP0%\bin\x32\*.idb"
DEL /Q "%DP0%\bin\x32\*.ilk"
DEL /Q "%DP0%\bin\x32\*.pdb"
DEL /Q "%DP0%\bin\x64\*.exp"
DEL /Q "%DP0%\bin\x64\*.idb"
DEL /Q "%DP0%\bin\x64\*.ilk"
DEL /Q "%DP0%\bin\x64\*.pdb"
RMDIR /S /Q "%DP0%\build\vs2010\ipch"
RMDIR /S /Q "%DP0%\build\vs2010\obj"
DEL /Q "%DP0%\build\vs2010\*.sdf"
DEL /Q "%DP0%\build\*.log"
DEL /Q "%DP0%\build\backupenv.bat"
DEL /Q "%DP0%\* Debug.exe"
DEL /Q "%DP0%\*.idb"
DEL /Q "%DP0%\*.ilk"
DEL /Q "%DP0%\*.pdb"

RMDIR /S /Q "%DP0%\build\linux\obj"
DEL /Q "%DP0%\game_d32"
DEL /Q "%DP0%\game_d64"
DEL /Q "%DP0%\lsbappchk_filtered.txt"
DEL /Q "%DP0%\lsbappchk_full.txt"

RMDIR /S /Q "%DP0%\build\macosx\obj"
RMDIR /S /Q "%DP0%\build\xcode3\build"
RMDIR /S /Q "%DP0%\build\xcode4\build"
RMDIR /S /Q "%DP0%\game_d.app"

IF "%BUILD%"=="all" (
  RMDIR /S /Q "%DP0%\bin\d32"
  RMDIR /S /Q "%DP0%\bin\d64"
  DEL /Q "%DP0%\bin\x32\*.lib"
  DEL /Q "%DP0%\bin\x64\*.lib"

  RMDIR /S /Q "%DP0%\lib\d32"
  RMDIR /S /Q "%DP0%\lib\d64"
  DEL /Q "%DP0%\lib\x32\liblocal_SDLmain.a"
  DEL /Q "%DP0%\lib\x64\liblocal_SDLmain.a"

  RMDIR /S /Q "%DP0%\lib\univ_d"
) ELSE (
  DEL /Q "%DP0%\bin\d32\game.exe"
  DEL /Q "%DP0%\bin\d32\game.idb"
  DEL /Q "%DP0%\bin\d32\game.pdb"
  DEL /Q "%DP0%\bin\d32\*.exp"
  DEL /Q "%DP0%\bin\d32\*.idb"
  DEL /Q "%DP0%\bin\d32\*.ilk"
  DEL /Q "%DP0%\bin\d64\game.exe"
  DEL /Q "%DP0%\bin\d64\game.idb"
  DEL /Q "%DP0%\bin\d64\game.pdb"
  DEL /Q "%DP0%\bin\d64\*.exp"
  DEL /Q "%DP0%\bin\d64\*.idb"
  DEL /Q "%DP0%\bin\d64\*.ilk"
)



EXIT /B
