The Spatial Visual System (SVS) is a module of the Soar cognitive
architecture. See

http://sitemaker.umich.edu/soar/home

Usage
=====
SVS is meant to be used as a component of Soar. In order to compile
Soar with SVS, first clone this repo into the directory
SoarSuite/Core/SVS. Next, you need to patch some kernel code to put an
svs instance in the agent. If you're on Windows, you may need to
obtain the "patch" program. You can get it at

http://gnuwin32.sourceforge.net/packages/patch.htm

After you have patch, run this command from the SoarSuite top
directory:

patch -p0 < Core/SVS/patch

If you're not in Linux, see below about compiling Bullet Physics,
which is a dependency of SVS. After that, you should be able to
compile Soar normally.

Compiling Bullet Physics
========================
SVS uses code from the Bullet Physics library to do collision
detection. 32 and 64 bit precompiled libraries for Linux are included
in the repo, so if you're on Linux, you don't need to install Bullet
separately. If you're on Mac OS X or Windows, you need to compile
Bullet yourself. You can download the source code at

http://code.google.com/p/bullet/downloads/list

Bullet uses cmake for cross platform compilation. Please install cmake
if you don't have it already. Get it at

http://www.cmake.org/cmake/resources/software.html

After obtaining cmake and the Bullet source, follow these steps:

1. Expand the source archive to some directory, say bullet_source.

2. Create a directory bullet_source/temp, then cd into that directory.

3. Run this command:

   cmake -DCMAKE_INSTALL_PREFIX=<install dir> -DBUILD_DEMOS=off \
         -DBUILD_EXTRAS=off -DINSTALL_LIBS=on -DBUILD_SHARED_LIBS=on

   where <install dir> is any directory you want to install the
   compiled code to.

4. In *nix and Mac OS X, run "make install". In Windows, open the
   generated Visual Studio solution, and build in release mode.

5. If you installed the libraries to a non-standard location, make
   sure the compiler can find the libraries by adding the location to
   the appropriate environment variable:
   
   * In *nix, use LD_LIBRARY_PATH
   * In Mac OS X, use DYLD_LIBRARY_PATH
   * In Windows, use PATH
