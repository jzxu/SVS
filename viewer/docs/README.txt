Zenipex Library           (Most Recently Updated at 0.4.0.0)
============================================================

Contents
========
1. Introduction
2. License
3. Supported Platforms
4. Documentation



1. Introduction
===============

The Zenipex Library (zenilib) is a game development 
framework, created with the goal of allowing someone with 
some knowledge of C++ to create 3D games for both Windows 
and Linux platforms (and hopefully others) quickly.

The original focuses of the project were to abstract away 
some basic differences between OpenGL and Direct3D 9 and to 
create an object-oriented framework for game development.  
It was never the goal to abstract away all differences 
between OpenGL and Direct3D or to create a fully optimzed 
game development library with every feature you could 
possibly want.

Truly it started as an exercise in learning both OpenGL and 
Direct3D, with the hope that it would become something 
useful to me if to nobody else.  Hopefully someone else in 
Wolverine Soft will use the thing too ;)

Since its initial release, ease of use has been a continual 
focus.  Features have been added, and performance has been 
generally improved as well.

~bazald



2. License/Authors
==================

The Zenipex Library is an open source project.  For 
details please refer to the first comment in any given file 
and refer to the included copies of the GPL and LGPL.

Keep in mind that to fully comply with the GPL, you must 
include a copy of the GPL with your your releases using 
zenilib.  Additionally, you should state what parts of your 
release are bound by the GPL.  Finally, you must meet one of 
the following conditions.

 1) A copy of the GPL'd source code is included.

 2) A written offer to provide the GPL'd source code is 
    included, and you keep an archive of all GPL'd source 
    code ready at all times.

In my opinion, the first option is easier, as it requires 
no effort post-release.  Additionally, in the second case, 
you must be willing to provide the source to anyone who 
asks -- not just the people who download your library.

If you have any questions about the GPL, feel free to shoot 
me an e-mail.

bazald -at- zenipex -dot- com



3. Supported Platforms
======================
 
This software is known to compile and run on:
 
- Windows: XP, Vista, 7
- MacOS X: 10.6 and up
- Linux: Ubuntu (and derivatives), Red Hat Enterprise Linux

It should be possible to build it on any platform supported by:

DirectX    9.0c      (Proprietary)           http://msdn.microsoft.com/directx/sdk/
OpenAL     1.1       (Proprietary)           http://connect.creativelabs.com/openal/Downloads/Forms/AllItems.aspx
OpenGL               (Proprietary)

FreeType2  2.4.6     (BSD License)           http://sourceforge.net/project/showfiles.php?group_id=3157
GLEW       1.6.0     (BSD License)           http://glew.sourceforge.net/
lib3ds     2.0 RC 1  (GNU LGPL)              http://lib3ds.sourceforge.net/
libogg     1.2.2     (Xiph BSD-like License) http://xiph.org/downloads/
libpng     1.2.46    (zlib/libpng License)   http://www.libpng.org/pub/png/libpng.html
libtiff    3.9.5     (BSD License)           http://gnuwin32.sourceforge.net/packages/tiff.htm
libvorbis  1.3.2     (Xiph BSD-like License) http://xiph.org/downloads/
SDL        1.2.14    (GNU LGPL)              http://www.libsdl.org/download-1.2.php
SDL_image  1.2.10    (GNU LGPL)              http://www.libsdl.org/projects/SDL_image/
SDL_net    1.2.7     (GNU LGPL)              http://www.libsdl.org/projects/SDL_net/
SDL_ttf    2.0.10    (GNU LGPL)              http://www.libsdl.org/projects/SDL_ttf/
TinyXML    2.6.2     (zlib/libpng License)   http://sourceforge.net/project/showfiles.php?group_id=13559
zlib       1.2.5     (zlib/libpng License)   http://www.zlib.net/

4. Documentation
================

Documentation is available in any format supported by Doxygen.
You can build any format by modifying the ./Doxyfile file to 
access the correct directories on your system.
