/* This file is part of the Zenipex Library (zenilib).
 * Copyright (C) 2011 Mitchell Keith Bloch (bazald).
 *
 * zenilib is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * zenilib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with zenilib.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * \class Zeni::Shader_System
 *
 * \ingroup zenilib
 *
 * \brief A rendering context for Cg
 *
 * This Singleton is used by Shaders and should not need direct interaction.
 * As this functionality is currently incomplete, its design will need to
 * be reassessed later.
 * Perhaps it is not even correct to use the Singleton patter.  (Though it
 * has made initial development simpler.)
 *
 * \author bazald
 *
 * Contact: bazald@zenipex.com
 */

/**
 * \class Zeni::Shader
 *
 * \ingroup zenilib
 *
 * \brief A base class for Vertex_Shader and Fragment_Shader
 *
 * Functionality shared between Vertex_Shader and Fragment_Shader is stored here.
 *
 * \author bazald
 *
 * Contact: bazald@zenipex.com
 */

/**
 * \class Zeni::Vertex_Shader
 *
 * \ingroup zenilib
 *
 * \brief A shader for manipulating vertex data as it is streamed through the GPU.
 *
 * \author bazald
 *
 * Contact: bazald@zenipex.com
 */

/**
 * \class Zeni::Fragment_Shader
 *
 * \ingroup zenilib
 *
 * \brief A shader for manipulating fragment/pixel data as it is streamed through the GPU.
 *
 * \author bazald
 *
 * Contact: bazald@zenipex.com
 */

#ifndef ZENI_SHADER_H
#define ZENI_SHADER_H

#include <Zeni/Core.h>
#include <Zeni/Hash_Map.h>

#ifndef DISABLE_CG

#ifndef DISABLE_DX9
#include <Cg/cgD3D9.h>
#endif

#ifndef DISABLE_GL
#if defined(REQUIRE_GL_ES)
#include <GLES/gl.h>
#elif defined(_MACOSX)
#include <GLEW/glew.h>
#else
#include <GL/glew.h>
#endif
#include <Cg/cgGL.h>
#endif

namespace Zeni {

  class ZENI_GRAPHICS_DLL Video_DX9;
  class ZENI_GRAPHICS_DLL Video_GL;

  class ZENI_GRAPHICS_DLL Shader_System {
    // Get reference to only instance;
    friend ZENI_GRAPHICS_DLL Shader_System & get_Shader_System(); ///< Get access to the singleton.

    Shader_System(const Shader_System &);
    Shader_System & operator=(const Shader_System &);

  public:
    Shader_System();
    ~Shader_System();

    inline const CGcontext & get_context() const;
    inline const CGprofile & get_vertex_profile() const;
    inline const CGprofile & get_fragment_profile() const;

#ifndef DISABLE_GL
    void init(Video_GL &screen);
#endif

#ifndef DISABLE_DX9
    void init(Video_DX9 &screen);
#endif

  private:
    CGcontext m_context;
    CGprofile m_cg_vertex_profile;
    CGprofile m_cg_fragment_profile;
  };

  ZENI_GRAPHICS_DLL Shader_System & get_Shader_System(); ///< Get access to the singleton.

  class ZENI_GRAPHICS_DLL Shader {
    Shader(const Shader &);
    Shader & operator=(const Shader &);

  public:
    Shader();
    ~Shader();

#ifndef DISABLE_GL
    void init(const String &filename, const String &entry_function, const CGprofile &profile, Video_GL &screen);
    void load(Video_GL &screen);
#endif

#ifndef DISABLE_DX9
    void init(const String &filename, const String &entry_function, const CGprofile &profile, Video_DX9 &screen);
    void load(Video_DX9 &screen);
#endif

    void compile();

  protected:
#ifndef DISABLE_GL
    void set(const CGprofile &profile, Video_GL &screen) const;
    void unset(const CGprofile &profile, Video_GL &screen) const;
#endif

#ifndef DISABLE_DX9
    void set(const CGprofile &profile, Video_DX9 &screen) const;
    void unset(const CGprofile &profile, Video_DX9 &screen) const;
#endif

  private:
    typedef Unordered_Map<String, std::pair<CGparameter, CGparameter> > Parameters;
#ifdef _WINDOWS
#pragma warning( push )
#pragma warning( disable : 4251 )
#endif
    Parameters m_parameters;
#ifdef _WINDOWS
#pragma warning( pop )
#endif

    void initialize_parameter(const String &parameter_name);
    CGparameter get_from_parameter(const String &parameter_name);
    // cgSetParameter*() routines
    void connect_parameter(const String &parameter_name);

    CGprogram m_program;
  };

  class ZENI_GRAPHICS_DLL Vertex_Shader : public Shader {
    Vertex_Shader(const Vertex_Shader &);
    Vertex_Shader & operator=(const Vertex_Shader &);

  public:
    inline Vertex_Shader(const String &filename, const String &entry_function = "main");

#ifndef DISABLE_GL
    inline void set(Video_GL &screen) const;
    inline void unset(Video_GL &screen) const;
#endif

#ifndef DISABLE_DX9
    inline void set(Video_DX9 &screen) const;
    inline void unset(Video_DX9 &screen) const;
#endif
  };

  class ZENI_GRAPHICS_DLL Fragment_Shader : public Shader {
    Fragment_Shader(const Fragment_Shader &);
    Fragment_Shader & operator=(const Fragment_Shader &);

  public:
    inline Fragment_Shader(const String &filename, const String &entry_function = "main");

#ifndef DISABLE_GL
    inline void set(Video_GL &screen) const;
    inline void unset(Video_GL &screen) const;
#endif

#ifndef DISABLE_DX9
    inline void set(Video_DX9 &screen) const;
    inline void unset(Video_DX9 &screen) const;
#endif
  };

  struct ZENI_GRAPHICS_DLL Shader_System_Init_Failure : public Error {
    Shader_System_Init_Failure() : Error("Zeni Shader System Failed to Initialize Correctly") {}
  };

  struct ZENI_GRAPHICS_DLL Shader_Init_Failure : public Error {
    Shader_Init_Failure() : Error("Zeni Shader Failed to Initialize Correctly") {}
  };

  struct ZENI_GRAPHICS_DLL Shader_Bind_Failure : public Error {
    Shader_Bind_Failure() : Error("Zeni Shader Failed to Bind Correctly") {}
  };

  struct ZENI_GRAPHICS_DLL Shader_Parameter_Error : public Error {
    Shader_Parameter_Error() : Error("Zeni Shader Parameter Error") {}
  };

}

#endif

#endif
