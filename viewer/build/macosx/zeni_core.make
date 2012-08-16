# GNU Make project makefile autogenerated by Premake
ifndef config
  config=debug
endif

ifndef verbose
  SILENT = @
endif

ifndef CC
  CC = clang
endif

ifndef CXX
  CXX = clang++
endif

ifndef AR
  AR = ar
endif

ifndef RESCOMP
  ifdef WINDRES
    RESCOMP = $(WINDRES)
  else
    RESCOMP = windres
  endif
endif

ifeq ($(config),debug)
  OBJDIR     = obj/Debug/zeni_core
  TARGETDIR  = ../../lib/univ_d
  TARGET     = $(TARGETDIR)/libzeni_core_d.dylib
  DEFINES   += -D_MACOSX -D_DEBUG -DDEBUG -DTEST_NASTY_CONDITIONS -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT=
  INCLUDES  += -I../../jni/external/zenilib/zeni_core -I../../jni/external/zenilib/zeni -I../../jni/external/sdl_net -I../../jni/external/sdl -I../../jni/external/tinyxml
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -g -Wall -fPIC -Qunused-arguments -stdlib=libc++ -ffast-math -fpch-preprocess -Wall
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ_d -dynamiclib -flat_namespace -stdlib=libc++ -install_name @loader_path/libzeni_core_d.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -lzeni_d -llocal_SDL_d
  LDDEPS    += ../../lib/univ_d/libzeni_d.dylib ../../lib/univ_d/liblocal_SDL_d.dylib
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),release)
  OBJDIR     = obj/Release/zeni_core
  TARGETDIR  = ../../lib/univ
  TARGET     = $(TARGETDIR)/libzeni_core.dylib
  DEFINES   += -D_MACOSX -DNDEBUG -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT=
  INCLUDES  += -I../../jni/external/zenilib/zeni_core -I../../jni/external/zenilib/zeni -I../../jni/external/sdl_net -I../../jni/external/sdl -I../../jni/external/tinyxml
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -O2 -Wall -fPIC -Qunused-arguments -stdlib=libc++ -ffast-math -fpch-preprocess -Wall
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ -Wl,-x -dynamiclib -flat_namespace -stdlib=libc++ -install_name @loader_path/libzeni_core.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -lzeni -llocal_SDL
  LDDEPS    += ../../lib/univ/libzeni.dylib ../../lib/univ/liblocal_SDL.dylib
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),debuguniv)
  OBJDIR     = obj/Universal/Debug/zeni_core
  TARGETDIR  = ../../lib/univ_d
  TARGET     = $(TARGETDIR)/libzeni_core_d.dylib
  DEFINES   += -D_MACOSX -D_DEBUG -DDEBUG -DTEST_NASTY_CONDITIONS -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT=
  INCLUDES  += -I../../jni/external/zenilib/zeni_core -I../../jni/external/zenilib/zeni -I../../jni/external/sdl_net -I../../jni/external/sdl -I../../jni/external/tinyxml
  CPPFLAGS  +=  $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -g -Wall -arch i386 -arch x86_64 -fPIC -Qunused-arguments -stdlib=libc++ -ffast-math -fpch-preprocess -Wall
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ_d -dynamiclib -flat_namespace -arch i386 -arch x86_64 -stdlib=libc++ -install_name @loader_path/libzeni_core_d.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -lzeni_d -llocal_SDL_d
  LDDEPS    += ../../lib/univ_d/libzeni_d.dylib ../../lib/univ_d/liblocal_SDL_d.dylib
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),releaseuniv)
  OBJDIR     = obj/Universal/Release/zeni_core
  TARGETDIR  = ../../lib/univ
  TARGET     = $(TARGETDIR)/libzeni_core.dylib
  DEFINES   += -D_MACOSX -DNDEBUG -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT=
  INCLUDES  += -I../../jni/external/zenilib/zeni_core -I../../jni/external/zenilib/zeni -I../../jni/external/sdl_net -I../../jni/external/sdl -I../../jni/external/tinyxml
  CPPFLAGS  +=  $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -O2 -Wall -arch i386 -arch x86_64 -fPIC -Qunused-arguments -stdlib=libc++ -ffast-math -fpch-preprocess -Wall
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ -Wl,-x -dynamiclib -flat_namespace -arch i386 -arch x86_64 -stdlib=libc++ -install_name @loader_path/libzeni_core.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -lzeni -llocal_SDL
  LDDEPS    += ../../lib/univ/libzeni.dylib ../../lib/univ/liblocal_SDL.dylib
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

OBJECTS := \
	$(OBJDIR)/Core.o \
	$(OBJDIR)/Joysticks.o \
	$(OBJDIR)/Timer.o \

RESOURCES := \

SHELLTYPE := msdos
ifeq (,$(ComSpec)$(COMSPEC))
  SHELLTYPE := posix
endif
ifeq (/bin,$(findstring /bin,$(SHELL)))
  SHELLTYPE := posix
endif

.PHONY: clean prebuild prelink

all: $(TARGETDIR) $(OBJDIR) prebuild prelink $(TARGET)
	@:

$(TARGET): $(GCH) $(OBJECTS) $(LDDEPS) $(RESOURCES)
	@echo Linking zeni_core
	$(SILENT) $(LINKCMD)
	$(POSTBUILDCMDS)

$(TARGETDIR):
	@echo Creating $(TARGETDIR)
ifeq (posix,$(SHELLTYPE))
	$(SILENT) mkdir -p $(TARGETDIR)
else
	$(SILENT) mkdir $(subst /,\\,$(TARGETDIR))
endif

$(OBJDIR):
	@echo Creating $(OBJDIR)
ifeq (posix,$(SHELLTYPE))
	$(SILENT) mkdir -p $(OBJDIR)
else
	$(SILENT) mkdir $(subst /,\\,$(OBJDIR))
endif

clean:
	@echo Cleaning zeni_core
ifeq (posix,$(SHELLTYPE))
	$(SILENT) rm -f  $(TARGET)
	$(SILENT) rm -rf $(OBJDIR)
else
	$(SILENT) if exist $(subst /,\\,$(TARGET)) del $(subst /,\\,$(TARGET))
	$(SILENT) if exist $(subst /,\\,$(OBJDIR)) rmdir /s /q $(subst /,\\,$(OBJDIR))
endif

prebuild:
	$(PREBUILDCMDS)

prelink:
	$(PRELINKCMDS)

ifneq (,$(PCH))
$(GCH): $(PCH)
	@echo $(notdir $<)
ifeq (posix,$(SHELLTYPE))
	-$(SILENT) cp $< $(OBJDIR)
else
	$(SILENT) xcopy /D /Y /Q "$(subst /,\,$<)" "$(subst /,\,$(OBJDIR))" 1>nul
endif
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
endif

$(OBJDIR)/Core.o: ../../jni/external/zenilib/zeni_core/Core.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/Joysticks.o: ../../jni/external/zenilib/zeni_core/Joysticks.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"
$(OBJDIR)/Timer.o: ../../jni/external/zenilib/zeni_core/Timer.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o "$@" -c "$<"

-include $(OBJECTS:%.o=%.d)
