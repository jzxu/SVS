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
  OBJDIR     = obj/Debug/local_png
  TARGETDIR  = ../../lib/univ_d
  TARGET     = $(TARGETDIR)/liblocal_png_d.dylib
  DEFINES   += -D_MACOSX -D_DEBUG -DDEBUG -DTEST_NASTY_CONDITIONS -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT=
  INCLUDES  += -I../../jni/external/zlib
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -g -fPIC -Qunused-arguments -stdlib=libc++ -w
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ_d -dynamiclib -flat_namespace -stdlib=libc++ -install_name @loader_path/liblocal_png_d.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -llocal_z_d
  LDDEPS    += ../../lib/univ_d/liblocal_z_d.dylib
  LINKCMD    = $(CC) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),release)
  OBJDIR     = obj/Release/local_png
  TARGETDIR  = ../../lib/univ
  TARGET     = $(TARGETDIR)/liblocal_png.dylib
  DEFINES   += -D_MACOSX -DNDEBUG -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT=
  INCLUDES  += -I../../jni/external/zlib
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -O2 -fPIC -Qunused-arguments -stdlib=libc++ -w
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ -Wl,-x -dynamiclib -flat_namespace -stdlib=libc++ -install_name @loader_path/liblocal_png.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -llocal_z
  LDDEPS    += ../../lib/univ/liblocal_z.dylib
  LINKCMD    = $(CC) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),debuguniv)
  OBJDIR     = obj/Universal/Debug/local_png
  TARGETDIR  = ../../lib/univ_d
  TARGET     = $(TARGETDIR)/liblocal_png_d.dylib
  DEFINES   += -D_MACOSX -D_DEBUG -DDEBUG -DTEST_NASTY_CONDITIONS -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT=
  INCLUDES  += -I../../jni/external/zlib
  CPPFLAGS  +=  $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -g -arch i386 -arch x86_64 -fPIC -Qunused-arguments -stdlib=libc++ -w
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ_d -dynamiclib -flat_namespace -arch i386 -arch x86_64 -stdlib=libc++ -install_name @loader_path/liblocal_png_d.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -llocal_z_d
  LDDEPS    += ../../lib/univ_d/liblocal_z_d.dylib
  LINKCMD    = $(CC) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),releaseuniv)
  OBJDIR     = obj/Universal/Release/local_png
  TARGETDIR  = ../../lib/univ
  TARGET     = $(TARGETDIR)/liblocal_png.dylib
  DEFINES   += -D_MACOSX -DNDEBUG -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT=
  INCLUDES  += -I../../jni/external/zlib
  CPPFLAGS  +=  $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -O2 -arch i386 -arch x86_64 -fPIC -Qunused-arguments -stdlib=libc++ -w
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ -Wl,-x -dynamiclib -flat_namespace -arch i386 -arch x86_64 -stdlib=libc++ -install_name @loader_path/liblocal_png.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -llocal_z
  LDDEPS    += ../../lib/univ/liblocal_z.dylib
  LINKCMD    = $(CC) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

OBJECTS := \
	$(OBJDIR)/png.o \
	$(OBJDIR)/pngerror.o \
	$(OBJDIR)/pngget.o \
	$(OBJDIR)/pngmem.o \
	$(OBJDIR)/pngpread.o \
	$(OBJDIR)/pngread.o \
	$(OBJDIR)/pngrio.o \
	$(OBJDIR)/pngrtran.o \
	$(OBJDIR)/pngrutil.o \
	$(OBJDIR)/pngset.o \
	$(OBJDIR)/pngtest.o \
	$(OBJDIR)/pngtrans.o \
	$(OBJDIR)/pngwio.o \
	$(OBJDIR)/pngwrite.o \
	$(OBJDIR)/pngwtran.o \
	$(OBJDIR)/pngwutil.o \

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
	@echo Linking local_png
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
	@echo Cleaning local_png
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
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
endif

$(OBJDIR)/png.o: ../../jni/external/libpng/png.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngerror.o: ../../jni/external/libpng/pngerror.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngget.o: ../../jni/external/libpng/pngget.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngmem.o: ../../jni/external/libpng/pngmem.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngpread.o: ../../jni/external/libpng/pngpread.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngread.o: ../../jni/external/libpng/pngread.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngrio.o: ../../jni/external/libpng/pngrio.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngrtran.o: ../../jni/external/libpng/pngrtran.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngrutil.o: ../../jni/external/libpng/pngrutil.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngset.o: ../../jni/external/libpng/pngset.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngtest.o: ../../jni/external/libpng/pngtest.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngtrans.o: ../../jni/external/libpng/pngtrans.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngwio.o: ../../jni/external/libpng/pngwio.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngwrite.o: ../../jni/external/libpng/pngwrite.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngwtran.o: ../../jni/external/libpng/pngwtran.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"
$(OBJDIR)/pngwutil.o: ../../jni/external/libpng/pngwutil.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"

-include $(OBJECTS:%.o=%.d)
