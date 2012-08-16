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
  OBJDIR     = obj/Debug/local_vorbisfile
  TARGETDIR  = ../../lib/univ_d
  TARGET     = $(TARGETDIR)/liblocal_vorbisfile_d.dylib
  DEFINES   += -D_MACOSX -D_DEBUG -DDEBUG -DTEST_NASTY_CONDITIONS -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT= -D_REENTRANT -DHAVE_ALLOCA=1 -DHAVE_ALLOCA_H=1 -DHAVE_DLFCN_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STDINT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRINGS_H=1 -DHAVE_STRING_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_UNISTD_H=1 -DLT_OBJDIR=\".libs/\" -DPACKAGE=\"libvorbis\" -DPACKAGE_BUGREPORT=\"vorbis-dev@xiph.org\" -DPACKAGE_NAME=\"libvorbis\" -DPACKAGE_STRING=\"libvorbis=1.3.2\" -DPACKAGE_TARNAME=\"libvorbis\" -DPACKAGE_VERSION=\"1.3.2\" -DSTDC_HEADERS=1 -DVERSION=\"1.3.2\"
  INCLUDES  += -I../../jni/external/libvorbis -I../../jni/external/libvorbis/include -I../../jni/external/libogg/include
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -g -fPIC -Qunused-arguments -stdlib=libc++ -w
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ_d -dynamiclib -flat_namespace -stdlib=libc++ -install_name @loader_path/liblocal_vorbisfile_d.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -llocal_vorbis_d -llocal_ogg_d
  LDDEPS    += ../../lib/univ_d/liblocal_vorbis_d.dylib ../../lib/univ_d/liblocal_ogg_d.dylib
  LINKCMD    = $(CC) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),release)
  OBJDIR     = obj/Release/local_vorbisfile
  TARGETDIR  = ../../lib/univ
  TARGET     = $(TARGETDIR)/liblocal_vorbisfile.dylib
  DEFINES   += -D_MACOSX -DNDEBUG -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT= -D_REENTRANT -DHAVE_ALLOCA=1 -DHAVE_ALLOCA_H=1 -DHAVE_DLFCN_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STDINT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRINGS_H=1 -DHAVE_STRING_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_UNISTD_H=1 -DLT_OBJDIR=\".libs/\" -DPACKAGE=\"libvorbis\" -DPACKAGE_BUGREPORT=\"vorbis-dev@xiph.org\" -DPACKAGE_NAME=\"libvorbis\" -DPACKAGE_STRING=\"libvorbis=1.3.2\" -DPACKAGE_TARNAME=\"libvorbis\" -DPACKAGE_VERSION=\"1.3.2\" -DSTDC_HEADERS=1 -DVERSION=\"1.3.2\"
  INCLUDES  += -I../../jni/external/libvorbis -I../../jni/external/libvorbis/include -I../../jni/external/libogg/include
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -O2 -fPIC -Qunused-arguments -stdlib=libc++ -w
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ -Wl,-x -dynamiclib -flat_namespace -stdlib=libc++ -install_name @loader_path/liblocal_vorbisfile.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -llocal_vorbis -llocal_ogg
  LDDEPS    += ../../lib/univ/liblocal_vorbis.dylib ../../lib/univ/liblocal_ogg.dylib
  LINKCMD    = $(CC) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),debuguniv)
  OBJDIR     = obj/Universal/Debug/local_vorbisfile
  TARGETDIR  = ../../lib/univ_d
  TARGET     = $(TARGETDIR)/liblocal_vorbisfile_d.dylib
  DEFINES   += -D_MACOSX -D_DEBUG -DDEBUG -DTEST_NASTY_CONDITIONS -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT= -D_REENTRANT -DHAVE_ALLOCA=1 -DHAVE_ALLOCA_H=1 -DHAVE_DLFCN_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STDINT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRINGS_H=1 -DHAVE_STRING_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_UNISTD_H=1 -DLT_OBJDIR=\".libs/\" -DPACKAGE=\"libvorbis\" -DPACKAGE_BUGREPORT=\"vorbis-dev@xiph.org\" -DPACKAGE_NAME=\"libvorbis\" -DPACKAGE_STRING=\"libvorbis=1.3.2\" -DPACKAGE_TARNAME=\"libvorbis\" -DPACKAGE_VERSION=\"1.3.2\" -DSTDC_HEADERS=1 -DVERSION=\"1.3.2\"
  INCLUDES  += -I../../jni/external/libvorbis -I../../jni/external/libvorbis/include -I../../jni/external/libogg/include
  CPPFLAGS  +=  $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -g -arch i386 -arch x86_64 -fPIC -Qunused-arguments -stdlib=libc++ -w
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ_d -dynamiclib -flat_namespace -arch i386 -arch x86_64 -stdlib=libc++ -install_name @loader_path/liblocal_vorbisfile_d.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -llocal_vorbis_d -llocal_ogg_d
  LDDEPS    += ../../lib/univ_d/liblocal_vorbis_d.dylib ../../lib/univ_d/liblocal_ogg_d.dylib
  LINKCMD    = $(CC) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),releaseuniv)
  OBJDIR     = obj/Universal/Release/local_vorbisfile
  TARGETDIR  = ../../lib/univ
  TARGET     = $(TARGETDIR)/liblocal_vorbisfile.dylib
  DEFINES   += -D_MACOSX -DNDEBUG -DDISABLE_CPP11 -DGLEW_NO_GLU -DDISABLE_CG -DOGG_DLL= -DVORBIS_DLL= -DVORBISFILE_DLL= -DDISABLE_DX9 -DDISABLE_WGL -DTINYXML_DLL= -DTINYXML_EXT= -DZENI_DLL= -DZENI_EXT= -DZENI_AUDIO_DLL= -DZENI_AUDIO_EXT= -DZENI_CORE_DLL= -DZENI_CORE_EXT= -DZENI_GRAPHICS_DLL= -DZENI_GRAPHICS_EXT= -DZENI_NET_DLL= -DZENI_NET_EXT= -DZENI_REST_DLL= -DZENI_REST_EXT= -D_REENTRANT -DHAVE_ALLOCA=1 -DHAVE_ALLOCA_H=1 -DHAVE_DLFCN_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STDINT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRINGS_H=1 -DHAVE_STRING_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_UNISTD_H=1 -DLT_OBJDIR=\".libs/\" -DPACKAGE=\"libvorbis\" -DPACKAGE_BUGREPORT=\"vorbis-dev@xiph.org\" -DPACKAGE_NAME=\"libvorbis\" -DPACKAGE_STRING=\"libvorbis=1.3.2\" -DPACKAGE_TARNAME=\"libvorbis\" -DPACKAGE_VERSION=\"1.3.2\" -DSTDC_HEADERS=1 -DVERSION=\"1.3.2\"
  INCLUDES  += -I../../jni/external/libvorbis -I../../jni/external/libvorbis/include -I../../jni/external/libogg/include
  CPPFLAGS  +=  $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -O2 -arch i386 -arch x86_64 -fPIC -Qunused-arguments -stdlib=libc++ -w
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -L../../lib/univ -Wl,-x -dynamiclib -flat_namespace -arch i386 -arch x86_64 -stdlib=libc++ -install_name @loader_path/liblocal_vorbisfile.dylib
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LIBS      += -llocal_vorbis -llocal_ogg
  LDDEPS    += ../../lib/univ/liblocal_vorbis.dylib ../../lib/univ/liblocal_ogg.dylib
  LINKCMD    = $(CC) -o $(TARGET) $(OBJECTS) $(RESOURCES) $(ARCH) $(LIBS) $(LDFLAGS)
  define PREBUILDCMDS
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

OBJECTS := \
	$(OBJDIR)/vorbisfile.o \

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
	@echo Linking local_vorbisfile
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
	@echo Cleaning local_vorbisfile
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

$(OBJDIR)/vorbisfile.o: ../../jni/external/libvorbis/lib/vorbisfile.c
	@echo $(notdir $<)
	$(SILENT) $(CC) $(CFLAGS) -o "$@" -c "$<"

-include $(OBJECTS:%.o=%.d)