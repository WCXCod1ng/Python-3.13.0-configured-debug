# Generated automatically from Makefile.pre by makesetup.
# Top-level Makefile for Python
#
# As distributed, this file is called Makefile.pre.in; it is processed
# into the real Makefile by running the script ./configure, which
# replaces things like @spam@ with values appropriate for your system.
# This means that if you edit Makefile, your changes get lost the next
# time you run the configure script.  Ideally, you can do:
#
#	./configure
#	make
#	make test
#	make install
#
# If you have a previous version of Python installed that you don't
# want to overwrite, you can use "make altinstall" instead of "make
# install".  Refer to the "Installing" section in the README file for
# additional details.
#
# See also the section "Build instructions" in the README file.

# === Variables set by makesetup ===

MODBUILT_NAMES=      array  _asyncio  _bisect  _contextvars  _csv  _heapq  _json  _lsprof  _opcode  _pickle  _queue  _random  _struct  _interpreters  _interpchannels  _interpqueues  _zoneinfo  math  cmath  _statistics  _datetime  _decimal  binascii  _bz2  _lzma  zlib  _dbm  _gdbm  readline  _md5  _sha1  _sha2  _sha3  _blake2  pyexpat  _elementtree  _codecs_cn  _codecs_hk  _codecs_iso2022  _codecs_jp  _codecs_kr  _codecs_tw  _multibytecodec  unicodedata  fcntl  grp  mmap  _posixsubprocess  resource  select  _socket  syslog  termios  _posixshmem  _multiprocessing  _ctypes  _curses  _curses_panel  _sqlite3  _uuid  _tkinter  xxsubtype  _xxtestfuzz  _testbuffer  _testinternalcapi  _testcapi  _testlimitedcapi  _testclinic  _testclinic_limited  _testimportmultiple  _testmultiphase  _testsinglephase  _testexternalinspection  _ctypes_test  xxlimited  xxlimited_35  atexit  faulthandler  posix  _signal  _tracemalloc  _suggestions  _codecs  _collections  errno  _io  itertools  _sre  _sysconfig  _thread  time  _typing  _weakref  _abc  _functools  _locale  _operator  _stat  _symtable  pwd
MODSHARED_NAMES=    array _asyncio _bisect _contextvars _csv _heapq _json _lsprof _opcode _pickle _queue _random _struct _interpreters _interpchannels _interpqueues _zoneinfo math cmath _statistics _datetime _decimal binascii _bz2 _lzma zlib _dbm _gdbm readline _md5 _sha1 _sha2 _sha3 _blake2 pyexpat _elementtree _codecs_cn _codecs_hk _codecs_iso2022 _codecs_jp _codecs_kr _codecs_tw _multibytecodec unicodedata fcntl grp mmap _posixsubprocess resource select _socket syslog termios _posixshmem _multiprocessing _ctypes _curses _curses_panel _sqlite3 _uuid _tkinter xxsubtype _xxtestfuzz _testbuffer _testinternalcapi _testcapi _testlimitedcapi _testclinic _testclinic_limited _testimportmultiple _testmultiphase _testsinglephase _testexternalinspection _ctypes_test xxlimited xxlimited_35
MODDISABLED_NAMES= 
MODOBJS=             Modules/atexitmodule.o  Modules/faulthandler.o  Modules/posixmodule.o  Modules/signalmodule.o  Modules/_tracemalloc.o  Modules/_suggestions.o  Modules/_codecsmodule.o  Modules/_collectionsmodule.o  Modules/errnomodule.o  Modules/_io/_iomodule.o Modules/_io/iobase.o Modules/_io/fileio.o Modules/_io/bytesio.o Modules/_io/bufferedio.o Modules/_io/textio.o Modules/_io/stringio.o  Modules/itertoolsmodule.o  Modules/_sre/sre.o  Modules/_sysconfig.o  Modules/_threadmodule.o  Modules/timemodule.o  Modules/_typingmodule.o  Modules/_weakref.o  Modules/_abc.o  Modules/_functoolsmodule.o  Modules/_localemodule.o  Modules/_operator.o  Modules/_stat.o  Modules/symtablemodule.o  Modules/pwdmodule.o
MODLIBS=           $(LOCALMODLIBS) $(BASEMODLIBS)

# === Variables set by configure
VERSION=	3.13
srcdir=		.

abs_srcdir=	/home/wang/code/Python-3.13.0-configured
abs_builddir=	/home/wang/code/Python-3.13.0-configured


CC=		gcc
CXX=		g++
LINKCC=		$(PURIFY) $(CC)
AR=		ar
READELF=	@READELF@
SOABI=		cpython-313d-x86_64-linux-gnu
ABIFLAGS=	d
ABI_THREAD=	
LDVERSION=	$(VERSION)$(ABIFLAGS)
LIBPYTHON=
GITVERSION=	
GITTAG=		
GITBRANCH=	
PGO_PROF_GEN_FLAG=-fprofile-generate
PGO_PROF_USE_FLAG=-fprofile-use -fprofile-correction
LLVM_PROF_MERGER=true
LLVM_PROF_FILE=
LLVM_PROF_ERR=no
DTRACE=         
DFLAGS=         
DTRACE_HEADERS= 
DTRACE_OBJS=    
DSYMUTIL=       
DSYMUTIL_PATH=  

GNULD=		yes

# Shell used by make (some versions default to the login shell, which is bad)
SHELL=		/bin/sh -e

# Use this to make a link between python$(VERSION) and python in $(BINDIR)
LN=		ln

# Portable install script (configure doesn't always guess right)
INSTALL=	/usr/bin/install -c
INSTALL_PROGRAM=${INSTALL}
INSTALL_SCRIPT= ${INSTALL}
INSTALL_DATA=	${INSTALL} -m 644
# Shared libraries must be installed with executable mode on some systems;
# rather than figuring out exactly which, we always give them executable mode.
INSTALL_SHARED= ${INSTALL} -m 755

MKDIR_P=	/usr/bin/mkdir -p

MAKESETUP=      $(srcdir)/Modules/makesetup

# Compiler options
OPT=		-g -Og -Wall
BASECFLAGS=	 -fno-strict-overflow -Wsign-compare -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer
BASECPPFLAGS=	
CONFIGURE_CFLAGS=	
# CFLAGS_NODIST is used for building the interpreter and stdlib C extensions.
# Use it when a compiler flag should _not_ be part of the distutils CFLAGS
# once Python is installed (Issue #21121).
CONFIGURE_CFLAGS_NODIST= -D_Py_TIER2=1 -D_Py_JIT -fno-semantic-interposition -std=c11 -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -Wstrict-prototypes -Werror=implicit-function-declaration -fvisibility=hidden
# LDFLAGS_NODIST is used in the same manner as CFLAGS_NODIST.
# Use it when a linker flag should _not_ be part of the distutils LDFLAGS
# once Python is installed (bpo-35257)
CONFIGURE_LDFLAGS_NODIST= -fno-semantic-interposition
# LDFLAGS_NOLTO is an extra flag to disable lto. It is used to speed up building
# of _bootstrap_python and _freeze_module tools, which don't need LTO.
CONFIGURE_LDFLAGS_NOLTO=
CONFIGURE_CPPFLAGS=	
CONFIGURE_LDFLAGS=	
# Avoid assigning CFLAGS, LDFLAGS, etc. so users can use them on the
# command line to append to these values without stomping the pre-set
# values.
PY_CFLAGS=	$(BASECFLAGS) $(OPT) $(CONFIGURE_CFLAGS) $(CFLAGS) $(EXTRA_CFLAGS)
PY_CFLAGS_NODIST=$(CONFIGURE_CFLAGS_NODIST) $(CFLAGS_NODIST) -I$(srcdir)/Include/internal -I$(srcdir)/Include/internal/mimalloc
# Both CPPFLAGS and LDFLAGS need to contain the shell's value for setup.py to
# be able to build extension modules using the directories specified in the
# environment variables
PY_CPPFLAGS=	$(BASECPPFLAGS) -I. -I$(srcdir)/Include $(CONFIGURE_CPPFLAGS) $(CPPFLAGS)
PY_LDFLAGS=	$(CONFIGURE_LDFLAGS) $(LDFLAGS)
PY_LDFLAGS_NODIST=$(CONFIGURE_LDFLAGS_NODIST) $(LDFLAGS_NODIST)
PY_LDFLAGS_NOLTO=$(PY_LDFLAGS) $(CONFIGURE_LDFLAGS_NOLTO) $(LDFLAGS_NODIST)
NO_AS_NEEDED=	-Wl,--no-as-needed
CCSHARED=	-fPIC
# LINKFORSHARED are the flags passed to the $(CC) command that links
# the python executable -- this is only needed for a few systems
LINKFORSHARED=	-Xlinker -export-dynamic
ARFLAGS=	rcs
# Extra C flags added for building the interpreter object files.
CFLAGSFORSHARED=
# C flags used for building the interpreter object files
PY_STDMODULE_CFLAGS= $(PY_CFLAGS) $(PY_CFLAGS_NODIST) $(PY_CPPFLAGS) $(CFLAGSFORSHARED)
PY_BUILTIN_MODULE_CFLAGS= $(PY_STDMODULE_CFLAGS) -DPy_BUILD_CORE_BUILTIN
PY_CORE_CFLAGS=	$(PY_STDMODULE_CFLAGS) -DPy_BUILD_CORE
# Linker flags used for building the interpreter object files
PY_CORE_LDFLAGS=$(PY_LDFLAGS) $(PY_LDFLAGS_NODIST)
# Strict or non-strict aliasing flags used to compile dtoa.c, see above
CFLAGS_ALIASING=


# Machine-dependent subdirectories
MACHDEP=	linux

# Multiarch directory (may be empty)
MULTIARCH=	x86_64-linux-gnu
MULTIARCH_CPPFLAGS = -DMULTIARCH=\"x86_64-linux-gnu\"

# Install prefix for architecture-independent files
prefix=		/usr/local/python3.13-debug

# Install prefix for architecture-dependent files
exec_prefix=	${prefix}

# Install prefix for data files
datarootdir=    ${prefix}/share

# Expanded directories
BINDIR=		${exec_prefix}/bin
LIBDIR=		${exec_prefix}/lib
MANDIR=		${datarootdir}/man
INCLUDEDIR=	${prefix}/include
CONFINCLUDEDIR=	$(exec_prefix)/include
PLATLIBDIR=	lib
SCRIPTDIR=	$(prefix)/$(PLATLIBDIR)
# executable name for shebangs
EXENAME=	$(BINDIR)/python$(LDVERSION)$(EXE)
# Variable used by ensurepip
WHEEL_PKG_DIR=	

# Detailed destination directories
BINLIBDEST=	$(LIBDIR)/python$(VERSION)$(ABI_THREAD)
LIBDEST=	$(SCRIPTDIR)/python$(VERSION)$(ABI_THREAD)
INCLUDEPY=	$(INCLUDEDIR)/python$(LDVERSION)
CONFINCLUDEPY=	$(CONFINCLUDEDIR)/python$(LDVERSION)

# Symbols used for using shared libraries
SHLIB_SUFFIX=	.so
EXT_SUFFIX=	.cpython-313d-x86_64-linux-gnu.so
LDSHARED=	$(CC) -shared $(PY_LDFLAGS)
BLDSHARED=	$(CC) -shared $(PY_CORE_LDFLAGS)
LDCXXSHARED=	$(CXX) -shared $(PY_LDFLAGS)
DESTSHARED=	$(BINLIBDEST)/lib-dynload

# List of exported symbols for AIX
EXPORTSYMS=	
EXPORTSFROM=	

# Executable suffix (.exe on Windows and Mac OS X)
EXE=		
BUILDEXE=	

# Name of the patch file to apply for app store compliance
APP_STORE_COMPLIANCE_PATCH=

# Short name and location for Mac OS X Python framework
UNIVERSALSDK=
PYTHONFRAMEWORK=	
PYTHONFRAMEWORKDIR=	no-framework
PYTHONFRAMEWORKPREFIX=	
PYTHONFRAMEWORKINSTALLDIR= 
PYTHONFRAMEWORKINSTALLNAMEPREFIX= 
RESSRCDIR= 
# macOS deployment target selected during configure, to be checked
# by distutils. The export statement is needed to ensure that the
# deployment target is active during build.
MACOSX_DEPLOYMENT_TARGET=
#export MACOSX_DEPLOYMENT_TARGET

# iOS Deployment target selected during configure. Unlike macOS, the iOS
# deployment target is controlled using `-mios-version-min` arguments added to
# CFLAGS and LDFLAGS by the configure script. This variable is not used during
# the build, and is only listed here so it will be included in sysconfigdata.
IPHONEOS_DEPLOYMENT_TARGET=

# Option to install to strip binaries
STRIPFLAG=-s

# Flags to lipo to produce a 32-bit-only universal executable
LIPO_32BIT_FLAGS=

# Flags to lipo to produce an intel-64-only universal executable
LIPO_INTEL64_FLAGS=

# Environment to run shared python without installed libraries
RUNSHARED=       

# ensurepip options
ENSUREPIP=      upgrade

# Internal static libraries
LIBMPDEC_A= Modules/_decimal/libmpdec/libmpdec.a
LIBEXPAT_A= Modules/expat/libexpat.a
LIBHACL_SHA2_A= Modules/_hacl/libHacl_Hash_SHA2.a

# Module state, compiler flags and linker flags
# Empty CFLAGS and LDFLAGS are omitted.
# states:
#   * yes: module is available
#   * missing: build dependency is missing
#   * disabled: module is disabled
#   * n/a: module is not available on the current platform
# MODULE_EGG_STATE=yes  # yes, missing, disabled, n/a
# MODULE_EGG_CFLAGS=
# MODULE_EGG_LDFLAGS=
MODULE__IO_STATE=yes
MODULE__IO_CFLAGS=-I$(srcdir)/Modules/_io
MODULE_TIME_STATE=yes
MODULE_TIME_LDFLAGS=
MODULE_ARRAY_STATE=yes
MODULE__ASYNCIO_STATE=yes
MODULE__BISECT_STATE=yes
MODULE__CONTEXTVARS_STATE=yes
MODULE__CSV_STATE=yes
MODULE__HEAPQ_STATE=yes
MODULE__JSON_STATE=yes
MODULE__LSPROF_STATE=yes
MODULE__OPCODE_STATE=yes
MODULE__PICKLE_STATE=yes
MODULE__POSIXSUBPROCESS_STATE=yes
MODULE__QUEUE_STATE=yes
MODULE__RANDOM_STATE=yes
MODULE_SELECT_STATE=yes
MODULE__STRUCT_STATE=yes
MODULE__TYPING_STATE=yes
MODULE__INTERPRETERS_STATE=yes
MODULE__INTERPCHANNELS_STATE=yes
MODULE__INTERPQUEUES_STATE=yes
MODULE__ZONEINFO_STATE=yes
MODULE__MULTIPROCESSING_STATE=yes
MODULE__MULTIPROCESSING_CFLAGS=-I$(srcdir)/Modules/_multiprocessing
MODULE__POSIXSHMEM_STATE=yes
MODULE__POSIXSHMEM_CFLAGS=-I$(srcdir)/Modules/_multiprocessing
MODULE__POSIXSHMEM_LDFLAGS=
MODULE__STATISTICS_STATE=yes
MODULE__STATISTICS_LDFLAGS=-lm
MODULE_CMATH_STATE=yes
MODULE_CMATH_LDFLAGS=-lm
MODULE_MATH_STATE=yes
MODULE_MATH_LDFLAGS=-lm
MODULE__DATETIME_STATE=yes
MODULE__DATETIME_LDFLAGS= -lm
MODULE_FCNTL_STATE=yes
MODULE_FCNTL_LDFLAGS=
MODULE_MMAP_STATE=yes
MODULE__SOCKET_STATE=yes
MODULE_GRP_STATE=yes
MODULE_PWD_STATE=yes
MODULE_RESOURCE_STATE=yes
MODULE__SCPROXY_STATE=n/a
MODULE_SYSLOG_STATE=yes
MODULE_TERMIOS_STATE=yes
MODULE_PYEXPAT_STATE=yes
MODULE_PYEXPAT_CFLAGS=-I$(srcdir)/Modules/expat
MODULE_PYEXPAT_LDFLAGS=-lm $(LIBEXPAT_A)
MODULE__ELEMENTTREE_STATE=yes
MODULE__ELEMENTTREE_CFLAGS=-I$(srcdir)/Modules/expat
MODULE__CODECS_CN_STATE=yes
MODULE__CODECS_HK_STATE=yes
MODULE__CODECS_ISO2022_STATE=yes
MODULE__CODECS_JP_STATE=yes
MODULE__CODECS_KR_STATE=yes
MODULE__CODECS_TW_STATE=yes
MODULE__MULTIBYTECODEC_STATE=yes
MODULE_UNICODEDATA_STATE=yes
MODULE__MD5_STATE=yes
MODULE__MD5_CFLAGS=-I$(srcdir)/Modules/_hacl/include -I$(srcdir)/Modules/_hacl/internal -D_BSD_SOURCE -D_DEFAULT_SOURCE
MODULE__SHA1_STATE=yes
MODULE__SHA1_CFLAGS=-I$(srcdir)/Modules/_hacl/include -I$(srcdir)/Modules/_hacl/internal -D_BSD_SOURCE -D_DEFAULT_SOURCE
MODULE__SHA2_STATE=yes
MODULE__SHA2_CFLAGS=-I$(srcdir)/Modules/_hacl/include -I$(srcdir)/Modules/_hacl/internal -D_BSD_SOURCE -D_DEFAULT_SOURCE
MODULE__SHA3_STATE=yes
MODULE__BLAKE2_STATE=yes
MODULE__BLAKE2_CFLAGS=
MODULE__BLAKE2_LDFLAGS=
MODULE__CTYPES_STATE=yes
MODULE__CTYPES_CFLAGS=-fno-strict-overflow 
MODULE__CTYPES_LDFLAGS=-lffi  -ldl
MODULE__CURSES_STATE=yes
MODULE__CURSES_CFLAGS=-D_DEFAULT_SOURCE 
MODULE__CURSES_LDFLAGS=-lncursesw -ltinfo 

MODULE__CURSES_PANEL_STATE=yes
MODULE__CURSES_PANEL_CFLAGS=-D_DEFAULT_SOURCE  -D_DEFAULT_SOURCE 
MODULE__CURSES_PANEL_LDFLAGS=-lpanelw  -lncursesw -ltinfo 

MODULE__DECIMAL_STATE=yes
MODULE__DECIMAL_CFLAGS=-I$(srcdir)/Modules/_decimal/libmpdec -DTEST_COVERAGE -DCONFIG_64=1 -DANSI=1 -DHAVE_UINT128_T=1
MODULE__DECIMAL_LDFLAGS=-lm $(LIBMPDEC_A)
MODULE__DBM_STATE=yes
MODULE__DBM_CFLAGS=-DUSE_GDBM_COMPAT
MODULE__DBM_LDFLAGS=-lgdbm_compat
MODULE__GDBM_STATE=yes
MODULE__GDBM_CFLAGS=
MODULE__GDBM_LDFLAGS=-lgdbm
MODULE_READLINE_STATE=yes
MODULE_READLINE_CFLAGS=-D_DEFAULT_SOURCE 
MODULE_READLINE_LDFLAGS=-lreadline 
MODULE__SQLITE3_STATE=yes
MODULE__SQLITE3_CFLAGS= -I$(srcdir)/Modules/_sqlite
MODULE__SQLITE3_LDFLAGS=-lsqlite3 
MODULE__TKINTER_STATE=yes
MODULE__TKINTER_CFLAGS=-I/usr/include/tcl8.6  -Wno-strict-prototypes -DWITH_APPINIT=1
MODULE__TKINTER_LDFLAGS=-ltk8.6 -ltkstub8.6 -ltcl8.6 -ltclstub8.6 
MODULE__UUID_STATE=yes
MODULE__UUID_CFLAGS=-I/usr/include/uuid 
MODULE__UUID_LDFLAGS=-luuid 
MODULE_ZLIB_STATE=yes
MODULE_ZLIB_CFLAGS=
MODULE_ZLIB_LDFLAGS=-lz 
MODULE_BINASCII_STATE=yes
MODULE_BINASCII_CFLAGS=-DUSE_ZLIB_CRC32 
MODULE_BINASCII_LDFLAGS=-lz 
MODULE__BZ2_STATE=yes
MODULE__BZ2_CFLAGS=
MODULE__BZ2_LDFLAGS=-lbz2
MODULE__LZMA_STATE=yes
MODULE__LZMA_CFLAGS=
MODULE__LZMA_LDFLAGS=-llzma 
MODULE__SSL_STATE=missing
MODULE__HASHLIB_STATE=missing
MODULE__TESTCAPI_STATE=yes
MODULE__TESTCAPI_LDFLAGS=
MODULE__TESTCLINIC_STATE=yes
MODULE__TESTCLINIC_LIMITED_STATE=yes
MODULE__TESTLIMITEDCAPI_STATE=yes
MODULE__TESTINTERNALCAPI_STATE=yes
MODULE__TESTBUFFER_STATE=yes
MODULE__TESTIMPORTMULTIPLE_STATE=yes
MODULE__TESTMULTIPHASE_STATE=yes
MODULE__TESTSINGLEPHASE_STATE=yes
MODULE__TESTEXTERNALINSPECTION_STATE=yes
MODULE_XXSUBTYPE_STATE=yes
MODULE__XXTESTFUZZ_STATE=yes
MODULE__CTYPES_TEST_STATE=yes
MODULE__CTYPES_TEST_LDFLAGS=-lm
MODULE_XXLIMITED_STATE=yes
MODULE_XXLIMITED_35_STATE=yes


# Default zoneinfo.TZPATH. Added here to expose it in sysconfig.get_config_var
TZPATH=/usr/share/zoneinfo:/usr/lib/zoneinfo:/usr/share/lib/zoneinfo:/etc/zoneinfo

# If to install mimalloc headers
INSTALL_MIMALLOC=yes

# Modes for directories, executables and data files created by the
# install process.  Default to user-only-writable for all file types.
DIRMODE=	755
EXEMODE=	755
FILEMODE=	644

# configure script arguments
CONFIG_ARGS=	 '--enable-optimizations' '--enable-experimental-jit' '--with-pydebug' '--prefix=/usr/local/python3.13-debug'


# Subdirectories with code
SRCDIRS= 	  Modules   Modules/_blake2   Modules/_ctypes   Modules/_decimal   Modules/_decimal/libmpdec   Modules/_hacl   Modules/_io   Modules/_multiprocessing   Modules/_sqlite   Modules/_sre   Modules/_testcapi   Modules/_testinternalcapi   Modules/_testlimitedcapi   Modules/_xxtestfuzz   Modules/cjkcodecs   Modules/expat   Objects   Objects/mimalloc   Objects/mimalloc/prim   Parser   Parser/tokenizer   Parser/lexer   Programs   Python   Python/frozen_modules

# Other subdirectories
SUBDIRSTOO=	Include Lib Misc

# assets for Emscripten browser builds
WASM_ASSETS_DIR=.$(prefix)
WASM_STDLIB=$(WASM_ASSETS_DIR)/lib/python$(VERSION)/os.py

# Files and directories to be distributed
CONFIGFILES=	configure configure.ac acconfig.h pyconfig.h.in Makefile.pre.in
DISTFILES=	README.rst ChangeLog $(CONFIGFILES)
DISTDIRS=	$(SUBDIRS) $(SUBDIRSTOO) Ext-dummy
DIST=		$(DISTFILES) $(DISTDIRS)


LIBRARY=	libpython$(VERSION)$(ABIFLAGS).a
LDLIBRARY=      libpython$(VERSION)$(ABIFLAGS).a
BLDLIBRARY=     $(LDLIBRARY)
PY3LIBRARY=     
DLLLIBRARY=	
LDLIBRARYDIR=   
INSTSONAME=	$(LDLIBRARY)
LIBRARY_DEPS=	$(LIBRARY) $(PY3LIBRARY) $(EXPORTSYMS)
LINK_PYTHON_DEPS=$(LIBRARY_DEPS)
PY_ENABLE_SHARED=	0
STATIC_LIBPYTHON=	1


LIBS=		-ldl 
LIBM=		-lm
LIBC=		
SYSLIBS=	$(LIBM) $(LIBC)
SHLIBS=		$(LIBS)

DLINCLDIR=	.
DYNLOADFILE=	dynload_shlib.o
MACHDEP_OBJS=	
LIBOBJDIR=	Python/
LIBOBJS=	

PYTHON=		python$(EXE)
BUILDPYTHON=	python$(BUILDEXE)

HOSTRUNNER= 

PYTHON_FOR_REGEN?=python3.13
UPDATE_FILE=$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/update_file.py
PYTHON_FOR_BUILD=./$(BUILDPYTHON) -E
# Single-platform builds depend on $(BUILDPYTHON). Cross builds use an
# external "build Python" and have an empty PYTHON_FOR_BUILD_DEPS.
PYTHON_FOR_BUILD_DEPS=$(BUILDPYTHON)

# Single-platform builds use Programs/_freeze_module.c for bootstrapping and
# ./_bootstrap_python Programs/_freeze_module.py for remaining modules
# Cross builds use an external "build Python" for all modules.
PYTHON_FOR_FREEZE=./_bootstrap_python
FREEZE_MODULE_BOOTSTRAP=./Programs/_freeze_module
FREEZE_MODULE_BOOTSTRAP_DEPS=Programs/_freeze_module
FREEZE_MODULE=$(PYTHON_FOR_FREEZE) $(srcdir)/Programs/_freeze_module.py
FREEZE_MODULE_DEPS=_bootstrap_python $(srcdir)/Programs/_freeze_module.py

_PYTHON_HOST_PLATFORM=
BUILD_GNU_TYPE=	x86_64-pc-linux-gnu
HOST_GNU_TYPE=	x86_64-pc-linux-gnu

# The task to run while instrumented when building the profile-opt target.
# To speed up profile generation, we don't run the full unit test suite
# by default. The default is "-m test --pgo". To run more tests, use
# PROFILE_TASK="-m test --pgo-extended"
PROFILE_TASK=	-m test --pgo --timeout=$(TESTTIMEOUT)

# report files for gcov / lcov coverage report
COVERAGE_INFO=	$(abs_builddir)/coverage.info
COVERAGE_REPORT=$(abs_builddir)/lcov-report
COVERAGE_LCOV_OPTIONS=--rc lcov_branch_coverage=1
COVERAGE_REPORT_OPTIONS=--rc lcov_branch_coverage=1 --branch-coverage --title "CPython $(VERSION) LCOV report [commit $(shell $(GITVERSION))]"


# === Definitions added by makesetup ===


LOCALMODLIBS= $(MODULE_ATEXIT_LDFLAGS) $(MODULE_FAULTHANDLER_LDFLAGS) $(MODULE_POSIX_LDFLAGS) $(MODULE__SIGNAL_LDFLAGS) $(MODULE__TRACEMALLOC_LDFLAGS) $(MODULE__SUGGESTIONS_LDFLAGS) $(MODULE__CODECS_LDFLAGS) $(MODULE__COLLECTIONS_LDFLAGS) $(MODULE_ERRNO_LDFLAGS) $(MODULE__IO_LDFLAGS) $(MODULE_ITERTOOLS_LDFLAGS) $(MODULE__SRE_LDFLAGS) $(MODULE__SYSCONFIG_LDFLAGS) $(MODULE__THREAD_LDFLAGS) $(MODULE_TIME_LDFLAGS) $(MODULE__TYPING_LDFLAGS) $(MODULE__WEAKREF_LDFLAGS) $(MODULE__ABC_LDFLAGS) $(MODULE__FUNCTOOLS_LDFLAGS) $(MODULE__LOCALE_LDFLAGS) $(MODULE__OPERATOR_LDFLAGS) $(MODULE__STAT_LDFLAGS) $(MODULE__SYMTABLE_LDFLAGS) $(MODULE_PWD_LDFLAGS)
BASEMODLIBS=
SHAREDMODS= Modules/array$(EXT_SUFFIX) Modules/_asyncio$(EXT_SUFFIX) Modules/_bisect$(EXT_SUFFIX) Modules/_contextvars$(EXT_SUFFIX) Modules/_csv$(EXT_SUFFIX) Modules/_heapq$(EXT_SUFFIX) Modules/_json$(EXT_SUFFIX) Modules/_lsprof$(EXT_SUFFIX) Modules/_opcode$(EXT_SUFFIX) Modules/_pickle$(EXT_SUFFIX) Modules/_queue$(EXT_SUFFIX) Modules/_random$(EXT_SUFFIX) Modules/_struct$(EXT_SUFFIX) Modules/_interpreters$(EXT_SUFFIX) Modules/_interpchannels$(EXT_SUFFIX) Modules/_interpqueues$(EXT_SUFFIX) Modules/_zoneinfo$(EXT_SUFFIX) Modules/math$(EXT_SUFFIX) Modules/cmath$(EXT_SUFFIX) Modules/_statistics$(EXT_SUFFIX) Modules/_datetime$(EXT_SUFFIX) Modules/_decimal$(EXT_SUFFIX) Modules/binascii$(EXT_SUFFIX) Modules/_bz2$(EXT_SUFFIX) Modules/_lzma$(EXT_SUFFIX) Modules/zlib$(EXT_SUFFIX) Modules/_dbm$(EXT_SUFFIX) Modules/_gdbm$(EXT_SUFFIX) Modules/readline$(EXT_SUFFIX) Modules/_md5$(EXT_SUFFIX) Modules/_sha1$(EXT_SUFFIX) Modules/_sha2$(EXT_SUFFIX) Modules/_sha3$(EXT_SUFFIX) Modules/_blake2$(EXT_SUFFIX) Modules/pyexpat$(EXT_SUFFIX) Modules/_elementtree$(EXT_SUFFIX) Modules/_codecs_cn$(EXT_SUFFIX) Modules/_codecs_hk$(EXT_SUFFIX) Modules/_codecs_iso2022$(EXT_SUFFIX) Modules/_codecs_jp$(EXT_SUFFIX) Modules/_codecs_kr$(EXT_SUFFIX) Modules/_codecs_tw$(EXT_SUFFIX) Modules/_multibytecodec$(EXT_SUFFIX) Modules/unicodedata$(EXT_SUFFIX) Modules/fcntl$(EXT_SUFFIX) Modules/grp$(EXT_SUFFIX) Modules/mmap$(EXT_SUFFIX) Modules/_posixsubprocess$(EXT_SUFFIX) Modules/resource$(EXT_SUFFIX) Modules/select$(EXT_SUFFIX) Modules/_socket$(EXT_SUFFIX) Modules/syslog$(EXT_SUFFIX) Modules/termios$(EXT_SUFFIX) Modules/_posixshmem$(EXT_SUFFIX) Modules/_multiprocessing$(EXT_SUFFIX) Modules/_ctypes$(EXT_SUFFIX) Modules/_curses$(EXT_SUFFIX) Modules/_curses_panel$(EXT_SUFFIX) Modules/_sqlite3$(EXT_SUFFIX) Modules/_uuid$(EXT_SUFFIX) Modules/_tkinter$(EXT_SUFFIX) Modules/xxsubtype$(EXT_SUFFIX) Modules/_xxtestfuzz$(EXT_SUFFIX) Modules/_testbuffer$(EXT_SUFFIX) Modules/_testinternalcapi$(EXT_SUFFIX) Modules/_testcapi$(EXT_SUFFIX) Modules/_testlimitedcapi$(EXT_SUFFIX) Modules/_testclinic$(EXT_SUFFIX) Modules/_testclinic_limited$(EXT_SUFFIX) Modules/_testimportmultiple$(EXT_SUFFIX) Modules/_testmultiphase$(EXT_SUFFIX) Modules/_testsinglephase$(EXT_SUFFIX) Modules/_testexternalinspection$(EXT_SUFFIX) Modules/_ctypes_test$(EXT_SUFFIX) Modules/xxlimited$(EXT_SUFFIX) Modules/xxlimited_35$(EXT_SUFFIX)
PYTHONPATH=$(COREPYTHONPATH)
COREPYTHONPATH=$(DESTPATH)$(SITEPATH)$(TESTPATH)
TESTPATH=
SITEPATH=
DESTPATH=
MACHDESTLIB=$(BINLIBDEST)
DESTLIB=$(LIBDEST)



##########################################################################
# Modules
MODULE_OBJS=	\
		Modules/config.o \
		Modules/main.o \
		Modules/gcmodule.o

IO_H=		Modules/_io/_iomodule.h

IO_OBJS=	\
		Modules/_io/_iomodule.o \
		Modules/_io/iobase.o \
		Modules/_io/fileio.o \
		Modules/_io/bufferedio.o \
		Modules/_io/textio.o \
		Modules/_io/bytesio.o \
		Modules/_io/stringio.o


##########################################################################
# mimalloc

MIMALLOC_HEADERS= \
	$(srcdir)/Include/internal/pycore_mimalloc.h \
	$(srcdir)/Include/internal/mimalloc/mimalloc.h \
	$(srcdir)/Include/internal/mimalloc/mimalloc/atomic.h \
	$(srcdir)/Include/internal/mimalloc/mimalloc/internal.h \
	$(srcdir)/Include/internal/mimalloc/mimalloc/prim.h \
	$(srcdir)/Include/internal/mimalloc/mimalloc/track.h \
	$(srcdir)/Include/internal/mimalloc/mimalloc/types.h


##########################################################################
# Parser

PEGEN_OBJS=		\
		Parser/pegen.o \
		Parser/pegen_errors.o \
		Parser/action_helpers.o \
		Parser/parser.o \
		Parser/string_parser.o \
		Parser/peg_api.o

TOKENIZER_OBJS=		\
		Parser/lexer/buffer.o \
		Parser/lexer/lexer.o \
		Parser/lexer/state.o \
		Parser/tokenizer/file_tokenizer.o \
		Parser/tokenizer/readline_tokenizer.o \
		Parser/tokenizer/string_tokenizer.o \
		Parser/tokenizer/utf8_tokenizer.o \
		Parser/tokenizer/helpers.o

PEGEN_HEADERS= \
		$(srcdir)/Include/internal/pycore_parser.h \
		$(srcdir)/Parser/pegen.h \
		$(srcdir)/Parser/string_parser.h

TOKENIZER_HEADERS= \
		Parser/lexer/buffer.h \
		Parser/lexer/lexer.h \
		Parser/lexer/state.h \
		Parser/tokenizer/tokenizer.h \
		Parser/tokenizer/helpers.h

POBJS=		\
		Parser/token.o \

PARSER_OBJS=	$(POBJS) $(PEGEN_OBJS) $(TOKENIZER_OBJS) Parser/myreadline.o

PARSER_HEADERS= \
		$(PEGEN_HEADERS) \
		$(TOKENIZER_HEADERS)

##########################################################################
# Python

PYTHON_OBJS=	\
		Python/_warnings.o \
		Python/Python-ast.o \
		Python/Python-tokenize.o \
		Python/asdl.o \
		Python/assemble.o \
		Python/ast.o \
		Python/ast_opt.o \
		Python/ast_unparse.o \
		Python/bltinmodule.o \
		Python/brc.o \
		Python/ceval.o \
		Python/codecs.o \
		Python/compile.o \
		Python/context.o \
		Python/critical_section.o \
		Python/crossinterp.o \
		Python/dynamic_annotations.o \
		Python/errors.o \
		Python/flowgraph.o \
		Python/frame.o \
		Python/frozenmain.o \
		Python/future.o \
		Python/gc.o \
		Python/gc_free_threading.o \
		Python/gc_gil.o \
		Python/getargs.o \
		Python/getcompiler.o \
		Python/getcopyright.o \
		Python/getplatform.o \
		Python/getversion.o \
		Python/ceval_gil.o \
		Python/hamt.o \
		Python/hashtable.o \
		Python/import.o \
		Python/importdl.o \
		Python/initconfig.o \
		Python/interpconfig.o \
		Python/instrumentation.o \
		Python/instruction_sequence.o \
		Python/intrinsics.o \
		Python/jit.o \
		Python/legacy_tracing.o \
		Python/lock.o \
		Python/marshal.o \
		Python/modsupport.o \
		Python/mysnprintf.o \
		Python/mystrtoul.o \
		Python/object_stack.o \
		Python/optimizer.o \
		Python/optimizer_analysis.o \
		Python/optimizer_symbols.o \
		Python/parking_lot.o \
		Python/pathconfig.o \
		Python/preconfig.o \
		Python/pyarena.o \
		Python/pyctype.o \
		Python/pyfpe.o \
		Python/pyhash.o \
		Python/pylifecycle.o \
		Python/pymath.o \
		Python/pystate.o \
		Python/pythonrun.o \
		Python/pytime.o \
		Python/qsbr.o \
		Python/bootstrap_hash.o \
		Python/specialize.o \
		Python/structmember.o \
		Python/symtable.o \
		Python/sysmodule.o \
		Python/thread.o \
		Python/traceback.o \
		Python/tracemalloc.o \
		Python/getopt.o \
		Python/pystrcmp.o \
		Python/pystrtod.o \
		Python/pystrhex.o \
		Python/dtoa.o \
		Python/formatter_unicode.o \
		Python/fileutils.o \
		Python/suggestions.o \
		Python/perf_trampoline.o \
		Python/perf_jit_trampoline.o \
		Python/$(DYNLOADFILE) \
		$(LIBOBJS) \
		$(MACHDEP_OBJS) \
		$(DTRACE_OBJS) \
		


##########################################################################
# Objects
OBJECT_OBJS=	\
		Objects/abstract.o \
		Objects/boolobject.o \
		Objects/bytes_methods.o \
		Objects/bytearrayobject.o \
		Objects/bytesobject.o \
		Objects/call.o \
		Objects/capsule.o \
		Objects/cellobject.o \
		Objects/classobject.o \
		Objects/codeobject.o \
		Objects/complexobject.o \
		Objects/descrobject.o \
		Objects/enumobject.o \
		Objects/exceptions.o \
		Objects/genericaliasobject.o \
		Objects/genobject.o \
		Objects/fileobject.o \
		Objects/floatobject.o \
		Objects/frameobject.o \
		Objects/funcobject.o \
		Objects/iterobject.o \
		Objects/listobject.o \
		Objects/longobject.o \
		Objects/dictobject.o \
		Objects/odictobject.o \
		Objects/memoryobject.o \
		Objects/methodobject.o \
		Objects/moduleobject.o \
		Objects/namespaceobject.o \
		Objects/object.o \
		Objects/obmalloc.o \
		Objects/picklebufobject.o \
		Objects/rangeobject.o \
		Objects/setobject.o \
		Objects/sliceobject.o \
		Objects/structseq.o \
		Objects/tupleobject.o \
		Objects/typeobject.o \
		Objects/typevarobject.o \
		Objects/unicodeobject.o \
		Objects/unicodectype.o \
		Objects/unionobject.o \
		Objects/weakrefobject.o \
		Python/asm_trampoline.o

##########################################################################
# objects that get linked into the Python library
LIBRARY_OBJS_OMIT_FROZEN=	\
		Modules/getbuildinfo.o \
		$(PARSER_OBJS) \
		$(OBJECT_OBJS) \
		$(PYTHON_OBJS) \
		$(MODULE_OBJS) \
		$(MODOBJS)

LIBRARY_OBJS=	\
		$(LIBRARY_OBJS_OMIT_FROZEN) \
		Modules/getpath.o \
		Python/frozen.o

LINK_PYTHON_OBJS=$(LIBRARY_OBJS)

##########################################################################
# DTrace

# On some systems, object files that reference DTrace probes need to be modified
# in-place by dtrace(1).
DTRACE_DEPS = \
	Python/ceval.o Python/gc.o Python/import.o Python/sysmodule.o

##########################################################################
# decimal's libmpdec

LIBMPDEC_OBJS= \
		Modules/_decimal/libmpdec/basearith.o \
		Modules/_decimal/libmpdec/constants.o \
		Modules/_decimal/libmpdec/context.o \
		Modules/_decimal/libmpdec/convolute.o \
		Modules/_decimal/libmpdec/crt.o \
		Modules/_decimal/libmpdec/difradix2.o \
		Modules/_decimal/libmpdec/fnt.o \
		Modules/_decimal/libmpdec/fourstep.o \
		Modules/_decimal/libmpdec/io.o \
		Modules/_decimal/libmpdec/mpalloc.o \
		Modules/_decimal/libmpdec/mpdecimal.o \
		Modules/_decimal/libmpdec/numbertheory.o \
		Modules/_decimal/libmpdec/sixstep.o \
		Modules/_decimal/libmpdec/transpose.o
		# _decimal does not use signaling API
		# Modules/_decimal/libmpdec/mpsignal.o

LIBMPDEC_HEADERS= \
		$(srcdir)/Modules/_decimal/libmpdec/basearith.h \
		$(srcdir)/Modules/_decimal/libmpdec/bits.h \
		$(srcdir)/Modules/_decimal/libmpdec/constants.h \
		$(srcdir)/Modules/_decimal/libmpdec/convolute.h \
		$(srcdir)/Modules/_decimal/libmpdec/crt.h \
		$(srcdir)/Modules/_decimal/libmpdec/difradix2.h \
		$(srcdir)/Modules/_decimal/libmpdec/fnt.h \
		$(srcdir)/Modules/_decimal/libmpdec/fourstep.h \
		$(srcdir)/Modules/_decimal/libmpdec/io.h \
		$(srcdir)/Modules/_decimal/libmpdec/mpalloc.h \
		$(srcdir)/Modules/_decimal/libmpdec/mpdecimal.h \
		$(srcdir)/Modules/_decimal/libmpdec/numbertheory.h \
		$(srcdir)/Modules/_decimal/libmpdec/sixstep.h \
		$(srcdir)/Modules/_decimal/libmpdec/transpose.h \
		$(srcdir)/Modules/_decimal/libmpdec/typearith.h \
		$(srcdir)/Modules/_decimal/libmpdec/umodarith.h

##########################################################################
# pyexpat's expat library

LIBEXPAT_OBJS= \
		Modules/expat/xmlparse.o \
		Modules/expat/xmlrole.o \
		Modules/expat/xmltok.o

LIBEXPAT_HEADERS= \
		Modules/expat/ascii.h \
		Modules/expat/asciitab.h \
		Modules/expat/expat.h \
		Modules/expat/expat_config.h \
		Modules/expat/expat_external.h \
		Modules/expat/iasciitab.h \
		Modules/expat/internal.h \
		Modules/expat/latin1tab.h \
		Modules/expat/nametab.h \
		Modules/expat/pyexpatns.h \
		Modules/expat/siphash.h \
		Modules/expat/utf8tab.h \
		Modules/expat/xmlrole.h \
		Modules/expat/xmltok.h \
		Modules/expat/xmltok_impl.h \
		Modules/expat/xmltok_impl.c \
		Modules/expat/xmltok_ns.c

##########################################################################
# hashlib's HACL* library

LIBHACL_SHA2_OBJS= \
                Modules/_hacl/Hacl_Hash_SHA2.o

LIBHACL_HEADERS= \
                Modules/_hacl/include/krml/FStar_UInt128_Verified.h \
                Modules/_hacl/include/krml/FStar_UInt_8_16_32_64.h \
                Modules/_hacl/include/krml/fstar_uint128_struct_endianness.h \
                Modules/_hacl/include/krml/internal/target.h \
                Modules/_hacl/include/krml/lowstar_endianness.h \
                Modules/_hacl/include/krml/types.h \
		Modules/_hacl/Hacl_Streaming_Types.h \
                Modules/_hacl/python_hacl_namespaces.h

LIBHACL_SHA2_HEADERS= \
                Modules/_hacl/Hacl_Hash_SHA2.h \
                Modules/_hacl/internal/Hacl_Hash_SHA2.h \
		$(LIBHACL_HEADERS)

#########################################################################
# Rules

# Default target
all:		profile-opt

# First target in Makefile is implicit default. So .PHONY needs to come after
# all.
.PHONY: all

# Provide quick help for common Makefile targets.
.PHONY: help
help:
	@echo "Run 'make' to build the Python executable and extension modules"
	@echo ""
	@echo "or 'make <target>' where <target> is one of:"
	@echo "  test         run the test suite"
	@echo "  install      install built files"
	@echo "  regen-all    regenerate a number of generated source files"
	@echo "  clinic       run Argument Clinic over source files"
	@echo ""
	@echo "  clean        to remove build files"
	@echo "  distclean    'clean' + remove other generated files (patch, exe, etc)"
	@echo ""
	@echo "  recheck      rerun configure with last cmdline options"
	@echo "  reindent     reindent .py files in Lib directory"
	@echo "  tags         build a tags file (useful for Emacs and other editors)"
	@echo "  list-targets list all targets in the Makefile"

# Display a full list of Makefile targets
.PHONY: list-targets
list-targets:
	@grep -E '^[A-Za-z][-A-Za-z0-9]+:' Makefile | awk -F : '{print $$1}'

.PHONY: build_all
build_all:	check-clean-src check-app-store-compliance $(BUILDPYTHON) platform sharedmods \
		gdbhooks Programs/_testembed scripts checksharedmods rundsymutil

.PHONY: build_wasm
build_wasm: check-clean-src $(BUILDPYTHON) platform sharedmods \
		python-config checksharedmods

# Check that the source is clean when building out of source.
.PHONY: check-clean-src
check-clean-src:
	@if test -n "$(VPATH)" -a \( \
	    -f "$(srcdir)/$(BUILDPYTHON)" \
	    -o -f "$(srcdir)/Programs/python.o" \
	    -o -f "$(srcdir)/Python/frozen_modules/importlib._bootstrap.h" \
	\); then \
		echo "Error: The source directory ($(srcdir)) is not clean" ; \
		echo "Building Python out of the source tree (in $(abs_builddir)) requires a clean source tree ($(abs_srcdir))" ; \
		echo "Build artifacts such as .o files, executables, and Python/frozen_modules/*.h must not exist within $(srcdir)." ; \
		echo "Try to run:" ; \
		echo "  (cd \"$(srcdir)\" && make clean || git clean -fdx -e Doc/venv)" ; \
		exit 1; \
	fi

# Check that the app store compliance patch can be applied (if configured).
# This is checked as a dry-run against the original library sources;
# the patch will be actually applied during the install phase.
.PHONY: check-app-store-compliance
check-app-store-compliance:
	@if [ "$(APP_STORE_COMPLIANCE_PATCH)" != "" ]; then \
		patch --dry-run --quiet --force --strip 1 --directory "$(abs_srcdir)" --input "$(abs_srcdir)/$(APP_STORE_COMPLIANCE_PATCH)"; \
		echo "App store compliance patch can be applied."; \
	fi

# Profile generation build must start from a clean tree.
profile-clean-stamp:
	$(MAKE) clean
	touch $@

# Compile with profile generation enabled.
profile-gen-stamp: profile-clean-stamp
	@if [ $(LLVM_PROF_ERR) = yes ]; then \
		echo "Error: Cannot perform PGO build because llvm-profdata was not found in PATH" ;\
		echo "Please add it to PATH and run ./configure again" ;\
		exit 1;\
	fi
	@echo "Building with support for profile generation:"
	$(MAKE) build_all CFLAGS_NODIST="$(CFLAGS_NODIST) $(PGO_PROF_GEN_FLAG)" LDFLAGS_NODIST="$(LDFLAGS_NODIST) $(PGO_PROF_GEN_FLAG)" LIBS="$(LIBS)"
	touch $@

# Run task with profile generation build to create profile information.
profile-run-stamp:
	@echo "Running code to generate profile data (this can take a while):"
	# First, we need to create a clean build with profile generation
	# enabled.
	$(MAKE) profile-gen-stamp
	# Next, run the profile task to generate the profile information.
	@ # FIXME: can't run for a cross build
	$(LLVM_PROF_FILE) $(RUNSHARED) ./$(BUILDPYTHON) $(PROFILE_TASK)
	$(LLVM_PROF_MERGER)
	# Remove profile generation binary since we are done with it.
	$(MAKE) clean-retain-profile
	# This is an expensive target to build and it does not have proper
	# makefile dependency information.  So, we create a "stamp" file
	# to record its completion and avoid re-running it.
	touch $@

# Compile Python binary with profile guided optimization.
# To force re-running of the profile task, remove the profile-run-stamp file.
.PHONY: profile-opt
profile-opt: profile-run-stamp
	@echo "Rebuilding with profile guided optimizations:"
	-rm -f profile-clean-stamp
	$(MAKE) build_all CFLAGS_NODIST="$(CFLAGS_NODIST) $(PGO_PROF_USE_FLAG)" LDFLAGS_NODIST="$(LDFLAGS_NODIST)"

# List of binaries that BOLT runs on.
BOLT_BINARIES := $(BUILDPYTHON)

BOLT_INSTRUMENT_FLAGS := 
BOLT_APPLY_FLAGS :=  -update-debug-sections -reorder-blocks=ext-tsp -reorder-functions=hfsort+ -split-functions -icf=1 -inline-all -split-eh -reorder-functions-use-hot-size -peepholes=none -jump-tables=aggressive -inline-ap -indirect-call-promotion=all -dyno-stats -use-gnu-stack -frame-opt=hot 

.PHONY: clean-bolt
clean-bolt:
	# Profile data.
	rm -f *.fdata
	# Pristine binaries before BOLT optimization.
	rm -f *.prebolt
	# BOLT instrumented binaries.
	rm -f *.bolt_inst

profile-bolt-stamp: $(BUILDPYTHON)
	# Ensure a pristine, pre-BOLT copy of the binary and no profile data from last run.
	for bin in $(BOLT_BINARIES); do \
	  prebolt="$${bin}.prebolt"; \
	  if [ -e "$${prebolt}" ]; then \
	    echo "Restoring pre-BOLT binary $${prebolt}"; \
	    mv "$${bin}.prebolt" "$${bin}"; \
	  fi; \
	  cp "$${bin}" "$${prebolt}"; \
	  rm -f $${bin}.bolt.*.fdata $${bin}.fdata; \
	done
	# Instrument each binary.
	for bin in $(BOLT_BINARIES); do \
	   "$${bin}" -instrument -instrumentation-file-append-pid -instrumentation-file=$(abspath $${bin}.bolt) -o $${bin}.bolt_inst $(BOLT_INSTRUMENT_FLAGS); \
	  mv "$${bin}.bolt_inst" "$${bin}"; \
	done
	# Run instrumented binaries to collect data.
	$(RUNSHARED) ./$(BUILDPYTHON) $(PROFILE_TASK)
	# Merge all the data files together.
	for bin in $(BOLT_BINARIES); do \
	   $${bin}.*.fdata > "$${bin}.fdata"; \
	  rm -f $${bin}.*.fdata; \
	done
	# Run bolt against the merged data to produce an optimized binary.
	for bin in $(BOLT_BINARIES); do \
	   "$${bin}.prebolt" -o "$${bin}.bolt" -data="$${bin}.fdata" $(BOLT_APPLY_FLAGS); \
	  mv "$${bin}.bolt" "$${bin}"; \
	done
	touch $@

.PHONY: bolt-opt
bolt-opt:
	$(MAKE) 
	$(MAKE) profile-bolt-stamp

# Compile and run with gcov
.PHONY: coverage
coverage:
	@echo "Building with support for coverage checking:"
	$(MAKE) clean
	$(MAKE) build_all CFLAGS="$(CFLAGS) -O0 -pg --coverage" LDFLAGS="$(LDFLAGS) --coverage"

.PHONY: coverage-lcov
coverage-lcov:
	@echo "Creating Coverage HTML report with LCOV:"
	@rm -f $(COVERAGE_INFO)
	@rm -rf $(COVERAGE_REPORT)
	@lcov $(COVERAGE_LCOV_OPTIONS) --capture \
	    --directory $(abs_builddir) \
	    --base-directory $(realpath $(abs_builddir)) \
	    --path $(realpath $(abs_srcdir)) \
	    --output-file $(COVERAGE_INFO)
	@ # remove 3rd party modules, system headers and internal files with
	@ # debug, test or dummy functions.
	@lcov $(COVERAGE_LCOV_OPTIONS) --remove $(COVERAGE_INFO) \
	    '*/Modules/_blake2/impl/*' \
	    '*/Modules/_ctypes/libffi*/*' \
	    '*/Modules/_decimal/libmpdec/*' \
	    '*/Modules/expat/*' \
	    '*/Modules/xx*.c' \
	    '*/Python/pyfpe.c' \
	    '*/Python/pystrcmp.c' \
	    '/usr/include/*' \
	    '/usr/local/include/*' \
	    '/usr/lib/gcc/*' \
	    --output-file $(COVERAGE_INFO)
	@genhtml $(COVERAGE_INFO) \
	    --output-directory $(COVERAGE_REPORT) \
	    $(COVERAGE_REPORT_OPTIONS)
	@echo
	@echo "lcov report at $(COVERAGE_REPORT)/index.html"
	@echo

# Force regeneration of parser and frozen modules
.PHONY: coverage-report
coverage-report: regen-token regen-frozen
	@ # build with coverage info
	$(MAKE) coverage
	@ # run tests, ignore failures
	$(TESTRUNNER) --fast-ci --timeout=$(TESTTIMEOUT) $(TESTOPTS) || true
	@ # build lcov report
	$(MAKE) coverage-lcov

# Run "Argument Clinic" over all source files
.PHONY: clinic
clinic: check-clean-src $(srcdir)/Modules/_blake2/blake2s_impl.c
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/clinic/clinic.py --make --exclude Lib/test/clinic.test.c --srcdir $(srcdir)

.PHONY: clinic-tests
clinic-tests: check-clean-src $(srcdir)/Lib/test/clinic.test.c
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/clinic/clinic.py -f $(srcdir)/Lib/test/clinic.test.c

# Build the interpreter
$(BUILDPYTHON):	Programs/python.o $(LINK_PYTHON_DEPS)
	$(LINKCC) $(PY_CORE_LDFLAGS) $(LINKFORSHARED) -o $@ Programs/python.o $(LINK_PYTHON_OBJS) $(LIBS) $(MODLIBS) $(SYSLIBS)

platform: $(PYTHON_FOR_BUILD_DEPS) pybuilddir.txt
	$(RUNSHARED) $(PYTHON_FOR_BUILD) -c 'import sys ; from sysconfig import get_platform ; print("%s-%d.%d" % (get_platform(), *sys.version_info[:2]))' >platform

# Create build directory and generate the sysconfig build-time data there.
# pybuilddir.txt contains the name of the build dir and is used for
# sys.path fixup -- see Modules/getpath.c.
# Since this step runs before shared modules are built, try to avoid bootstrap
# problems by creating a dummy pybuilddir.txt just to allow interpreter
# initialization to succeed.  It will be overwritten by generate-posix-vars
# or removed in case of failure.
pybuilddir.txt: $(PYTHON_FOR_BUILD_DEPS)
	@echo "none" > ./pybuilddir.txt
	$(RUNSHARED) $(PYTHON_FOR_BUILD) -S -m sysconfig --generate-posix-vars ;\
	if test $$? -ne 0 ; then \
		echo "generate-posix-vars failed" ; \
		rm -f ./pybuilddir.txt ; \
		exit 1 ; \
	fi

# blake2s is auto-generated from blake2b
$(srcdir)/Modules/_blake2/blake2s_impl.c: $(srcdir)/Modules/_blake2/blake2b_impl.c $(srcdir)/Modules/_blake2/blake2b2s.py
	$(PYTHON_FOR_REGEN) $(srcdir)/Modules/_blake2/blake2b2s.py
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/clinic/clinic.py -f $@

# Build static library
$(LIBRARY): $(LIBRARY_OBJS)
	-rm -f $@
	$(AR) $(ARFLAGS) $@ $(LIBRARY_OBJS)

libpython$(LDVERSION).so: $(LIBRARY_OBJS) $(DTRACE_OBJS)
	$(BLDSHARED) -Wl,-h$(INSTSONAME) -o $(INSTSONAME) $(LIBRARY_OBJS) $(MODLIBS) $(SHLIBS) $(LIBC) $(LIBM)
	if test $(INSTSONAME) != $@; then \
		$(LN) -f $(INSTSONAME) $@; \
	fi

libpython3.so:	libpython$(LDVERSION).so
	$(BLDSHARED) $(NO_AS_NEEDED) -o $@ -Wl,-h$@ $^

libpython$(LDVERSION).dylib: $(LIBRARY_OBJS)
	 $(CC) -dynamiclib $(PY_CORE_LDFLAGS) -undefined dynamic_lookup -Wl,-install_name,$(prefix)/lib/libpython$(LDVERSION).dylib -Wl,-compatibility_version,$(VERSION) -Wl,-current_version,$(VERSION) -o $@ $(LIBRARY_OBJS) $(DTRACE_OBJS) $(SHLIBS) $(LIBC) $(LIBM); \


libpython$(VERSION).sl: $(LIBRARY_OBJS)
	$(LDSHARED) -o $@ $(LIBRARY_OBJS) $(MODLIBS) $(SHLIBS) $(LIBC) $(LIBM)

# List of exported symbols for AIX
Modules/python.exp: $(LIBRARY)
	$(srcdir)/Modules/makexp_aix $@ "$(EXPORTSFROM)" $?

# Copy up the gdb python hooks into a position where they can be automatically
# loaded by gdb during Lib/test/test_gdb.py
#
# Distributors are likely to want to install this somewhere else e.g. relative
# to the stripped DWARF data for the shared library.
.PHONY: gdbhooks
gdbhooks: $(BUILDPYTHON)-gdb.py

SRC_GDB_HOOKS=$(srcdir)/Tools/gdb/libpython.py
$(BUILDPYTHON)-gdb.py: $(SRC_GDB_HOOKS)
	$(INSTALL_DATA) $(SRC_GDB_HOOKS) $(BUILDPYTHON)-gdb.py

# This rule is here for OPENSTEP/Rhapsody/MacOSX. It builds a temporary
# minimal framework (not including the Lib directory and such) in the current
# directory.
$(PYTHONFRAMEWORKDIR)/Versions/$(VERSION)/$(PYTHONFRAMEWORK): \
		$(LIBRARY) \
		$(RESSRCDIR)/Info.plist
	$(INSTALL) -d -m $(DIRMODE) $(PYTHONFRAMEWORKDIR)/Versions/$(VERSION)
	$(CC) -o $(LDLIBRARY) $(PY_CORE_LDFLAGS) -dynamiclib \
		-all_load $(LIBRARY) \
		-install_name $(DESTDIR)$(PYTHONFRAMEWORKINSTALLNAMEPREFIX)/$(PYTHONFRAMEWORK) \
		-compatibility_version $(VERSION) \
		-current_version $(VERSION) \
		-framework CoreFoundation $(LIBS);
	$(INSTALL) -d -m $(DIRMODE)  \
		$(PYTHONFRAMEWORKDIR)/Versions/$(VERSION)/Resources/English.lproj
	$(INSTALL_DATA) $(RESSRCDIR)/Info.plist \
		$(PYTHONFRAMEWORKDIR)/Versions/$(VERSION)/Resources/Info.plist
	$(LN) -fsn $(VERSION) $(PYTHONFRAMEWORKDIR)/Versions/Current
	$(LN) -fsn Versions/Current/$(PYTHONFRAMEWORK) $(PYTHONFRAMEWORKDIR)/$(PYTHONFRAMEWORK)
	$(LN) -fsn Versions/Current/Resources $(PYTHONFRAMEWORKDIR)/Resources

# This rule is for iOS, which requires an annoyingly just slighly different
# format for frameworks to macOS. It *doesn't* use a versioned framework, and
# the Info.plist must be in the root of the framework.
$(PYTHONFRAMEWORKDIR)/$(PYTHONFRAMEWORK): \
		$(LIBRARY) \
		$(RESSRCDIR)/Info.plist
	$(INSTALL) -d -m $(DIRMODE) $(PYTHONFRAMEWORKDIR)
	$(CC) -o $(LDLIBRARY) $(PY_CORE_LDFLAGS) -dynamiclib \
		-all_load $(LIBRARY) \
		-install_name $(PYTHONFRAMEWORKINSTALLNAMEPREFIX)/$(PYTHONFRAMEWORK) \
		-compatibility_version $(VERSION) \
		-current_version $(VERSION) \
		-framework CoreFoundation $(LIBS);
	$(INSTALL_DATA) $(RESSRCDIR)/Info.plist $(PYTHONFRAMEWORKDIR)/Info.plist

# This rule builds the Cygwin Python DLL and import library if configured
# for a shared core library; otherwise, this rule is a noop.
$(DLLLIBRARY) libpython$(LDVERSION).dll.a: $(LIBRARY_OBJS)
	if test -n "$(DLLLIBRARY)"; then \
		$(LDSHARED) -Wl,--out-implib=$@ -o $(DLLLIBRARY) $^ \
			$(LIBS) $(MODLIBS) $(SYSLIBS); \
	else true; \
	fi

# wasm32-emscripten browser build
# wasm assets directory is relative to current build dir, e.g. "./usr/local".
# --preload-file turns a relative asset path into an absolute path.

.PHONY: wasm_stdlib
wasm_stdlib: $(WASM_STDLIB)
$(WASM_STDLIB): $(srcdir)/Lib/*.py $(srcdir)/Lib/*/*.py \
	    $(srcdir)/Tools/wasm/wasm_assets.py \
	    Makefile pybuilddir.txt Modules/Setup.local
	$(PYTHON_FOR_BUILD) $(srcdir)/Tools/wasm/wasm_assets.py \
	    --buildroot . --prefix $(prefix)

python.html: $(srcdir)/Tools/wasm/python.html python.worker.js
	@cp $(srcdir)/Tools/wasm/python.html $@

python.worker.js: $(srcdir)/Tools/wasm/python.worker.js
	@cp $(srcdir)/Tools/wasm/python.worker.js $@

############################################################################
# Header files

PYTHON_HEADERS= \
		$(srcdir)/Include/Python.h \
		$(srcdir)/Include/abstract.h \
		$(srcdir)/Include/bltinmodule.h \
		$(srcdir)/Include/boolobject.h \
		$(srcdir)/Include/bytearrayobject.h \
		$(srcdir)/Include/bytesobject.h \
		$(srcdir)/Include/ceval.h \
		$(srcdir)/Include/codecs.h \
		$(srcdir)/Include/compile.h \
		$(srcdir)/Include/complexobject.h \
		$(srcdir)/Include/critical_section.h \
		$(srcdir)/Include/descrobject.h \
		$(srcdir)/Include/dictobject.h \
		$(srcdir)/Include/dynamic_annotations.h \
		$(srcdir)/Include/enumobject.h \
		$(srcdir)/Include/errcode.h \
		$(srcdir)/Include/exports.h \
		$(srcdir)/Include/fileobject.h \
		$(srcdir)/Include/fileutils.h \
		$(srcdir)/Include/floatobject.h \
		$(srcdir)/Include/frameobject.h \
		$(srcdir)/Include/genericaliasobject.h \
		$(srcdir)/Include/import.h \
		$(srcdir)/Include/intrcheck.h \
		$(srcdir)/Include/iterobject.h \
		$(srcdir)/Include/listobject.h \
		$(srcdir)/Include/lock.h \
		$(srcdir)/Include/longobject.h \
		$(srcdir)/Include/marshal.h \
		$(srcdir)/Include/memoryobject.h \
		$(srcdir)/Include/methodobject.h \
		$(srcdir)/Include/modsupport.h \
		$(srcdir)/Include/moduleobject.h \
		$(srcdir)/Include/monitoring.h \
		$(srcdir)/Include/object.h \
		$(srcdir)/Include/objimpl.h \
		$(srcdir)/Include/opcode.h \
		$(srcdir)/Include/opcode_ids.h \
		$(srcdir)/Include/osdefs.h \
		$(srcdir)/Include/osmodule.h \
		$(srcdir)/Include/patchlevel.h \
		$(srcdir)/Include/pyatomic.h \
		$(srcdir)/Include/pybuffer.h \
		$(srcdir)/Include/pycapsule.h \
		$(srcdir)/Include/pydtrace.h \
		$(srcdir)/Include/pyerrors.h \
		$(srcdir)/Include/pyexpat.h \
		$(srcdir)/Include/pyframe.h \
		$(srcdir)/Include/pyhash.h \
		$(srcdir)/Include/pylifecycle.h \
		$(srcdir)/Include/pymacconfig.h \
		$(srcdir)/Include/pymacro.h \
		$(srcdir)/Include/pymath.h \
		$(srcdir)/Include/pymem.h \
		$(srcdir)/Include/pyport.h \
		$(srcdir)/Include/pystate.h \
		$(srcdir)/Include/pystats.h \
		$(srcdir)/Include/pystrcmp.h \
		$(srcdir)/Include/pystrtod.h \
		$(srcdir)/Include/pythonrun.h \
		$(srcdir)/Include/pythread.h \
		$(srcdir)/Include/pytypedefs.h \
		$(srcdir)/Include/rangeobject.h \
		$(srcdir)/Include/setobject.h \
		$(srcdir)/Include/sliceobject.h \
		$(srcdir)/Include/structmember.h \
		$(srcdir)/Include/structseq.h \
		$(srcdir)/Include/sysmodule.h \
		$(srcdir)/Include/traceback.h \
		$(srcdir)/Include/tupleobject.h \
		$(srcdir)/Include/typeslots.h \
		$(srcdir)/Include/unicodeobject.h \
		$(srcdir)/Include/warnings.h \
		$(srcdir)/Include/weakrefobject.h \
		\
		pyconfig.h \
		$(PARSER_HEADERS) \
		\
		$(srcdir)/Include/cpython/abstract.h \
		$(srcdir)/Include/cpython/bytearrayobject.h \
		$(srcdir)/Include/cpython/bytesobject.h \
		$(srcdir)/Include/cpython/cellobject.h \
		$(srcdir)/Include/cpython/ceval.h \
		$(srcdir)/Include/cpython/classobject.h \
		$(srcdir)/Include/cpython/code.h \
		$(srcdir)/Include/cpython/compile.h \
		$(srcdir)/Include/cpython/complexobject.h \
		$(srcdir)/Include/cpython/context.h \
		$(srcdir)/Include/cpython/critical_section.h \
		$(srcdir)/Include/cpython/descrobject.h \
		$(srcdir)/Include/cpython/dictobject.h \
		$(srcdir)/Include/cpython/fileobject.h \
		$(srcdir)/Include/cpython/fileutils.h \
		$(srcdir)/Include/cpython/floatobject.h \
		$(srcdir)/Include/cpython/frameobject.h \
		$(srcdir)/Include/cpython/funcobject.h \
		$(srcdir)/Include/cpython/genobject.h \
		$(srcdir)/Include/cpython/import.h \
		$(srcdir)/Include/cpython/initconfig.h \
		$(srcdir)/Include/cpython/listobject.h \
		$(srcdir)/Include/cpython/lock.h \
		$(srcdir)/Include/cpython/longintrepr.h \
		$(srcdir)/Include/cpython/longobject.h \
		$(srcdir)/Include/cpython/memoryobject.h \
		$(srcdir)/Include/cpython/methodobject.h \
		$(srcdir)/Include/cpython/modsupport.h \
		$(srcdir)/Include/cpython/monitoring.h \
		$(srcdir)/Include/cpython/object.h \
		$(srcdir)/Include/cpython/objimpl.h \
		$(srcdir)/Include/cpython/odictobject.h \
		$(srcdir)/Include/cpython/picklebufobject.h \
		$(srcdir)/Include/cpython/pthread_stubs.h \
		$(srcdir)/Include/cpython/pyatomic.h \
		$(srcdir)/Include/cpython/pyatomic_gcc.h \
		$(srcdir)/Include/cpython/pyatomic_std.h \
		$(srcdir)/Include/cpython/pyctype.h \
		$(srcdir)/Include/cpython/pydebug.h \
		$(srcdir)/Include/cpython/pyerrors.h \
		$(srcdir)/Include/cpython/pyfpe.h \
		$(srcdir)/Include/cpython/pyframe.h \
		$(srcdir)/Include/cpython/pyhash.h \
		$(srcdir)/Include/cpython/pylifecycle.h \
		$(srcdir)/Include/cpython/pymem.h \
		$(srcdir)/Include/cpython/pystate.h \
		$(srcdir)/Include/cpython/pystats.h \
		$(srcdir)/Include/cpython/pythonrun.h \
		$(srcdir)/Include/cpython/pythread.h \
		$(srcdir)/Include/cpython/setobject.h \
		$(srcdir)/Include/cpython/sysmodule.h \
		$(srcdir)/Include/cpython/traceback.h \
		$(srcdir)/Include/cpython/tracemalloc.h \
		$(srcdir)/Include/cpython/tupleobject.h \
		$(srcdir)/Include/cpython/unicodeobject.h \
		$(srcdir)/Include/cpython/warnings.h \
		$(srcdir)/Include/cpython/weakrefobject.h \
		\
		$(MIMALLOC_HEADERS) \
		\
		$(srcdir)/Include/internal/pycore_abstract.h \
		$(srcdir)/Include/internal/pycore_asdl.h \
		$(srcdir)/Include/internal/pycore_ast.h \
		$(srcdir)/Include/internal/pycore_ast_state.h \
		$(srcdir)/Include/internal/pycore_atexit.h \
		$(srcdir)/Include/internal/pycore_backoff.h \
		$(srcdir)/Include/internal/pycore_bitutils.h \
		$(srcdir)/Include/internal/pycore_blocks_output_buffer.h \
		$(srcdir)/Include/internal/pycore_brc.h \
		$(srcdir)/Include/internal/pycore_bytes_methods.h \
		$(srcdir)/Include/internal/pycore_bytesobject.h \
		$(srcdir)/Include/internal/pycore_call.h \
		$(srcdir)/Include/internal/pycore_capsule.h \
		$(srcdir)/Include/internal/pycore_cell.h \
		$(srcdir)/Include/internal/pycore_ceval.h \
		$(srcdir)/Include/internal/pycore_ceval_state.h \
		$(srcdir)/Include/internal/pycore_code.h \
		$(srcdir)/Include/internal/pycore_codecs.h \
		$(srcdir)/Include/internal/pycore_compile.h \
		$(srcdir)/Include/internal/pycore_complexobject.h \
		$(srcdir)/Include/internal/pycore_condvar.h \
		$(srcdir)/Include/internal/pycore_context.h \
		$(srcdir)/Include/internal/pycore_critical_section.h \
		$(srcdir)/Include/internal/pycore_crossinterp.h \
		$(srcdir)/Include/internal/pycore_descrobject.h \
		$(srcdir)/Include/internal/pycore_dict.h \
		$(srcdir)/Include/internal/pycore_dict_state.h \
		$(srcdir)/Include/internal/pycore_dtoa.h \
		$(srcdir)/Include/internal/pycore_exceptions.h \
		$(srcdir)/Include/internal/pycore_faulthandler.h \
		$(srcdir)/Include/internal/pycore_fileutils.h \
		$(srcdir)/Include/internal/pycore_floatobject.h \
		$(srcdir)/Include/internal/pycore_flowgraph.h \
		$(srcdir)/Include/internal/pycore_format.h \
		$(srcdir)/Include/internal/pycore_frame.h \
		$(srcdir)/Include/internal/pycore_freelist.h \
		$(srcdir)/Include/internal/pycore_function.h \
		$(srcdir)/Include/internal/pycore_gc.h \
		$(srcdir)/Include/internal/pycore_genobject.h \
		$(srcdir)/Include/internal/pycore_getopt.h \
		$(srcdir)/Include/internal/pycore_gil.h \
		$(srcdir)/Include/internal/pycore_global_objects.h \
		$(srcdir)/Include/internal/pycore_global_objects_fini_generated.h \
		$(srcdir)/Include/internal/pycore_global_strings.h \
		$(srcdir)/Include/internal/pycore_hamt.h \
		$(srcdir)/Include/internal/pycore_hashtable.h \
		$(srcdir)/Include/internal/pycore_identifier.h \
		$(srcdir)/Include/internal/pycore_import.h \
		$(srcdir)/Include/internal/pycore_importdl.h \
		$(srcdir)/Include/internal/pycore_initconfig.h \
		$(srcdir)/Include/internal/pycore_instruments.h \
		$(srcdir)/Include/internal/pycore_instruction_sequence.h \
		$(srcdir)/Include/internal/pycore_interp.h \
		$(srcdir)/Include/internal/pycore_intrinsics.h \
		$(srcdir)/Include/internal/pycore_jit.h \
		$(srcdir)/Include/internal/pycore_list.h \
		$(srcdir)/Include/internal/pycore_llist.h \
		$(srcdir)/Include/internal/pycore_lock.h \
		$(srcdir)/Include/internal/pycore_long.h \
		$(srcdir)/Include/internal/pycore_memoryobject.h \
		$(srcdir)/Include/internal/pycore_mimalloc.h \
		$(srcdir)/Include/internal/pycore_modsupport.h \
		$(srcdir)/Include/internal/pycore_moduleobject.h \
		$(srcdir)/Include/internal/pycore_namespace.h \
		$(srcdir)/Include/internal/pycore_object.h \
		$(srcdir)/Include/internal/pycore_object_alloc.h \
		$(srcdir)/Include/internal/pycore_object_stack.h \
		$(srcdir)/Include/internal/pycore_object_state.h \
		$(srcdir)/Include/internal/pycore_obmalloc.h \
		$(srcdir)/Include/internal/pycore_obmalloc_init.h \
		$(srcdir)/Include/internal/pycore_opcode_metadata.h \
		$(srcdir)/Include/internal/pycore_opcode_utils.h \
		$(srcdir)/Include/internal/pycore_optimizer.h \
		$(srcdir)/Include/internal/pycore_parking_lot.h \
		$(srcdir)/Include/internal/pycore_parser.h \
		$(srcdir)/Include/internal/pycore_pathconfig.h \
		$(srcdir)/Include/internal/pycore_pyarena.h \
		$(srcdir)/Include/internal/pycore_pyatomic_ft_wrappers.h \
		$(srcdir)/Include/internal/pycore_pybuffer.h \
		$(srcdir)/Include/internal/pycore_pyerrors.h \
		$(srcdir)/Include/internal/pycore_pyhash.h \
		$(srcdir)/Include/internal/pycore_pylifecycle.h \
		$(srcdir)/Include/internal/pycore_pymath.h \
		$(srcdir)/Include/internal/pycore_pymem.h \
		$(srcdir)/Include/internal/pycore_pymem_init.h \
		$(srcdir)/Include/internal/pycore_pystate.h \
		$(srcdir)/Include/internal/pycore_pystats.h \
		$(srcdir)/Include/internal/pycore_pythonrun.h \
		$(srcdir)/Include/internal/pycore_pythread.h \
		$(srcdir)/Include/internal/pycore_qsbr.h \
		$(srcdir)/Include/internal/pycore_range.h \
		$(srcdir)/Include/internal/pycore_runtime.h \
		$(srcdir)/Include/internal/pycore_runtime_init.h \
		$(srcdir)/Include/internal/pycore_runtime_init_generated.h \
		$(srcdir)/Include/internal/pycore_semaphore.h \
		$(srcdir)/Include/internal/pycore_setobject.h \
		$(srcdir)/Include/internal/pycore_signal.h \
		$(srcdir)/Include/internal/pycore_sliceobject.h \
		$(srcdir)/Include/internal/pycore_strhex.h \
		$(srcdir)/Include/internal/pycore_structseq.h \
		$(srcdir)/Include/internal/pycore_symtable.h \
		$(srcdir)/Include/internal/pycore_sysmodule.h \
		$(srcdir)/Include/internal/pycore_stackref.h \
		$(srcdir)/Include/internal/pycore_time.h \
		$(srcdir)/Include/internal/pycore_token.h \
		$(srcdir)/Include/internal/pycore_traceback.h \
		$(srcdir)/Include/internal/pycore_tracemalloc.h \
		$(srcdir)/Include/internal/pycore_tstate.h \
		$(srcdir)/Include/internal/pycore_tuple.h \
		$(srcdir)/Include/internal/pycore_typeobject.h \
		$(srcdir)/Include/internal/pycore_typevarobject.h \
		$(srcdir)/Include/internal/pycore_ucnhash.h \
		$(srcdir)/Include/internal/pycore_unicodeobject.h \
		$(srcdir)/Include/internal/pycore_unicodeobject_generated.h \
		$(srcdir)/Include/internal/pycore_unionobject.h \
		$(srcdir)/Include/internal/pycore_uop_ids.h \
		$(srcdir)/Include/internal/pycore_uop_metadata.h \
		$(srcdir)/Include/internal/pycore_warnings.h \
		$(srcdir)/Include/internal/pycore_weakref.h \
		$(DTRACE_HEADERS) \
		 \
		\
		$(srcdir)/Python/stdlib_module_names.h

##########################################################################
# Build static libmpdec.a
LIBMPDEC_CFLAGS=-I$(srcdir)/Modules/_decimal/libmpdec -DTEST_COVERAGE -DCONFIG_64=1 -DANSI=1 -DHAVE_UINT128_T=1 $(PY_STDMODULE_CFLAGS) $(CCSHARED)

# "%.o: %c" is not portable
Modules/_decimal/libmpdec/basearith.o: $(srcdir)/Modules/_decimal/libmpdec/basearith.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/basearith.c

Modules/_decimal/libmpdec/constants.o: $(srcdir)/Modules/_decimal/libmpdec/constants.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/constants.c

Modules/_decimal/libmpdec/context.o: $(srcdir)/Modules/_decimal/libmpdec/context.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/context.c

Modules/_decimal/libmpdec/convolute.o: $(srcdir)/Modules/_decimal/libmpdec/convolute.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/convolute.c

Modules/_decimal/libmpdec/crt.o: $(srcdir)/Modules/_decimal/libmpdec/crt.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/crt.c

Modules/_decimal/libmpdec/difradix2.o: $(srcdir)/Modules/_decimal/libmpdec/difradix2.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/difradix2.c

Modules/_decimal/libmpdec/fnt.o: $(srcdir)/Modules/_decimal/libmpdec/fnt.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/fnt.c

Modules/_decimal/libmpdec/fourstep.o: $(srcdir)/Modules/_decimal/libmpdec/fourstep.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/fourstep.c

Modules/_decimal/libmpdec/io.o: $(srcdir)/Modules/_decimal/libmpdec/io.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/io.c

Modules/_decimal/libmpdec/mpalloc.o: $(srcdir)/Modules/_decimal/libmpdec/mpalloc.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/mpalloc.c

Modules/_decimal/libmpdec/mpdecimal.o: $(srcdir)/Modules/_decimal/libmpdec/mpdecimal.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/mpdecimal.c

Modules/_decimal/libmpdec/mpsignal.o: $(srcdir)/Modules/_decimal/libmpdec/mpsignal.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/mpsignal.c

Modules/_decimal/libmpdec/numbertheory.o: $(srcdir)/Modules/_decimal/libmpdec/numbertheory.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/numbertheory.c

Modules/_decimal/libmpdec/sixstep.o: $(srcdir)/Modules/_decimal/libmpdec/sixstep.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/sixstep.c

Modules/_decimal/libmpdec/transpose.o: $(srcdir)/Modules/_decimal/libmpdec/transpose.c $(LIBMPDEC_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBMPDEC_CFLAGS) -o $@ $(srcdir)/Modules/_decimal/libmpdec/transpose.c

$(LIBMPDEC_A): $(LIBMPDEC_OBJS)
	-rm -f $@
	$(AR) $(ARFLAGS) $@ $(LIBMPDEC_OBJS)

##########################################################################
# Build static libexpat.a
LIBEXPAT_CFLAGS=-I$(srcdir)/Modules/expat $(PY_STDMODULE_CFLAGS) $(CCSHARED)

Modules/expat/xmlparse.o: $(srcdir)/Modules/expat/xmlparse.c $(LIBEXPAT_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBEXPAT_CFLAGS) -o $@ $(srcdir)/Modules/expat/xmlparse.c

Modules/expat/xmlrole.o: $(srcdir)/Modules/expat/xmlrole.c $(LIBEXPAT_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBEXPAT_CFLAGS) -o $@ $(srcdir)/Modules/expat/xmlrole.c

Modules/expat/xmltok.o: $(srcdir)/Modules/expat/xmltok.c $(LIBEXPAT_HEADERS) $(PYTHON_HEADERS)
	$(CC) -c $(LIBEXPAT_CFLAGS) -o $@ $(srcdir)/Modules/expat/xmltok.c

$(LIBEXPAT_A): $(LIBEXPAT_OBJS)
	-rm -f $@
	$(AR) $(ARFLAGS) $@ $(LIBEXPAT_OBJS)

##########################################################################
# Build HACL* static libraries for hashlib: libHacl_Hash_SHA2.a
LIBHACL_CFLAGS=-I$(srcdir)/Modules/_hacl/include -D_BSD_SOURCE -D_DEFAULT_SOURCE $(PY_STDMODULE_CFLAGS) $(CCSHARED)

Modules/_hacl/Hacl_Hash_SHA2.o: $(srcdir)/Modules/_hacl/Hacl_Hash_SHA2.c $(LIBHACL_SHA2_HEADERS)
	$(CC) -c $(LIBHACL_CFLAGS) -o $@ $(srcdir)/Modules/_hacl/Hacl_Hash_SHA2.c

$(LIBHACL_SHA2_A): $(LIBHACL_SHA2_OBJS)
	-rm -f $@
	$(AR) $(ARFLAGS) $@ $(LIBHACL_SHA2_OBJS)

# create relative links from build/lib.platform/egg.so to Modules/egg.so
# pybuilddir.txt is created too late. We cannot use it in Makefile
# targets. ln --relative is not portable.
.PHONY: sharedmods
sharedmods: $(SHAREDMODS) pybuilddir.txt
	@target=`cat pybuilddir.txt`; \
	$(MKDIR_P) $$target; \
	for mod in X $(SHAREDMODS); do \
		if test $$mod != X; then \
			$(LN) -sf ../../$$mod $$target/`basename $$mod`; \
		fi; \
	done

# dependency on BUILDPYTHON ensures that the target is run last
.PHONY: checksharedmods
checksharedmods: sharedmods $(PYTHON_FOR_BUILD_DEPS) $(BUILDPYTHON)
	@$(RUNSHARED) $(PYTHON_FOR_BUILD) $(srcdir)/Tools/build/check_extension_modules.py

.PHONY: rundsymutil
rundsymutil: sharedmods $(PYTHON_FOR_BUILD_DEPS) $(BUILDPYTHON)
	@if [ ! -z $(DSYMUTIL) ] ; then \
		echo $(DSYMUTIL_PATH) $(BUILDPYTHON); \
		$(DSYMUTIL_PATH) $(BUILDPYTHON); \
		if test -f $(LDLIBRARY); then \
			echo $(DSYMUTIL_PATH) $(LDLIBRARY); \
			$(DSYMUTIL_PATH) $(LDLIBRARY); \
		fi; \
		for mod in X $(SHAREDMODS); do \
			if test $$mod != X; then \
				echo $(DSYMUTIL_PATH) $$mod; \
				$(DSYMUTIL_PATH) $$mod; \
			fi; \
		done \
	fi

Modules/Setup.local:
	@# Create empty Setup.local when file was deleted by user
	echo "# Edit this file for local setup changes" > $@

Modules/Setup.bootstrap: $(srcdir)/Modules/Setup.bootstrap.in config.status
	./config.status $@

Modules/Setup.stdlib: $(srcdir)/Modules/Setup.stdlib.in config.status
	./config.status $@

Makefile Modules/config.c: Makefile.pre \
				$(srcdir)/Modules/config.c.in \
				$(MAKESETUP) \
				$(srcdir)/Modules/Setup \
				Modules/Setup.local \
				Modules/Setup.bootstrap \
				Modules/Setup.stdlib
	$(MAKESETUP) -c $(srcdir)/Modules/config.c.in \
				-s Modules \
				Modules/Setup.local \
				Modules/Setup.stdlib \
				Modules/Setup.bootstrap \
				$(srcdir)/Modules/Setup
	@mv config.c Modules
	@echo "The Makefile was updated, you may need to re-run make."

.PHONY: regen-test-frozenmain
regen-test-frozenmain: $(BUILDPYTHON)
	# Regenerate Programs/test_frozenmain.h
	# from Programs/test_frozenmain.py
	# using Programs/freeze_test_frozenmain.py
	$(RUNSHARED) ./$(BUILDPYTHON) $(srcdir)/Programs/freeze_test_frozenmain.py Programs/test_frozenmain.h

.PHONY: regen-test-levenshtein
regen-test-levenshtein:
	# Regenerate Lib/test/levenshtein_examples.json
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/generate_levenshtein_examples.py $(srcdir)/Lib/test/levenshtein_examples.json

.PHONY: regen-re
regen-re: $(BUILDPYTHON)
	# Regenerate Lib/re/_casefix.py
	# using Tools/build/generate_re_casefix.py
	$(RUNSHARED) ./$(BUILDPYTHON) $(srcdir)/Tools/build/generate_re_casefix.py $(srcdir)/Lib/re/_casefix.py

Programs/_testembed: Programs/_testembed.o $(LINK_PYTHON_DEPS)
	$(LINKCC) $(PY_CORE_LDFLAGS) $(LINKFORSHARED) -o $@ Programs/_testembed.o $(LINK_PYTHON_OBJS) $(LIBS) $(MODLIBS) $(SYSLIBS)

############################################################################
# "Bootstrap Python" used to run Programs/_freeze_module.py

BOOTSTRAP_HEADERS = \
	Python/frozen_modules/importlib._bootstrap.h \
	Python/frozen_modules/importlib._bootstrap_external.h \
	Python/frozen_modules/zipimport.h

Programs/_bootstrap_python.o: Programs/_bootstrap_python.c $(BOOTSTRAP_HEADERS) $(PYTHON_HEADERS)

_bootstrap_python: $(LIBRARY_OBJS_OMIT_FROZEN) Programs/_bootstrap_python.o Modules/getpath.o Modules/Setup.local
	$(LINKCC) $(PY_LDFLAGS_NOLTO) -o $@ $(LIBRARY_OBJS_OMIT_FROZEN) \
		Programs/_bootstrap_python.o Modules/getpath.o $(LIBS) $(MODLIBS) $(SYSLIBS)


############################################################################
# frozen modules (including importlib)
#
# Freezing is a multi step process. It works differently for standard builds
# and cross builds. Standard builds use Programs/_freeze_module and
# _bootstrap_python for freezing, so users can build Python
# without an existing Python installation. Cross builds cannot execute
# compiled binaries and therefore rely on an external build Python
# interpreter. The build interpreter must have same version and same bytecode
# as the host (target) binary.
#
# Standard build process:
# 1) compile minimal core objects for Py_Compile*() and PyMarshal_Write*().
# 2) build Programs/_freeze_module binary.
# 3) create frozen module headers for importlib and getpath.
# 4) build _bootstrap_python binary.
# 5) create remaining frozen module headers with
#    ``./_bootstrap_python Programs/_freeze_module.py``. The pure Python
#    script is used to test the cross compile code path.
#
# Cross compile process:
# 1) create all frozen module headers with external build Python and
#    Programs/_freeze_module.py script.
#

# FROZEN_FILES_* are auto-generated by Tools/build/freeze_modules.py.
FROZEN_FILES_IN = \
		Lib/importlib/_bootstrap.py \
		Lib/importlib/_bootstrap_external.py \
		Lib/zipimport.py \
		Lib/abc.py \
		Lib/codecs.py \
		Lib/io.py \
		Lib/_collections_abc.py \
		Lib/_sitebuiltins.py \
		Lib/genericpath.py \
		Lib/ntpath.py \
		Lib/posixpath.py \
		Lib/os.py \
		Lib/site.py \
		Lib/stat.py \
		Lib/importlib/util.py \
		Lib/importlib/machinery.py \
		Lib/runpy.py \
		Lib/__hello__.py \
		Lib/__phello__/__init__.py \
		Lib/__phello__/ham/__init__.py \
		Lib/__phello__/ham/eggs.py \
		Lib/__phello__/spam.py \
		Tools/freeze/flag.py
# End FROZEN_FILES_IN
FROZEN_FILES_OUT = \
		Python/frozen_modules/importlib._bootstrap.h \
		Python/frozen_modules/importlib._bootstrap_external.h \
		Python/frozen_modules/zipimport.h \
		Python/frozen_modules/abc.h \
		Python/frozen_modules/codecs.h \
		Python/frozen_modules/io.h \
		Python/frozen_modules/_collections_abc.h \
		Python/frozen_modules/_sitebuiltins.h \
		Python/frozen_modules/genericpath.h \
		Python/frozen_modules/ntpath.h \
		Python/frozen_modules/posixpath.h \
		Python/frozen_modules/os.h \
		Python/frozen_modules/site.h \
		Python/frozen_modules/stat.h \
		Python/frozen_modules/importlib.util.h \
		Python/frozen_modules/importlib.machinery.h \
		Python/frozen_modules/runpy.h \
		Python/frozen_modules/__hello__.h \
		Python/frozen_modules/__phello__.h \
		Python/frozen_modules/__phello__.ham.h \
		Python/frozen_modules/__phello__.ham.eggs.h \
		Python/frozen_modules/__phello__.spam.h \
		Python/frozen_modules/frozen_only.h
# End FROZEN_FILES_OUT

Programs/_freeze_module.o: Programs/_freeze_module.c Makefile

Modules/getpath_noop.o: $(srcdir)/Modules/getpath_noop.c Makefile

Programs/_freeze_module: Programs/_freeze_module.o Modules/getpath_noop.o $(LIBRARY_OBJS_OMIT_FROZEN)
	$(LINKCC) $(PY_CORE_LDFLAGS) -o $@ Programs/_freeze_module.o Modules/getpath_noop.o $(LIBRARY_OBJS_OMIT_FROZEN) $(LIBS) $(MODLIBS) $(SYSLIBS)

# We manually freeze getpath.py rather than through freeze_modules
Python/frozen_modules/getpath.h: Modules/getpath.py $(FREEZE_MODULE_BOOTSTRAP_DEPS)
	$(FREEZE_MODULE_BOOTSTRAP) getpath $(srcdir)/Modules/getpath.py Python/frozen_modules/getpath.h

# BEGIN: freezing modules

Python/frozen_modules/importlib._bootstrap.h: Lib/importlib/_bootstrap.py $(FREEZE_MODULE_BOOTSTRAP_DEPS)
	$(FREEZE_MODULE_BOOTSTRAP) importlib._bootstrap $(srcdir)/Lib/importlib/_bootstrap.py Python/frozen_modules/importlib._bootstrap.h

Python/frozen_modules/importlib._bootstrap_external.h: Lib/importlib/_bootstrap_external.py $(FREEZE_MODULE_BOOTSTRAP_DEPS)
	$(FREEZE_MODULE_BOOTSTRAP) importlib._bootstrap_external $(srcdir)/Lib/importlib/_bootstrap_external.py Python/frozen_modules/importlib._bootstrap_external.h

Python/frozen_modules/zipimport.h: Lib/zipimport.py $(FREEZE_MODULE_BOOTSTRAP_DEPS)
	$(FREEZE_MODULE_BOOTSTRAP) zipimport $(srcdir)/Lib/zipimport.py Python/frozen_modules/zipimport.h

Python/frozen_modules/abc.h: Lib/abc.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) abc $(srcdir)/Lib/abc.py Python/frozen_modules/abc.h

Python/frozen_modules/codecs.h: Lib/codecs.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) codecs $(srcdir)/Lib/codecs.py Python/frozen_modules/codecs.h

Python/frozen_modules/io.h: Lib/io.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) io $(srcdir)/Lib/io.py Python/frozen_modules/io.h

Python/frozen_modules/_collections_abc.h: Lib/_collections_abc.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) _collections_abc $(srcdir)/Lib/_collections_abc.py Python/frozen_modules/_collections_abc.h

Python/frozen_modules/_sitebuiltins.h: Lib/_sitebuiltins.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) _sitebuiltins $(srcdir)/Lib/_sitebuiltins.py Python/frozen_modules/_sitebuiltins.h

Python/frozen_modules/genericpath.h: Lib/genericpath.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) genericpath $(srcdir)/Lib/genericpath.py Python/frozen_modules/genericpath.h

Python/frozen_modules/ntpath.h: Lib/ntpath.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) ntpath $(srcdir)/Lib/ntpath.py Python/frozen_modules/ntpath.h

Python/frozen_modules/posixpath.h: Lib/posixpath.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) posixpath $(srcdir)/Lib/posixpath.py Python/frozen_modules/posixpath.h

Python/frozen_modules/os.h: Lib/os.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) os $(srcdir)/Lib/os.py Python/frozen_modules/os.h

Python/frozen_modules/site.h: Lib/site.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) site $(srcdir)/Lib/site.py Python/frozen_modules/site.h

Python/frozen_modules/stat.h: Lib/stat.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) stat $(srcdir)/Lib/stat.py Python/frozen_modules/stat.h

Python/frozen_modules/importlib.util.h: Lib/importlib/util.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) importlib.util $(srcdir)/Lib/importlib/util.py Python/frozen_modules/importlib.util.h

Python/frozen_modules/importlib.machinery.h: Lib/importlib/machinery.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) importlib.machinery $(srcdir)/Lib/importlib/machinery.py Python/frozen_modules/importlib.machinery.h

Python/frozen_modules/runpy.h: Lib/runpy.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) runpy $(srcdir)/Lib/runpy.py Python/frozen_modules/runpy.h

Python/frozen_modules/__hello__.h: Lib/__hello__.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) __hello__ $(srcdir)/Lib/__hello__.py Python/frozen_modules/__hello__.h

Python/frozen_modules/__phello__.h: Lib/__phello__/__init__.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) __phello__ $(srcdir)/Lib/__phello__/__init__.py Python/frozen_modules/__phello__.h

Python/frozen_modules/__phello__.ham.h: Lib/__phello__/ham/__init__.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) __phello__.ham $(srcdir)/Lib/__phello__/ham/__init__.py Python/frozen_modules/__phello__.ham.h

Python/frozen_modules/__phello__.ham.eggs.h: Lib/__phello__/ham/eggs.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) __phello__.ham.eggs $(srcdir)/Lib/__phello__/ham/eggs.py Python/frozen_modules/__phello__.ham.eggs.h

Python/frozen_modules/__phello__.spam.h: Lib/__phello__/spam.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) __phello__.spam $(srcdir)/Lib/__phello__/spam.py Python/frozen_modules/__phello__.spam.h

Python/frozen_modules/frozen_only.h: Tools/freeze/flag.py $(FREEZE_MODULE_DEPS)
	$(FREEZE_MODULE) frozen_only $(srcdir)/Tools/freeze/flag.py Python/frozen_modules/frozen_only.h

# END: freezing modules

Tools/build/freeze_modules.py: $(FREEZE_MODULE)

.PHONY: regen-frozen
regen-frozen: Tools/build/freeze_modules.py $(FROZEN_FILES_IN)
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/freeze_modules.py --frozen-modules
	@echo "The Makefile was updated, you may need to re-run make."

# We keep this renamed target around for folks with muscle memory.
.PHONY: regen-importlib
regen-importlib: regen-frozen

############################################################################
# Global objects

# Dependencies which can add and/or remove _Py_ID() identifiers:
# - "make clinic"
.PHONY: regen-global-objects
regen-global-objects: $(srcdir)/Tools/build/generate_global_objects.py clinic
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/generate_global_objects.py

############################################################################
# ABI

.PHONY: regen-abidump
regen-abidump: all
	@$(MKDIR_P) $(srcdir)/Doc/data/
	abidw "libpython$(LDVERSION).so" --no-architecture --out-file $(srcdir)/Doc/data/python$(LDVERSION).abi.new
	@$(UPDATE_FILE) --create $(srcdir)/Doc/data/python$(LDVERSION).abi $(srcdir)/Doc/data/python$(LDVERSION).abi.new

.PHONY: check-abidump
check-abidump: all
	abidiff $(srcdir)/Doc/data/python$(LDVERSION).abi "libpython$(LDVERSION).so" --drop-private-types --no-architecture --no-added-syms

.PHONY: regen-limited-abi
regen-limited-abi: all
	$(RUNSHARED) ./$(BUILDPYTHON) $(srcdir)/Tools/build/stable_abi.py --generate-all $(srcdir)/Misc/stable_abi.toml

############################################################################
# Regenerate Unicode Data

.PHONY: regen-unicodedata
regen-unicodedata:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/unicode/makeunicodedata.py


############################################################################
# Regenerate all generated files

# "clinic" is regenerated implicitly via "regen-global-objects".
.PHONY: regen-all
regen-all: regen-cases regen-typeslots \
	regen-token regen-ast regen-keyword regen-sre regen-frozen \
	regen-pegen-metaparser regen-pegen regen-test-frozenmain \
	regen-test-levenshtein regen-global-objects
	@echo
	@echo "Note: make regen-stdlib-module-names, make regen-limited-abi, "
	@echo "make regen-configure, make regen-sbom, and make regen-unicodedata should be run manually"

############################################################################
# Special rules for object files

Modules/getbuildinfo.o: $(PARSER_OBJS) \
		$(OBJECT_OBJS) \
		$(PYTHON_OBJS) \
		$(MODULE_OBJS) \
		$(MODOBJS) \
		$(DTRACE_OBJS) \
		$(srcdir)/Modules/getbuildinfo.c
	$(CC) -c $(PY_CORE_CFLAGS) \
	      -DGITVERSION="\"`LC_ALL=C $(GITVERSION)`\"" \
	      -DGITTAG="\"`LC_ALL=C $(GITTAG)`\"" \
	      -DGITBRANCH="\"`LC_ALL=C $(GITBRANCH)`\"" \
	      -o $@ $(srcdir)/Modules/getbuildinfo.c

Modules/getpath.o: $(srcdir)/Modules/getpath.c Python/frozen_modules/getpath.h Makefile $(PYTHON_HEADERS)
	$(CC) -c $(PY_CORE_CFLAGS) -DPYTHONPATH='"$(PYTHONPATH)"' \
		-DPREFIX='"$(prefix)"' \
		-DEXEC_PREFIX='"$(exec_prefix)"' \
		-DVERSION='"$(VERSION)"' \
		-DVPATH='"$(VPATH)"' \
		-DPLATLIBDIR='"$(PLATLIBDIR)"' \
		-DPYTHONFRAMEWORK='"$(PYTHONFRAMEWORK)"' \
		-o $@ $(srcdir)/Modules/getpath.c

Programs/python.o: $(srcdir)/Programs/python.c
	$(CC) -c $(PY_CORE_CFLAGS) -o $@ $(srcdir)/Programs/python.c

Programs/_testembed.o: $(srcdir)/Programs/_testembed.c Programs/test_frozenmain.h $(PYTHON_HEADERS)
	$(CC) -c $(PY_CORE_CFLAGS) -o $@ $(srcdir)/Programs/_testembed.c

Modules/_sre/sre.o: $(srcdir)/Modules/_sre/sre.c $(srcdir)/Modules/_sre/sre.h $(srcdir)/Modules/_sre/sre_constants.h $(srcdir)/Modules/_sre/sre_lib.h

Modules/posixmodule.o: $(srcdir)/Modules/posixmodule.c $(srcdir)/Modules/posixmodule.h

Modules/grpmodule.o: $(srcdir)/Modules/grpmodule.c $(srcdir)/Modules/posixmodule.h

Modules/pwdmodule.o: $(srcdir)/Modules/pwdmodule.c $(srcdir)/Modules/posixmodule.h

Modules/signalmodule.o: $(srcdir)/Modules/signalmodule.c $(srcdir)/Modules/posixmodule.h

Modules/_interpretersmodule.o: $(srcdir)/Modules/_interpretersmodule.c $(srcdir)/Modules/_interpreters_common.h

Modules/_interpqueuesmodule.o: $(srcdir)/Modules/_interpqueuesmodule.c $(srcdir)/Modules/_interpreters_common.h

Modules/_interpchannelsmodule.o: $(srcdir)/Modules/_interpchannelsmodule.c $(srcdir)/Modules/_interpreters_common.h

Python/crossinterp.o: $(srcdir)/Python/crossinterp.c $(srcdir)/Python/crossinterp_data_lookup.h $(srcdir)/Python/crossinterp_exceptions.h

Python/initconfig.o: $(srcdir)/Python/initconfig.c $(srcdir)/Python/config_common.h

Python/interpconfig.o: $(srcdir)/Python/interpconfig.c $(srcdir)/Python/config_common.h

Python/dynload_shlib.o: $(srcdir)/Python/dynload_shlib.c Makefile
	$(CC) -c $(PY_CORE_CFLAGS) \
		-DSOABI='"$(SOABI)"' \
		-o $@ $(srcdir)/Python/dynload_shlib.c

Python/dynload_hpux.o: $(srcdir)/Python/dynload_hpux.c Makefile
	$(CC) -c $(PY_CORE_CFLAGS) \
		-DSHLIB_EXT='"$(EXT_SUFFIX)"' \
		-o $@ $(srcdir)/Python/dynload_hpux.c

Python/sysmodule.o: $(srcdir)/Python/sysmodule.c Makefile $(srcdir)/Include/pydtrace.h
	$(CC) -c $(PY_CORE_CFLAGS) \
		-DABIFLAGS='"$(ABIFLAGS)"' \
		$(MULTIARCH_CPPFLAGS) \
		-o $@ $(srcdir)/Python/sysmodule.c

$(IO_OBJS): $(IO_H)

.PHONY: regen-pegen-metaparser
regen-pegen-metaparser:
	@$(MKDIR_P) $(srcdir)/Tools/peg_generator/pegen
	PYTHONPATH=$(srcdir)/Tools/peg_generator $(PYTHON_FOR_REGEN) -m pegen -q python \
	$(srcdir)/Tools/peg_generator/pegen/metagrammar.gram \
	-o $(srcdir)/Tools/peg_generator/pegen/grammar_parser.py.new
	$(UPDATE_FILE) $(srcdir)/Tools/peg_generator/pegen/grammar_parser.py \
	$(srcdir)/Tools/peg_generator/pegen/grammar_parser.py.new

.PHONY: regen-pegen
regen-pegen:
	@$(MKDIR_P) $(srcdir)/Parser
	@$(MKDIR_P) $(srcdir)/Parser/tokenizer
	@$(MKDIR_P) $(srcdir)/Parser/lexer
	PYTHONPATH=$(srcdir)/Tools/peg_generator $(PYTHON_FOR_REGEN) -m pegen -q c \
		$(srcdir)/Grammar/python.gram \
		$(srcdir)/Grammar/Tokens \
		-o $(srcdir)/Parser/parser.c.new
	$(UPDATE_FILE) $(srcdir)/Parser/parser.c $(srcdir)/Parser/parser.c.new

.PHONY: regen-ast
regen-ast:
	# Regenerate 3 files using using Parser/asdl_c.py:
	# - Include/internal/pycore_ast.h
	# - Include/internal/pycore_ast_state.h
	# - Python/Python-ast.c
	$(MKDIR_P) $(srcdir)/Include
	$(MKDIR_P) $(srcdir)/Python
	$(PYTHON_FOR_REGEN) $(srcdir)/Parser/asdl_c.py \
		$(srcdir)/Parser/Python.asdl \
		-H $(srcdir)/Include/internal/pycore_ast.h.new \
		-I $(srcdir)/Include/internal/pycore_ast_state.h.new \
		-C $(srcdir)/Python/Python-ast.c.new

	$(UPDATE_FILE) $(srcdir)/Include/internal/pycore_ast.h $(srcdir)/Include/internal/pycore_ast.h.new
	$(UPDATE_FILE) $(srcdir)/Include/internal/pycore_ast_state.h $(srcdir)/Include/internal/pycore_ast_state.h.new
	$(UPDATE_FILE) $(srcdir)/Python/Python-ast.c $(srcdir)/Python/Python-ast.c.new

.PHONY: regen-token
regen-token:
	# Regenerate Doc/library/token-list.inc from Grammar/Tokens
	# using Tools/build/generate_token.py
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/generate_token.py rst \
		$(srcdir)/Grammar/Tokens \
		$(srcdir)/Doc/library/token-list.inc
	# Regenerate Include/internal/pycore_token.h from Grammar/Tokens
	# using Tools/build/generate_token.py
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/generate_token.py h \
		$(srcdir)/Grammar/Tokens \
		$(srcdir)/Include/internal/pycore_token.h
	# Regenerate Parser/token.c from Grammar/Tokens
	# using Tools/build/generate_token.py
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/generate_token.py c \
		$(srcdir)/Grammar/Tokens \
		$(srcdir)/Parser/token.c
	# Regenerate Lib/token.py from Grammar/Tokens
	# using Tools/build/generate_token.py
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/generate_token.py py \
		$(srcdir)/Grammar/Tokens \
		$(srcdir)/Lib/token.py

.PHONY: regen-keyword
regen-keyword:
	# Regenerate Lib/keyword.py from Grammar/python.gram and Grammar/Tokens
	# using Tools/peg_generator/pegen
	PYTHONPATH=$(srcdir)/Tools/peg_generator $(PYTHON_FOR_REGEN) -m pegen.keywordgen \
		$(srcdir)/Grammar/python.gram \
		$(srcdir)/Grammar/Tokens \
		$(srcdir)/Lib/keyword.py.new
	$(UPDATE_FILE) $(srcdir)/Lib/keyword.py $(srcdir)/Lib/keyword.py.new

.PHONY: regen-stdlib-module-names
regen-stdlib-module-names: all Programs/_testembed
	# Regenerate Python/stdlib_module_names.h
	# using Tools/build/generate_stdlib_module_names.py
	$(RUNSHARED) ./$(BUILDPYTHON) \
		$(srcdir)/Tools/build/generate_stdlib_module_names.py \
		> $(srcdir)/Python/stdlib_module_names.h.new
	$(UPDATE_FILE) $(srcdir)/Python/stdlib_module_names.h $(srcdir)/Python/stdlib_module_names.h.new

.PHONY: regen-sre
regen-sre:
	# Regenerate Modules/_sre/sre_constants.h and Modules/_sre/sre_targets.h
	# from Lib/re/_constants.py using Tools/build/generate_sre_constants.py
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/generate_sre_constants.py \
		$(srcdir)/Lib/re/_constants.py \
		$(srcdir)/Modules/_sre/sre_constants.h \
		$(srcdir)/Modules/_sre/sre_targets.h

Python/compile.o Python/symtable.o Python/ast_unparse.o Python/ast.o Python/future.o: $(srcdir)/Include/internal/pycore_ast.h $(srcdir)/Include/internal/pycore_ast.h

Python/getplatform.o: $(srcdir)/Python/getplatform.c
		$(CC) -c $(PY_CORE_CFLAGS) -DPLATFORM='"$(MACHDEP)"' -o $@ $(srcdir)/Python/getplatform.c

Python/importdl.o: $(srcdir)/Python/importdl.c
		$(CC) -c $(PY_CORE_CFLAGS) -I$(DLINCLDIR) -o $@ $(srcdir)/Python/importdl.c

Objects/unicodectype.o:	$(srcdir)/Objects/unicodectype.c \
				$(srcdir)/Objects/unicodetype_db.h

BYTESTR_DEPS = \
		$(srcdir)/Objects/stringlib/count.h \
		$(srcdir)/Objects/stringlib/ctype.h \
		$(srcdir)/Objects/stringlib/fastsearch.h \
		$(srcdir)/Objects/stringlib/find.h \
		$(srcdir)/Objects/stringlib/join.h \
		$(srcdir)/Objects/stringlib/partition.h \
		$(srcdir)/Objects/stringlib/split.h \
		$(srcdir)/Objects/stringlib/stringdefs.h \
		$(srcdir)/Objects/stringlib/transmogrify.h

UNICODE_DEPS = \
		$(srcdir)/Objects/stringlib/asciilib.h \
		$(srcdir)/Objects/stringlib/codecs.h \
		$(srcdir)/Objects/stringlib/count.h \
		$(srcdir)/Objects/stringlib/fastsearch.h \
		$(srcdir)/Objects/stringlib/find.h \
		$(srcdir)/Objects/stringlib/find_max_char.h \
		$(srcdir)/Objects/stringlib/localeutil.h \
		$(srcdir)/Objects/stringlib/partition.h \
		$(srcdir)/Objects/stringlib/replace.h \
		$(srcdir)/Objects/stringlib/split.h \
		$(srcdir)/Objects/stringlib/ucs1lib.h \
		$(srcdir)/Objects/stringlib/ucs2lib.h \
		$(srcdir)/Objects/stringlib/ucs4lib.h \
		$(srcdir)/Objects/stringlib/undef.h \
		$(srcdir)/Objects/stringlib/unicode_format.h

Objects/bytes_methods.o: $(srcdir)/Objects/bytes_methods.c $(BYTESTR_DEPS)
Objects/bytesobject.o: $(srcdir)/Objects/bytesobject.c $(BYTESTR_DEPS)
Objects/bytearrayobject.o: $(srcdir)/Objects/bytearrayobject.c $(BYTESTR_DEPS)

Objects/unicodeobject.o: $(srcdir)/Objects/unicodeobject.c $(UNICODE_DEPS)

Objects/dictobject.o: $(srcdir)/Objects/stringlib/eq.h
Objects/setobject.o: $(srcdir)/Objects/stringlib/eq.h

Objects/obmalloc.o: $(srcdir)/Objects/mimalloc/alloc.c \
		$(srcdir)/Objects/mimalloc/alloc-aligned.c \
		$(srcdir)/Objects/mimalloc/alloc-posix.c \
		$(srcdir)/Objects/mimalloc/arena.c \
		$(srcdir)/Objects/mimalloc/bitmap.c \
		$(srcdir)/Objects/mimalloc/heap.c \
		$(srcdir)/Objects/mimalloc/init.c \
		$(srcdir)/Objects/mimalloc/options.c \
		$(srcdir)/Objects/mimalloc/os.c \
		$(srcdir)/Objects/mimalloc/page.c \
		$(srcdir)/Objects/mimalloc/random.c \
		$(srcdir)/Objects/mimalloc/segment.c \
		$(srcdir)/Objects/mimalloc/segment-map.c \
		$(srcdir)/Objects/mimalloc/stats.c \
		$(srcdir)/Objects/mimalloc/prim/prim.c \
		$(srcdir)/Objects/mimalloc/prim/osx/prim.c \
		$(srcdir)/Objects/mimalloc/prim/unix/prim.c \
		$(srcdir)/Objects/mimalloc/prim/wasi/prim.c

Objects/mimalloc/page.o: $(srcdir)/Objects/mimalloc/page-queue.c


# Regenerate various files from Python/bytecodes.c
# Pass CASESFLAG=-l to insert #line directives in the output

.PHONY: regen-cases
regen-cases: \
        regen-opcode-ids regen-opcode-targets regen-uop-ids regen-opcode-metadata-py \
		regen-generated-cases regen-executor-cases regen-optimizer-cases \
		regen-opcode-metadata regen-uop-metadata

.PHONY: regen-opcode-ids
regen-opcode-ids:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/cases_generator/opcode_id_generator.py \
	    -o $(srcdir)/Include/opcode_ids.h.new $(srcdir)/Python/bytecodes.c
	$(UPDATE_FILE) $(srcdir)/Include/opcode_ids.h $(srcdir)/Include/opcode_ids.h.new

.PHONY: regen-opcode-targets
regen-opcode-targets:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/cases_generator/target_generator.py \
	    -o $(srcdir)/Python/opcode_targets.h.new $(srcdir)/Python/bytecodes.c
	$(UPDATE_FILE) $(srcdir)/Python/opcode_targets.h $(srcdir)/Python/opcode_targets.h.new

.PHONY: regen-uop-ids
regen-uop-ids:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/cases_generator/uop_id_generator.py \
	    -o $(srcdir)/Include/internal/pycore_uop_ids.h.new $(srcdir)/Python/bytecodes.c
	$(UPDATE_FILE) $(srcdir)/Include/internal/pycore_uop_ids.h $(srcdir)/Include/internal/pycore_uop_ids.h.new

.PHONY: regen-opcode-metadata-py
regen-opcode-metadata-py:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/cases_generator/py_metadata_generator.py \
	    -o $(srcdir)/Lib/_opcode_metadata.py.new $(srcdir)/Python/bytecodes.c
	$(UPDATE_FILE) $(srcdir)/Lib/_opcode_metadata.py $(srcdir)/Lib/_opcode_metadata.py.new

.PHONY: regen-generated-cases
regen-generated-cases:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/cases_generator/tier1_generator.py \
	    -o $(srcdir)/Python/generated_cases.c.h.new $(srcdir)/Python/bytecodes.c
	$(UPDATE_FILE) $(srcdir)/Python/generated_cases.c.h $(srcdir)/Python/generated_cases.c.h.new

.PHONY: regen-executor-cases
regen-executor-cases:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/cases_generator/tier2_generator.py \
	    -o $(srcdir)/Python/executor_cases.c.h.new $(srcdir)/Python/bytecodes.c
	$(UPDATE_FILE) $(srcdir)/Python/executor_cases.c.h $(srcdir)/Python/executor_cases.c.h.new

.PHONY: regen-optimizer-cases
regen-optimizer-cases:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/cases_generator/optimizer_generator.py \
	    -o $(srcdir)/Python/optimizer_cases.c.h.new \
	    $(srcdir)/Python/optimizer_bytecodes.c \
	    $(srcdir)/Python/bytecodes.c
	$(UPDATE_FILE) $(srcdir)/Python/optimizer_cases.c.h $(srcdir)/Python/optimizer_cases.c.h.new

.PHONY: regen-opcode-metadata
regen-opcode-metadata:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/cases_generator/opcode_metadata_generator.py \
	    -o $(srcdir)/Include/internal/pycore_opcode_metadata.h.new $(srcdir)/Python/bytecodes.c
	$(UPDATE_FILE) $(srcdir)/Include/internal/pycore_opcode_metadata.h $(srcdir)/Include/internal/pycore_opcode_metadata.h.new

.PHONY: regen-uop-metadata
regen-uop-metadata:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/cases_generator/uop_metadata_generator.py -o \
	    $(srcdir)/Include/internal/pycore_uop_metadata.h.new $(srcdir)/Python/bytecodes.c
	$(UPDATE_FILE) $(srcdir)/Include/internal/pycore_uop_metadata.h $(srcdir)/Include/internal/pycore_uop_metadata.h.new

Python/compile.o Python/assemble.o Python/flowgraph.o Python/instruction_sequence.o: \
                $(srcdir)/Include/internal/pycore_compile.h \
                $(srcdir)/Include/internal/pycore_flowgraph.h \
                $(srcdir)/Include/internal/pycore_instruction_sequence.h \
                $(srcdir)/Include/internal/pycore_opcode_metadata.h \
                $(srcdir)/Include/internal/pycore_opcode_utils.h

Python/ceval.o: \
		$(srcdir)/Python/ceval_macros.h \
		$(srcdir)/Python/condvar.h \
		$(srcdir)/Python/generated_cases.c.h \
		$(srcdir)/Python/executor_cases.c.h \
		$(srcdir)/Python/opcode_targets.h

Python/flowgraph.o: \
		$(srcdir)/Include/internal/pycore_opcode_metadata.h

Python/optimizer.o: \
		$(srcdir)/Python/executor_cases.c.h \
		$(srcdir)/Include/internal/pycore_opcode_metadata.h \
		$(srcdir)/Include/internal/pycore_optimizer.h

Python/optimizer_analysis.o: \
		$(srcdir)/Include/internal/pycore_opcode_metadata.h \
		$(srcdir)/Include/internal/pycore_optimizer.h \
		$(srcdir)/Python/optimizer_cases.c.h

Python/frozen.o: $(FROZEN_FILES_OUT)

# Generate DTrace probe macros, then rename them (PYTHON_ -> PyDTrace_) to
# follow our naming conventions. dtrace(1) uses the output filename to generate
# an include guard, so we can't use a pipeline to transform its output.
Include/pydtrace_probes.h: $(srcdir)/Include/pydtrace.d
	$(MKDIR_P) Include
	$(DTRACE) $(DFLAGS) -o $@ -h -s $<
	: sed in-place edit with POSIX-only tools
	sed 's/PYTHON_/PyDTrace_/' $@ > $@.tmp
	mv $@.tmp $@

Python/ceval.o: $(srcdir)/Include/pydtrace.h
Python/gc.o: $(srcdir)/Include/pydtrace.h
Python/import.o: $(srcdir)/Include/pydtrace.h

Python/pydtrace.o: $(srcdir)/Include/pydtrace.d $(DTRACE_DEPS)
	$(DTRACE) $(DFLAGS) -o $@ -G -s $< $(DTRACE_DEPS)

Objects/typeobject.o: Objects/typeslots.inc

.PHONY: regen-typeslots
regen-typeslots:
	# Regenerate Objects/typeslots.inc from Include/typeslotsh
	# using Objects/typeslots.py
	$(PYTHON_FOR_REGEN) $(srcdir)/Objects/typeslots.py \
		< $(srcdir)/Include/typeslots.h \
		$(srcdir)/Objects/typeslots.inc.new
	$(UPDATE_FILE) $(srcdir)/Objects/typeslots.inc $(srcdir)/Objects/typeslots.inc.new

$(LIBRARY_OBJS) $(MODOBJS) Programs/python.o: $(PYTHON_HEADERS)


######################################################################

TESTOPTS=	$(EXTRATESTOPTS)
TESTPYTHON=	$(RUNSHARED) $(PYTHON_FOR_BUILD) $(TESTPYTHONOPTS)
TESTRUNNER=	$(TESTPYTHON) -m test
TESTTIMEOUT=

# Remove "test_python_*" directories of previous failed test jobs.
# Pass TESTOPTS options because it can contain --tempdir option.
.PHONY: cleantest
cleantest: all
	$(TESTRUNNER) $(TESTOPTS) --cleanup

# Run a basic set of regression tests.
# This excludes some tests that are particularly resource-intensive.
# Similar to buildbottest, but use --fast-ci option, instead of --slow-ci.
.PHONY: test
test: all
	$(TESTRUNNER) --fast-ci --timeout=$(TESTTIMEOUT) $(TESTOPTS)

# Run the test suite for both architectures in a Universal build on OSX.
# Must be run on an Intel box.
.PHONY: testuniversal
testuniversal: all
	@if [ `arch` != 'i386' ]; then \
		echo "This can only be used on OSX/i386" ;\
		exit 1 ;\
	fi
	$(TESTRUNNER) --slow-ci --timeout=$(TESTTIMEOUT) $(TESTOPTS)
	$(RUNSHARED) /usr/libexec/oah/translate \
		./$(BUILDPYTHON) -E -m test -j 0 -u all $(TESTOPTS)

# Run the test suite on the iOS simulator. Must be run on a macOS machine with
# a full Xcode install that has an iPhone SE (3rd edition) simulator available.
# This must be run *after* a `make install` has completed the build. The
# `--with-framework-name` argument *cannot* be used when configuring the build.
XCFOLDER:=iOSTestbed.$(MULTIARCH).$(shell date +%s)
XCRESULT=$(XCFOLDER)/$(MULTIARCH).xcresult
.PHONY: testios
testios:
	@if test "$(MACHDEP)" != "ios"; then \
		echo "Cannot run the iOS testbed for a non-iOS build."; \
		exit 1;\
	fi
	@if test "$(findstring -iphonesimulator,$(MULTIARCH))" != "-iphonesimulator"; then \
		echo "Cannot run the iOS testbed for non-simulator builds."; \
		exit 1;\
	fi
	@if test $(PYTHONFRAMEWORK) != "Python"; then \
		echo "Cannot run the iOS testbed with a non-default framework name."; \
		exit 1;\
	fi
	@if ! test -d $(PYTHONFRAMEWORKPREFIX); then \
		echo "Cannot find a finalized iOS Python.framework. Have you run 'make install' to finalize the framework build?"; \
		exit 1;\
	fi
	# Copy the testbed project into the build folder
	cp -r $(srcdir)/iOS/testbed $(XCFOLDER)
	# Copy the framework from the install location to the testbed project.
	cp -r $(PYTHONFRAMEWORKPREFIX)/* $(XCFOLDER)/Python.xcframework/ios-arm64_x86_64-simulator

	# Run the test suite for the Xcode project, targeting the iOS simulator.
	# If the suite fails, touch a file in the test folder as a marker
	if ! xcodebuild test -project $(XCFOLDER)/iOSTestbed.xcodeproj -scheme "iOSTestbed" -destination "platform=iOS Simulator,name=iPhone SE (3rd Generation)" -resultBundlePath $(XCRESULT) -derivedDataPath $(XCFOLDER)/DerivedData ; then \
	 	touch $(XCFOLDER)/failed; \
	fi

	# Regardless of success or failure, extract and print the test output
	xcrun xcresulttool get --path $(XCRESULT) \
		--id $$( \
			xcrun xcresulttool get --path $(XCRESULT) --format json | \
			$(PYTHON_FOR_BUILD) -c "import sys, json; result = json.load(sys.stdin); print(result['actions']['_values'][0]['actionResult']['logRef']['id']['_value'])" \
		) \
		--format json | \
		$(PYTHON_FOR_BUILD) -c "import sys, json; result = json.load(sys.stdin); print(result['subsections']['_values'][1]['subsections']['_values'][0]['emittedOutput']['_value'])"

	@if test -e $(XCFOLDER)/failed ; then \
		exit 1; \
	fi

# Like test, but using --slow-ci which enables all test resources and use
# longer timeout. Run an optional pybuildbot.identify script to include
# information about the build environment.
.PHONY: buildbottest
buildbottest: all
	-@if which pybuildbot.identify >/dev/null 2>&1; then \
		pybuildbot.identify "CC='$(CC)'" "CXX='$(CXX)'"; \
	fi
	$(TESTRUNNER) --slow-ci --timeout=$(TESTTIMEOUT) $(TESTOPTS)

.PHONY: pythoninfo
pythoninfo: all
		$(RUNSHARED) $(HOSTRUNNER) ./$(BUILDPYTHON) -m test.pythoninfo

QUICKTESTOPTS=	-x test_subprocess test_io \
		test_multibytecodec test_urllib2_localnet test_itertools \
		test_multiprocessing_fork test_multiprocessing_spawn \
		test_multiprocessing_forkserver \
		test_mailbox test_socket test_poll \
		test_select test_zipfile test_concurrent_futures

.PHONY: quicktest
quicktest: all
	$(TESTRUNNER) --fast-ci --timeout=$(TESTTIMEOUT) $(TESTOPTS) $(QUICKTESTOPTS)

# SSL tests
.PHONY: multisslcompile
multisslcompile: all
	$(RUNSHARED) ./$(BUILDPYTHON) $(srcdir)/Tools/ssl/multissltests.py --steps=modules

.PHONY: multissltest
multissltest: all
	$(RUNSHARED) ./$(BUILDPYTHON) $(srcdir)/Tools/ssl/multissltests.py

# All install targets use the "all" target as synchronization point to
# prevent race conditions with PGO builds. PGO builds use recursive make,
# which can lead to two parallel `./python setup.py build` processes that
# step on each others toes.
.PHONY: install
install:  commoninstall bininstall maninstall 
	if test "x$(ENSUREPIP)" != "xno"  ; then \
		case $(ENSUREPIP) in \
			upgrade) ensurepip="--upgrade" ;; \
			install|*) ensurepip="" ;; \
		esac; \
		$(RUNSHARED) $(PYTHON_FOR_BUILD) -m ensurepip \
			$$ensurepip --root=$(DESTDIR)/ ; \
	fi

.PHONY: altinstall
altinstall: commoninstall
	if test "x$(ENSUREPIP)" != "xno"  ; then \
		case $(ENSUREPIP) in \
			upgrade) ensurepip="--altinstall --upgrade" ;; \
			install|*) ensurepip="--altinstall" ;; \
		esac; \
		$(RUNSHARED) $(PYTHON_FOR_BUILD) -m ensurepip \
			$$ensurepip --root=$(DESTDIR)/ ; \
	fi

.PHONY: commoninstall
commoninstall:  check-clean-src  \
		altbininstall libinstall inclinstall libainstall \
		sharedinstall altmaninstall 

# Install shared libraries enabled by Setup
DESTDIRS=	$(exec_prefix) $(LIBDIR) $(BINLIBDEST) $(DESTSHARED)

.PHONY: sharedinstall
sharedinstall: all
		@for i in $(DESTDIRS); \
		do \
			if test ! -d $(DESTDIR)$$i; then \
				echo "Creating directory $$i"; \
				$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$$i; \
			else    true; \
			fi; \
		done
		@for i in X $(SHAREDMODS); do \
		  if test $$i != X; then \
		    echo $(INSTALL_SHARED) $$i $(DESTSHARED)/`basename $$i`; \
		    $(INSTALL_SHARED) $$i $(DESTDIR)$(DESTSHARED)/`basename $$i`; \
			if test -d "$$i.dSYM"; then \
				echo $(DSYMUTIL_PATH) $(DESTDIR)$(DESTSHARED)/`basename $$i`; \
				$(DSYMUTIL_PATH) $(DESTDIR)$(DESTSHARED)/`basename $$i`; \
			fi; \
		  fi; \
		done

# Install the interpreter with $(VERSION) affixed
# This goes into $(exec_prefix)
.PHONY: altbininstall
altbininstall: $(BUILDPYTHON) 
	@for i in $(BINDIR) $(LIBDIR); \
	do \
		if test ! -d $(DESTDIR)$$i; then \
			echo "Creating directory $$i"; \
			$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$$i; \
		else	true; \
		fi; \
	done
	if test "$(PYTHONFRAMEWORKDIR)" = "no-framework" ; then \
		$(INSTALL_PROGRAM) $(BUILDPYTHON) $(DESTDIR)$(BINDIR)/python$(LDVERSION)$(EXE); \
	else \
		$(INSTALL_PROGRAM) $(STRIPFLAG) Mac/pythonw $(DESTDIR)$(BINDIR)/python$(LDVERSION)$(EXE); \
	fi
	-if test "$(VERSION)" != "$(LDVERSION)"; then \
		if test -f $(DESTDIR)$(BINDIR)/python$(VERSION)$(EXE) -o -h $(DESTDIR)$(BINDIR)/python$(VERSION)$(EXE); \
		then rm -f $(DESTDIR)$(BINDIR)/python$(VERSION)$(EXE); \
		fi; \
		(cd $(DESTDIR)$(BINDIR); $(LN) python$(LDVERSION)$(EXE) python$(VERSION)$(EXE)); \
	fi
	@if test "$(PY_ENABLE_SHARED)" = 1 -o "$(STATIC_LIBPYTHON)" = 1; then \
		if test -f $(LDLIBRARY) && test "$(PYTHONFRAMEWORKDIR)" = "no-framework" ; then \
			if test -n "$(DLLLIBRARY)" ; then \
				$(INSTALL_SHARED) $(DLLLIBRARY) $(DESTDIR)$(BINDIR); \
			else \
				$(INSTALL_SHARED) $(LDLIBRARY) $(DESTDIR)$(LIBDIR)/$(INSTSONAME); \
				if test $(LDLIBRARY) != $(INSTSONAME); then \
					(cd $(DESTDIR)$(LIBDIR); $(LN) -sf $(INSTSONAME) $(LDLIBRARY)) \
				fi \
			fi; \
			if test -n "$(PY3LIBRARY)"; then \
				$(INSTALL_SHARED) $(PY3LIBRARY) $(DESTDIR)$(LIBDIR)/$(PY3LIBRARY); \
			fi; \
		else	true; \
		fi; \
	fi
	if test "x$(LIPO_32BIT_FLAGS)" != "x" ; then \
		rm -f $(DESTDIR)$(BINDIR)/python$(VERSION)-32$(EXE); \
		lipo $(LIPO_32BIT_FLAGS) \
			-output $(DESTDIR)$(BINDIR)/python$(VERSION)-32$(EXE) \
			$(DESTDIR)$(BINDIR)/python$(VERSION)$(EXE); \
	fi
	if test "x$(LIPO_INTEL64_FLAGS)" != "x" ; then \
		rm -f $(DESTDIR)$(BINDIR)/python$(VERSION)-intel64$(EXE); \
		lipo $(LIPO_INTEL64_FLAGS) \
			-output $(DESTDIR)$(BINDIR)/python$(VERSION)-intel64$(EXE) \
			$(DESTDIR)$(BINDIR)/python$(VERSION)$(EXE); \
	fi
	# Install macOS debug information (if available)
	if test -d "$(BUILDPYTHON).dSYM"; then \
		echo $(DSYMUTIL_PATH) $(DESTDIR)$(BINDIR)/python$(LDVERSION)$(EXE); \
		$(DSYMUTIL_PATH) $(DESTDIR)$(BINDIR)/python$(LDVERSION)$(EXE); \
	fi
	if test "$(PYTHONFRAMEWORKDIR)" = "no-framework" ; then \
		if test -d "$(LDLIBRARY).dSYM"; then \
			echo $(DSYMUTIL_PATH) $(DESTDIR)$(LIBDIR)/$(INSTSONAME); \
			$(DSYMUTIL_PATH) $(DESTDIR)$(LIBDIR)/$(INSTSONAME); \
		fi \
	else \
		if test -d "$(LDLIBRARY).dSYM"; then \
			echo $(DSYMUTIL_PATH) $(DESTDIR)$(PYTHONFRAMEWORKPREFIX)/$(INSTSONAME); \
      $(DSYMUTIL_PATH) $(DESTDIR)$(PYTHONFRAMEWORKPREFIX)/$(INSTSONAME); \
		fi \
	fi

.PHONY: bininstall
# We depend on commoninstall here to make sure the installation is already usable
# before we possibly overwrite the global 'python3' symlink to avoid causing
# problems for anything else trying to run 'python3' while we install, particularly
# if we're installing in parallel with -j.
bininstall: commoninstall altbininstall
	if test ! -d $(DESTDIR)$(LIBPC); then \
		echo "Creating directory $(LIBPC)"; \
		$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$(LIBPC); \
	fi
	-if test -f $(DESTDIR)$(BINDIR)/python3$(EXE) -o -h $(DESTDIR)$(BINDIR)/python3$(EXE); \
	then rm -f $(DESTDIR)$(BINDIR)/python3$(EXE); \
	else true; \
	fi
	(cd $(DESTDIR)$(BINDIR); $(LN) -s python$(VERSION)$(EXE) python3$(EXE))
	-if test "$(VERSION)" != "$(LDVERSION)"; then \
		rm -f $(DESTDIR)$(BINDIR)/python$(VERSION)-config; \
		(cd $(DESTDIR)$(BINDIR); $(LN) -s python$(LDVERSION)-config python$(VERSION)-config); \
		rm -f $(DESTDIR)$(LIBPC)/python-$(VERSION).pc; \
		(cd $(DESTDIR)$(LIBPC); $(LN) -s python-$(LDVERSION).pc python-$(VERSION).pc); \
		rm -f $(DESTDIR)$(LIBPC)/python-$(VERSION)-embed.pc; \
		(cd $(DESTDIR)$(LIBPC); $(LN) -s python-$(LDVERSION)-embed.pc python-$(VERSION)-embed.pc); \
	fi
	-rm -f $(DESTDIR)$(BINDIR)/python3-config
	(cd $(DESTDIR)$(BINDIR); $(LN) -s python$(VERSION)-config python3-config)
	-rm -f $(DESTDIR)$(LIBPC)/python3.pc
	(cd $(DESTDIR)$(LIBPC); $(LN) -s python-$(VERSION).pc python3.pc)
	-rm -f $(DESTDIR)$(LIBPC)/python3-embed.pc
	(cd $(DESTDIR)$(LIBPC); $(LN) -s python-$(VERSION)-embed.pc python3-embed.pc)
	-rm -f $(DESTDIR)$(BINDIR)/idle3
	(cd $(DESTDIR)$(BINDIR); $(LN) -s idle$(VERSION) idle3)
	-rm -f $(DESTDIR)$(BINDIR)/pydoc3
	(cd $(DESTDIR)$(BINDIR); $(LN) -s pydoc$(VERSION) pydoc3)
	if test "x$(LIPO_32BIT_FLAGS)" != "x" ; then \
		rm -f $(DESTDIR)$(BINDIR)/python3-32$(EXE); \
		(cd $(DESTDIR)$(BINDIR); $(LN) -s python$(VERSION)-32$(EXE) python3-32$(EXE)) \
	fi
	if test "x$(LIPO_INTEL64_FLAGS)" != "x" ; then \
		rm -f $(DESTDIR)$(BINDIR)/python3-intel64$(EXE); \
		(cd $(DESTDIR)$(BINDIR); $(LN) -s python$(VERSION)-intel64$(EXE) python3-intel64$(EXE)) \
	fi

# Install the versioned manual page
.PHONY: altmaninstall
altmaninstall:
	@for i in $(MANDIR) $(MANDIR)/man1; \
	do \
		if test ! -d $(DESTDIR)$$i; then \
			echo "Creating directory $$i"; \
			$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$$i; \
		else	true; \
		fi; \
	done
	$(INSTALL_DATA) $(srcdir)/Misc/python.man \
		$(DESTDIR)$(MANDIR)/man1/python$(VERSION).1

# Install the unversioned manual page
.PHONY: maninstall
maninstall:	altmaninstall
	-rm -f $(DESTDIR)$(MANDIR)/man1/python3.1
	(cd $(DESTDIR)$(MANDIR)/man1; $(LN) -s python$(VERSION).1 python3.1)

# Install the library
XMLLIBSUBDIRS=  xml xml/dom xml/etree xml/parsers xml/sax
LIBSUBDIRS=	asyncio \
		collections \
		concurrent concurrent/futures \
		csv \
		ctypes ctypes/macholib \
		curses \
		dbm \
		email email/mime \
		encodings \
		ensurepip ensurepip/_bundled \
		html \
		http \
		idlelib idlelib/Icons \
		importlib importlib/resources importlib/metadata \
		json \
		logging \
		multiprocessing multiprocessing/dummy \
		pathlib \
		pydoc_data \
		re \
		site-packages \
		sqlite3 \
		sysconfig \
		tkinter \
		tomllib \
		turtledemo \
		unittest \
		urllib \
		venv venv/scripts venv/scripts/common venv/scripts/posix \
		wsgiref \
		$(XMLLIBSUBDIRS) \
		xmlrpc \
		zipfile zipfile/_path \
		zoneinfo \
		_pyrepl \
		__phello__
TESTSUBDIRS=	idlelib/idle_test \
		test \
		test/test_ast \
		test/archivetestdata \
		test/audiodata \
		test/certdata \
		test/certdata/capath \
		test/cjkencodings \
		test/configdata \
		test/crashers \
		test/data \
		test/decimaltestdata \
		test/dtracedata \
		test/encoded_modules \
		test/leakers \
		test/libregrtest \
		test/mathdata \
		test/regrtestdata \
		test/regrtestdata/import_from_tests \
		test/regrtestdata/import_from_tests/test_regrtest_b \
		test/subprocessdata \
		test/support \
		test/support/_hypothesis_stubs \
		test/support/interpreters \
		test/test_asyncio \
		test/test_capi \
		test/test_cext \
		test/test_concurrent_futures \
		test/test_cppext \
		test/test_ctypes \
		test/test_dataclasses \
		test/test_doctest \
		test/test_email \
		test/test_email/data \
		test/test_free_threading \
		test/test_future_stmt \
		test/test_gdb \
		test/test_import \
		test/test_import/data \
		test/test_import/data/circular_imports \
		test/test_import/data/circular_imports/subpkg \
		test/test_import/data/circular_imports/subpkg2 \
		test/test_import/data/circular_imports/subpkg2/parent \
		test/test_import/data/package \
		test/test_import/data/package2 \
		test/test_import/data/package3 \
		test/test_import/data/package4 \
		test/test_import/data/unwritable \
		test/test_importlib \
		test/test_importlib/builtin \
		test/test_importlib/extension \
		test/test_importlib/frozen \
		test/test_importlib/import_ \
		test/test_importlib/metadata \
		test/test_importlib/metadata/data \
		test/test_importlib/metadata/data/sources \
		test/test_importlib/metadata/data/sources/example \
		test/test_importlib/metadata/data/sources/example/example \
		test/test_importlib/metadata/data/sources/example2 \
		test/test_importlib/metadata/data/sources/example2/example2 \
		test/test_importlib/namespace_pkgs \
		test/test_importlib/namespace_pkgs/both_portions \
		test/test_importlib/namespace_pkgs/both_portions/foo \
		test/test_importlib/namespace_pkgs/module_and_namespace_package \
		test/test_importlib/namespace_pkgs/module_and_namespace_package/a_test \
		test/test_importlib/namespace_pkgs/not_a_namespace_pkg \
		test/test_importlib/namespace_pkgs/not_a_namespace_pkg/foo \
		test/test_importlib/namespace_pkgs/portion1 \
		test/test_importlib/namespace_pkgs/portion1/foo \
		test/test_importlib/namespace_pkgs/portion2 \
		test/test_importlib/namespace_pkgs/portion2/foo \
		test/test_importlib/namespace_pkgs/project1 \
		test/test_importlib/namespace_pkgs/project1/parent \
		test/test_importlib/namespace_pkgs/project1/parent/child \
		test/test_importlib/namespace_pkgs/project2 \
		test/test_importlib/namespace_pkgs/project2/parent \
		test/test_importlib/namespace_pkgs/project2/parent/child \
		test/test_importlib/namespace_pkgs/project3 \
		test/test_importlib/namespace_pkgs/project3/parent \
		test/test_importlib/namespace_pkgs/project3/parent/child \
		test/test_importlib/partial \
		test/test_importlib/resources \
		test/test_importlib/source \
		test/test_inspect \
		test/test_interpreters \
		test/test_json \
		test/test_module \
		test/test_multiprocessing_fork \
		test/test_multiprocessing_forkserver \
		test/test_multiprocessing_spawn \
		test/test_pathlib \
		test/test_peg_generator \
		test/test_pydoc \
		test/test_pyrepl \
		test/test_sqlite3 \
		test/test_tkinter \
		test/test_tomllib \
		test/test_tomllib/data \
		test/test_tomllib/data/invalid \
		test/test_tomllib/data/invalid/array \
		test/test_tomllib/data/invalid/array-of-tables \
		test/test_tomllib/data/invalid/boolean \
		test/test_tomllib/data/invalid/dates-and-times \
		test/test_tomllib/data/invalid/dotted-keys \
		test/test_tomllib/data/invalid/inline-table \
		test/test_tomllib/data/invalid/keys-and-vals \
		test/test_tomllib/data/invalid/literal-str \
		test/test_tomllib/data/invalid/multiline-basic-str \
		test/test_tomllib/data/invalid/multiline-literal-str \
		test/test_tomllib/data/invalid/table \
		test/test_tomllib/data/valid \
		test/test_tomllib/data/valid/array \
		test/test_tomllib/data/valid/dates-and-times \
		test/test_tomllib/data/valid/multiline-basic-str \
		test/test_tools \
		test/test_ttk \
		test/test_unittest \
		test/test_unittest/testmock \
		test/test_warnings \
		test/test_warnings/data \
		test/test_zipfile \
		test/test_zipfile/_path \
		test/test_zoneinfo \
		test/test_zoneinfo/data \
		test/tkinterdata \
		test/tokenizedata \
		test/tracedmodules \
		test/typinganndata \
		test/wheeldata \
		test/xmltestdata \
		test/xmltestdata/c14n-20 \
		test/zipimport_data

COMPILEALL_OPTS=-j0

TEST_MODULES=yes

.PHONY: libinstall
libinstall:	all $(srcdir)/Modules/xxmodule.c
	@for i in $(SCRIPTDIR) $(LIBDEST); \
	do \
		if test ! -d $(DESTDIR)$$i; then \
			echo "Creating directory $$i"; \
			$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$$i; \
		else	true; \
		fi; \
	done
	@if test "$(TEST_MODULES)" = yes; then \
		subdirs="$(LIBSUBDIRS) $(TESTSUBDIRS)"; \
	else \
		subdirs="$(LIBSUBDIRS)"; \
	fi; \
	for d in $$subdirs; \
	do \
		a=$(srcdir)/Lib/$$d; \
		if test ! -d $$a; then continue; else true; fi; \
		b=$(LIBDEST)/$$d; \
		if test ! -d $(DESTDIR)$$b; then \
			echo "Creating directory $$b"; \
			$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$$b; \
		else	true; \
		fi; \
	done
	@for i in $(srcdir)/Lib/*.py; \
	do \
		if test -x $$i; then \
			$(INSTALL_SCRIPT) $$i $(DESTDIR)$(LIBDEST); \
			echo $(INSTALL_SCRIPT) $$i $(LIBDEST); \
		else \
			$(INSTALL_DATA) $$i $(DESTDIR)$(LIBDEST); \
			echo $(INSTALL_DATA) $$i $(LIBDEST); \
		fi; \
	done
	@if test "$(TEST_MODULES)" = yes; then \
		subdirs="$(LIBSUBDIRS) $(TESTSUBDIRS)"; \
	else \
		subdirs="$(LIBSUBDIRS)"; \
	fi; \
	for d in $$subdirs; \
	do \
		a=$(srcdir)/Lib/$$d; \
		if test ! -d $$a; then continue; else true; fi; \
		if test `ls $$a | wc -l` -lt 1; then continue; fi; \
		b=$(LIBDEST)/$$d; \
		for i in $$a/*; \
		do \
			case $$i in \
			*CVS) ;; \
			*.py[co]) ;; \
			*.orig) ;; \
			*~) ;; \
			*) \
				if test -d $$i; then continue; fi; \
				if test -x $$i; then \
				    echo $(INSTALL_SCRIPT) $$i $$b; \
				    $(INSTALL_SCRIPT) $$i $(DESTDIR)$$b; \
				else \
				    echo $(INSTALL_DATA) $$i $$b; \
				    $(INSTALL_DATA) $$i $(DESTDIR)$$b; \
				fi;; \
			esac; \
		done; \
	done
	$(INSTALL_DATA) `cat pybuilddir.txt`/_sysconfigdata_$(ABIFLAGS)_$(MACHDEP)_$(MULTIARCH).py \
		$(DESTDIR)$(LIBDEST); \
	$(INSTALL_DATA) $(srcdir)/LICENSE $(DESTDIR)$(LIBDEST)/LICENSE.txt
	@ # If app store compliance has been configured, apply the patch to the
	@ # installed library code. The patch has been previously validated against
	@ # the original source tree, so we can ignore any errors that are raised
	@ # due to files that are missing because of --disable-test-modules etc.
	@if [ "$(APP_STORE_COMPLIANCE_PATCH)" != "" ]; then \
		echo "Applying app store compliance patch"; \
		patch --force --reject-file "$(abs_builddir)/app-store-compliance.rej" --strip 2 --directory "$(DESTDIR)$(LIBDEST)" --input "$(abs_srcdir)/$(APP_STORE_COMPLIANCE_PATCH)" || true ; \
	fi
	@ # Build PYC files for the 3 optimization levels (0, 1, 2)
	-PYTHONPATH=$(DESTDIR)$(LIBDEST) $(RUNSHARED) \
		$(PYTHON_FOR_BUILD) -Wi $(DESTDIR)$(LIBDEST)/compileall.py \
		-o 0 -o 1 -o 2 $(COMPILEALL_OPTS) -d $(LIBDEST) -f \
		-x 'bad_coding|badsyntax|site-packages' \
		$(DESTDIR)$(LIBDEST)
	-PYTHONPATH=$(DESTDIR)$(LIBDEST) $(RUNSHARED) \
		$(PYTHON_FOR_BUILD) -Wi $(DESTDIR)$(LIBDEST)/compileall.py \
		-o 0 -o 1 -o 2 $(COMPILEALL_OPTS) -d $(LIBDEST)/site-packages -f \
		-x badsyntax $(DESTDIR)$(LIBDEST)/site-packages

# bpo-21536: Misc/python-config.sh is generated in the build directory
# from $(srcdir)Misc/python-config.sh.in.
python-config: $(srcdir)/Misc/python-config.in Misc/python-config.sh
	@ # Substitution happens here, as the completely-expanded BINDIR
	@ # is not available in configure
	sed -e "s,@EXENAME@,$(EXENAME)," < $(srcdir)/Misc/python-config.in >python-config.py
	@ # Replace makefile compat. variable references with shell script compat. ones; $(VAR) -> ${VAR}
	LC_ALL=C sed -e 's,\$$(\([A-Za-z0-9_]*\)),\$$\{\1\},g' < Misc/python-config.sh >python-config
	@ # On Darwin, always use the python version of the script, the shell
	@ # version doesn't use the compiler customizations that are provided
	@ # in python (_osx_support.py).
	@if test `uname -s` = Darwin; then \
		cp python-config.py python-config; \
	fi

# macOS' make seems to ignore a dependency on a
# "$(BUILD_SCRIPTS_DIR): $(MKDIR_P) $@" rule.
BUILD_SCRIPTS_DIR=build/scripts-$(VERSION)
SCRIPT_IDLE=$(BUILD_SCRIPTS_DIR)/idle$(VERSION)
SCRIPT_PYDOC=$(BUILD_SCRIPTS_DIR)/pydoc$(VERSION)

$(SCRIPT_IDLE): $(srcdir)/Tools/scripts/idle3
	@$(MKDIR_P) $(BUILD_SCRIPTS_DIR)
	sed -e "s,/usr/bin/env python3,$(EXENAME)," < $(srcdir)/Tools/scripts/idle3 > $@
	@chmod +x $@

$(SCRIPT_PYDOC): $(srcdir)/Tools/scripts/pydoc3
	@$(MKDIR_P) $(BUILD_SCRIPTS_DIR)
	sed -e "s,/usr/bin/env python3,$(EXENAME)," < $(srcdir)/Tools/scripts/pydoc3 > $@
	@chmod +x $@

.PHONY: scripts
scripts: $(SCRIPT_IDLE) $(SCRIPT_PYDOC) python-config

# Install the include files
INCLDIRSTOMAKE=$(INCLUDEDIR) $(CONFINCLUDEDIR) $(INCLUDEPY) $(CONFINCLUDEPY)

.PHONY: inclinstall
inclinstall:
	@for i in $(INCLDIRSTOMAKE); \
	do \
		if test ! -d $(DESTDIR)$$i; then \
			echo "Creating directory $$i"; \
			$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$$i; \
		else	true; \
		fi; \
	done
	@if test ! -d $(DESTDIR)$(INCLUDEPY)/cpython; then \
		echo "Creating directory $(DESTDIR)$(INCLUDEPY)/cpython"; \
		$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$(INCLUDEPY)/cpython; \
	else	true; \
	fi
	@if test ! -d $(DESTDIR)$(INCLUDEPY)/internal; then \
		echo "Creating directory $(DESTDIR)$(INCLUDEPY)/internal"; \
		$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$(INCLUDEPY)/internal; \
	else	true; \
	fi
	@if test "$(INSTALL_MIMALLOC)" = "yes"; then \
		if test ! -d $(DESTDIR)$(INCLUDEPY)/internal/mimalloc/mimalloc; then \
			echo "Creating directory $(DESTDIR)$(INCLUDEPY)/internal/mimalloc/mimalloc"; \
			$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$(INCLUDEPY)/internal/mimalloc/mimalloc; \
		fi; \
	fi
	@for i in $(srcdir)/Include/*.h; \
	do \
		echo $(INSTALL_DATA) $$i $(INCLUDEPY); \
		$(INSTALL_DATA) $$i $(DESTDIR)$(INCLUDEPY); \
	done
	@for i in $(srcdir)/Include/cpython/*.h; \
	do \
		echo $(INSTALL_DATA) $$i $(INCLUDEPY)/cpython; \
		$(INSTALL_DATA) $$i $(DESTDIR)$(INCLUDEPY)/cpython; \
	done
	@for i in $(srcdir)/Include/internal/*.h; \
	do \
		echo $(INSTALL_DATA) $$i $(INCLUDEPY)/internal; \
		$(INSTALL_DATA) $$i $(DESTDIR)$(INCLUDEPY)/internal; \
	done
	@if test "$(INSTALL_MIMALLOC)" = "yes"; then \
		echo $(INSTALL_DATA) $(srcdir)/Include/internal/mimalloc/mimalloc.h $(DESTDIR)$(INCLUDEPY)/internal/mimalloc/mimalloc.h; \
		$(INSTALL_DATA) $(srcdir)/Include/internal/mimalloc/mimalloc.h $(DESTDIR)$(INCLUDEPY)/internal/mimalloc/mimalloc.h; \
		for i in $(srcdir)/Include/internal/mimalloc/mimalloc/*.h; \
		do \
			echo $(INSTALL_DATA) $$i $(INCLUDEPY)/internal/mimalloc/mimalloc; \
			$(INSTALL_DATA) $$i $(DESTDIR)$(INCLUDEPY)/internal/mimalloc/mimalloc; \
		done; \
	fi
	echo $(INSTALL_DATA) pyconfig.h $(DESTDIR)$(CONFINCLUDEPY)/pyconfig.h
	$(INSTALL_DATA) pyconfig.h $(DESTDIR)$(CONFINCLUDEPY)/pyconfig.h

# Install the library and miscellaneous stuff needed for extending/embedding
# This goes into $(exec_prefix)
LIBPL=		$(prefix)/lib/python3.13/config-$(VERSION)$(ABIFLAGS)-x86_64-linux-gnu

# pkgconfig directory
LIBPC=		$(LIBDIR)/pkgconfig

.PHONY: libainstall
libainstall: all scripts
	@for i in $(LIBDIR) $(LIBPL) $(LIBPC) $(BINDIR); \
	do \
		if test ! -d $(DESTDIR)$$i; then \
			echo "Creating directory $$i"; \
			$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$$i; \
		else	true; \
		fi; \
	done
	@if test "$(STATIC_LIBPYTHON)" = 1; then \
		if test -d $(LIBRARY); then :; else \
			if test "$(PYTHONFRAMEWORKDIR)" = no-framework; then \
				if test "$(SHLIB_SUFFIX)" = .dll; then \
					$(INSTALL_DATA) $(LDLIBRARY) $(DESTDIR)$(LIBPL) ; \
				else \
					$(INSTALL_DATA) $(LIBRARY) $(DESTDIR)$(LIBPL)/$(LIBRARY) ; \
				fi; \
			else \
				echo Skip install of $(LIBRARY) - use make frameworkinstall; \
			fi; \
		fi; \
		$(INSTALL_DATA) Programs/python.o $(DESTDIR)$(LIBPL)/python.o; \
	fi
	$(INSTALL_DATA) Modules/config.c $(DESTDIR)$(LIBPL)/config.c
	$(INSTALL_DATA) $(srcdir)/Modules/config.c.in $(DESTDIR)$(LIBPL)/config.c.in
	$(INSTALL_DATA) Makefile $(DESTDIR)$(LIBPL)/Makefile
	$(INSTALL_DATA) $(srcdir)/Modules/Setup $(DESTDIR)$(LIBPL)/Setup
	$(INSTALL_DATA) Modules/Setup.bootstrap $(DESTDIR)$(LIBPL)/Setup.bootstrap
	$(INSTALL_DATA) Modules/Setup.stdlib $(DESTDIR)$(LIBPL)/Setup.stdlib
	$(INSTALL_DATA) Modules/Setup.local $(DESTDIR)$(LIBPL)/Setup.local
	$(INSTALL_DATA) Misc/python.pc $(DESTDIR)$(LIBPC)/python-$(LDVERSION).pc
	$(INSTALL_DATA) Misc/python-embed.pc $(DESTDIR)$(LIBPC)/python-$(LDVERSION)-embed.pc
	$(INSTALL_SCRIPT) $(srcdir)/Modules/makesetup $(DESTDIR)$(LIBPL)/makesetup
	$(INSTALL_SCRIPT) $(srcdir)/install-sh $(DESTDIR)$(LIBPL)/install-sh
	$(INSTALL_SCRIPT) python-config.py $(DESTDIR)$(LIBPL)/python-config.py
	$(INSTALL_SCRIPT) python-config $(DESTDIR)$(BINDIR)/python$(LDVERSION)-config
	$(INSTALL_SCRIPT) $(SCRIPT_IDLE) $(DESTDIR)$(BINDIR)/idle$(VERSION)
	$(INSTALL_SCRIPT) $(SCRIPT_PYDOC) $(DESTDIR)$(BINDIR)/pydoc$(VERSION)
	@if [ -s Modules/python.exp -a \
		"`echo $(MACHDEP) | sed 's/^\(...\).*/\1/'`" = "aix" ]; then \
		echo; echo "Installing support files for building shared extension modules on AIX:"; \
		$(INSTALL_DATA) Modules/python.exp		\
				$(DESTDIR)$(LIBPL)/python.exp;		\
		echo; echo "$(LIBPL)/python.exp";		\
		$(INSTALL_SCRIPT) $(srcdir)/Modules/makexp_aix	\
				$(DESTDIR)$(LIBPL)/makexp_aix;		\
		echo "$(LIBPL)/makexp_aix";			\
		$(INSTALL_SCRIPT) Modules/ld_so_aix	\
				$(DESTDIR)$(LIBPL)/ld_so_aix;		\
		echo "$(LIBPL)/ld_so_aix";			\
		echo; echo "See Misc/README.AIX for details.";	\
	else true; \
	fi

# Here are a couple of targets for MacOSX again, to install a full
# framework-based Python. frameworkinstall installs everything, the
# subtargets install specific parts. Much of the actual work is offloaded to
# the Makefile in Mac
#
#
# This target is here for backward compatibility, previous versions of Python
# hadn't integrated framework installation in the normal install process.
.PHONY: frameworkinstall
frameworkinstall: install

# On install, we re-make the framework
# structure in the install location, /Library/Frameworks/ or the argument to
# --enable-framework. If --enable-framework has been specified then we have
# automatically set prefix to the location deep down in the framework, so we
# only have to cater for the structural bits of the framework.

.PHONY: frameworkinstallframework
frameworkinstallframework:  install frameworkinstallmaclib

# macOS uses a versioned frameworks structure that includes a full install
.PHONY: frameworkinstallversionedstructure
frameworkinstallversionedstructure:	$(LDLIBRARY)
	@if test "$(PYTHONFRAMEWORKDIR)" = no-framework; then \
		echo Not configured with --enable-framework; \
		exit 1; \
	else true; \
	fi
	@for i in $(prefix)/Resources/English.lproj $(prefix)/lib; do\
		if test ! -d $(DESTDIR)$$i; then \
			echo "Creating directory $(DESTDIR)$$i"; \
			$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$$i; \
		else	true; \
		fi; \
	done
	$(LN) -fsn include/python$(LDVERSION) $(DESTDIR)$(prefix)/Headers
	sed 's/%VERSION%/'"`$(RUNSHARED) ./$(BUILDPYTHON) -c 'import platform; print(platform.python_version())'`"'/g' < $(RESSRCDIR)/Info.plist > $(DESTDIR)$(prefix)/Resources/Info.plist
	$(LN) -fsn $(VERSION) $(DESTDIR)$(PYTHONFRAMEWORKINSTALLDIR)/Versions/Current
	$(LN) -fsn Versions/Current/$(PYTHONFRAMEWORK) $(DESTDIR)$(PYTHONFRAMEWORKINSTALLDIR)/$(PYTHONFRAMEWORK)
	$(LN) -fsn Versions/Current/Headers $(DESTDIR)$(PYTHONFRAMEWORKINSTALLDIR)/Headers
	$(LN) -fsn Versions/Current/Resources $(DESTDIR)$(PYTHONFRAMEWORKINSTALLDIR)/Resources
	$(INSTALL_SHARED) $(LDLIBRARY) $(DESTDIR)$(PYTHONFRAMEWORKPREFIX)/$(LDLIBRARY)

# iOS/tvOS/watchOS uses a non-versioned framework with Info.plist in the
# framework root, no .lproj data, and only stub compilation assistance binaries
.PHONY: frameworkinstallunversionedstructure
frameworkinstallunversionedstructure:	$(LDLIBRARY)
	@if test "$(PYTHONFRAMEWORKDIR)" = no-framework; then \
		echo Not configured with --enable-framework; \
		exit 1; \
	else true; \
	fi
	if test -d $(DESTDIR)$(PYTHONFRAMEWORKPREFIX)/include; then \
		echo "Clearing stale header symlink directory"; \
		rm -rf $(DESTDIR)$(PYTHONFRAMEWORKPREFIX)/include; \
	fi
	$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$(PYTHONFRAMEWORKINSTALLDIR)
	sed 's/%VERSION%/'"`$(RUNSHARED) $(PYTHON_FOR_BUILD) -c 'import platform; print(platform.python_version())'`"'/g' < $(RESSRCDIR)/Info.plist > $(DESTDIR)$(PYTHONFRAMEWORKINSTALLDIR)/Info.plist
	$(INSTALL_SHARED) $(LDLIBRARY) $(DESTDIR)$(PYTHONFRAMEWORKPREFIX)/$(LDLIBRARY)
	$(INSTALL) -d -m $(DIRMODE) $(DESTDIR)$(BINDIR)
	for file in $(srcdir)/$(RESSRCDIR)/bin/* ; do \
		$(INSTALL) -m $(EXEMODE) $$file $(DESTDIR)$(BINDIR); \
	done

# This installs Mac/Lib into the framework
# Install a number of symlinks to keep software that expects a normal unix
# install (which includes python-config) happy.
.PHONY: frameworkinstallmaclib
frameworkinstallmaclib:
	$(LN) -fs "../../../$(PYTHONFRAMEWORK)" "$(DESTDIR)$(LIBPL)/libpython$(LDVERSION).a"
	$(LN) -fs "../../../$(PYTHONFRAMEWORK)" "$(DESTDIR)$(LIBPL)/libpython$(LDVERSION).dylib"
	$(LN) -fs "../../../$(PYTHONFRAMEWORK)" "$(DESTDIR)$(LIBPL)/libpython$(VERSION).a"
	$(LN) -fs "../../../$(PYTHONFRAMEWORK)" "$(DESTDIR)$(LIBPL)/libpython$(VERSION).dylib"
	$(LN) -fs "../$(PYTHONFRAMEWORK)" "$(DESTDIR)$(prefix)/lib/libpython$(LDVERSION).dylib"
	$(LN) -fs "../$(PYTHONFRAMEWORK)" "$(DESTDIR)$(prefix)/lib/libpython$(VERSION).dylib"

# This installs the IDE, the Launcher and other apps into /Applications
.PHONY: frameworkinstallapps
frameworkinstallapps:
	cd Mac && $(MAKE) installapps DESTDIR="$(DESTDIR)"

# Build the bootstrap executable that will spawn the interpreter inside
# an app bundle within the framework.  This allows the interpreter to
# run OS X GUI APIs.
.PHONY: frameworkpythonw
frameworkpythonw:
	cd Mac && $(MAKE) pythonw

# This installs the python* and other bin symlinks in $prefix/bin or in
# a bin directory relative to the framework root
.PHONY: frameworkinstallunixtools
frameworkinstallunixtools:
	cd Mac && $(MAKE) installunixtools DESTDIR="$(DESTDIR)"

.PHONY: frameworkaltinstallunixtools
frameworkaltinstallunixtools:
	cd Mac && $(MAKE) altinstallunixtools DESTDIR="$(DESTDIR)"

# This installs the Tools into the applications directory.
# It is not part of a normal frameworkinstall
.PHONY: frameworkinstallextras
frameworkinstallextras:
	cd Mac && $(MAKE) installextras DESTDIR="$(DESTDIR)"

# On iOS, bin/lib can't live inside the framework; include needs to be called
# "Headers", but *must* be in the framework, and *not* include the `python3.X`
# subdirectory. The install has put these folders in the same folder as
# Python.framework; Move the headers to their final framework-compatible home.
.PHONY: frameworkinstallmobileheaders
frameworkinstallmobileheaders: frameworkinstallunversionedstructure inclinstall
	if test -d $(DESTDIR)$(PYTHONFRAMEWORKINSTALLDIR)/Headers; then \
		echo "Removing old framework headers"; \
		rm -rf $(DESTDIR)$(PYTHONFRAMEWORKINSTALLDIR)/Headers; \
	fi
	mv "$(DESTDIR)$(PYTHONFRAMEWORKPREFIX)/include/python$(LDVERSION)" "$(DESTDIR)$(PYTHONFRAMEWORKINSTALLDIR)/Headers"
	$(LN) -fs "../$(PYTHONFRAMEWORKDIR)/Headers" "$(DESTDIR)$(PYTHONFRAMEWORKPREFIX)/include/python$(LDVERSION)"

# Build the toplevel Makefile
Makefile.pre: $(srcdir)/Makefile.pre.in config.status
	CONFIG_FILES=Makefile.pre CONFIG_HEADERS= ./config.status
	$(MAKE) -f Makefile.pre Makefile

# Run the configure script.
config.status:	$(srcdir)/configure
	$(srcdir)/configure $(CONFIG_ARGS)

.PRECIOUS: config.status $(BUILDPYTHON) Makefile Makefile.pre

Python/asm_trampoline.o: $(srcdir)/Python/asm_trampoline.S
	$(CC) -c $(PY_CORE_CFLAGS) -o $@ $<


JIT_DEPS = \
		$(srcdir)/Tools/jit/*.c \
		$(srcdir)/Tools/jit/*.py \
		$(srcdir)/Python/executor_cases.c.h \
		pyconfig.h

jit_stencils.h: $(JIT_DEPS)
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/jit/build.py x86_64-pc-linux-gnu --debug

Python/jit.o: $(srcdir)/Python/jit.c jit_stencils.h
	$(CC) -c $(PY_CORE_CFLAGS) -o $@ $<

.PHONY: regen-jit
regen-jit:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/jit/build.py x86_64-pc-linux-gnu --debug

# Some make's put the object file in the current directory
.c.o:
	$(CC) -c $(PY_CORE_CFLAGS) -o $@ $<

# bpo-30104: dtoa.c uses union to cast double to unsigned long[2]. clang 4.0
# with -O2 or higher and strict aliasing miscompiles the ratio() function
# causing rounding issues. Compile dtoa.c using -fno-strict-aliasing on clang.
# https://bugs.llvm.org//show_bug.cgi?id=31928
Python/dtoa.o: Python/dtoa.c
	$(CC) -c $(PY_CORE_CFLAGS) $(CFLAGS_ALIASING) -o $@ $<

# Run reindent on the library
.PHONY: reindent
reindent:
	./$(BUILDPYTHON) $(srcdir)/Tools/patchcheck/reindent.py -r $(srcdir)/Lib

# Rerun configure with the same options as it was run last time,
# provided the config.status script exists
.PHONY: recheck
recheck:
	./config.status --recheck
	./config.status

# Regenerate configure and pyconfig.h.in
.PHONY: autoconf
autoconf:
	(cd $(srcdir); autoreconf -ivf -Werror)

.PHONY: regen-configure
regen-configure:
	$(srcdir)/Tools/build/regen-configure.sh

.PHONY: regen-sbom
regen-sbom:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/build/generate_sbom.py

# Create a tags file for vi
tags::
	ctags -w $(srcdir)/Include/*.h $(srcdir)/Include/cpython/*.h $(srcdir)/Include/internal/*.h
	for i in $(SRCDIRS); do ctags -f tags -w -a $(srcdir)/$$i/*.[ch]; done
	ctags -f tags -w -a $(srcdir)/Modules/_ctypes/*.[ch]
	find $(srcdir)/Lib -type f -name "*.py" -not -name "test_*.py" -not -path "*/test/*" -not -path "*/tests/*" -not -path "*/*_test/*" | ctags -f tags -w -a -L -
	LC_ALL=C sort -o tags tags

# Create a tags file for GNU Emacs
TAGS::
	cd $(srcdir); \
	etags Include/*.h Include/cpython/*.h Include/internal/*.h; \
	for i in $(SRCDIRS); do etags -a $$i/*.[ch]; done
	etags -a $(srcdir)/Modules/_ctypes/*.[ch]
	find $(srcdir)/Lib -type f -name "*.py" -not -name "test_*.py" -not -path "*/test/*" -not -path "*/tests/*" -not -path "*/*_test/*" | etags - -a

# Sanitation targets -- clean leaves libraries, executables and tags
# files, which clobber removes as well
.PHONY: pycremoval
pycremoval:
	-find $(srcdir) -depth -name '__pycache__' -exec rm -rf {} ';'
	-find $(srcdir) -name '*.py[co]' -exec rm -f {} ';'

.PHONY: rmtestturds
rmtestturds:
	-rm -f *BAD *GOOD *SKIPPED
	-rm -rf OUT
	-rm -f *.TXT
	-rm -f *.txt
	-rm -f gb-18030-2000.xml

.PHONY: docclean
docclean:
	$(MAKE) -C $(srcdir)/Doc clean

# like the 'clean' target but retain the profile guided optimization (PGO)
# data.  The PGO data is only valid if source code remains unchanged.
.PHONY: clean-retain-profile
clean-retain-profile: pycremoval
	find . -name '*.[oa]' -exec rm -f {} ';'
	find . -name '*.s[ol]' -exec rm -f {} ';'
	find . -name '*.so.[0-9]*.[0-9]*' -exec rm -f {} ';'
	find . -name '*.lto' -exec rm -f {} ';'
	find . -name '*.wasm' -exec rm -f {} ';'
	find . -name '*.lst' -exec rm -f {} ';'
	find build -name 'fficonfig.h' -exec rm -f {} ';' || true
	find build -name '*.py' -exec rm -f {} ';' || true
	find build -name '*.py[co]' -exec rm -f {} ';' || true
	-rm -f pybuilddir.txt
	-rm -f _bootstrap_python
	-rm -f python.html python*.js python.data python*.symbols python*.map
	-rm -f $(WASM_STDLIB)
	-rm -f Programs/_testembed Programs/_freeze_module
	-rm -rf Python/deepfreeze
	-rm -f Python/frozen_modules/*.h
	-rm -f Python/frozen_modules/MANIFEST
	-rm -f jit_stencils.h
	-find build -type f -a ! -name '*.gc??' -exec rm -f {} ';'
	-rm -f Include/pydtrace_probes.h
	-rm -f profile-gen-stamp
	-rm -rf iOS/testbed/Python.xcframework/ios-*/bin
	-rm -rf iOS/testbed/Python.xcframework/ios-*/lib
	-rm -rf iOS/testbed/Python.xcframework/ios-*/include
	-rm -rf iOS/testbed/Python.xcframework/ios-*/Python.framework

.PHONY: profile-removal
profile-removal:
	find . -name '*.gc??' -exec rm -f {} ';'
	find . -name '*.profclang?' -exec rm -f {} ';'
	find . -name '*.dyn' -exec rm -f {} ';'
	rm -f $(COVERAGE_INFO)
	rm -rf $(COVERAGE_REPORT)
	rm -f profile-run-stamp
	rm -f profile-bolt-stamp

.PHONY: clean
clean: clean-retain-profile clean-bolt
	@if test profile-opt = profile-opt -o profile-opt = bolt-opt; then \
		rm -f profile-gen-stamp profile-clean-stamp; \
		$(MAKE) profile-removal; \
	fi

.PHONY: clobber
clobber: clean
	-rm -f $(BUILDPYTHON) $(LIBRARY) $(LDLIBRARY) $(DLLLIBRARY) \
		tags TAGS \
		config.cache config.log pyconfig.h Modules/config.c
	-rm -rf build platform
	-rm -rf $(PYTHONFRAMEWORKDIR)
	-rm -rf iOS/Frameworks
	-rm -rf iOSTestbed.*
	-rm -f python-config.py python-config
	-rm -rf cross-build

# Make things extra clean, before making a distribution:
# remove all generated files, even Makefile[.pre]
# Keep configure and Python-ast.[ch], it's possible they can't be generated
.PHONY: distclean
distclean: clobber docclean
	for file in $(srcdir)/Lib/test/data/* ; do \
	    if test "$$file" != "$(srcdir)/Lib/test/data/README"; then rm "$$file"; fi; \
	done
	-rm -f core Makefile Makefile.pre config.status Modules/Setup.local \
	    Modules/Setup.bootstrap Modules/Setup.stdlib \
		Modules/ld_so_aix Modules/python.exp Misc/python.pc \
		Misc/python-embed.pc Misc/python-config.sh
	-rm -f python*-gdb.py
	# Issue #28258: set LC_ALL to avoid issues with Estonian locale.
	# Expansion is performed here by shell (spawned by make) itself before
	# arguments are passed to find. So LC_ALL=C must be set as a separate
	# command.
	LC_ALL=C; find $(srcdir)/[a-zA-Z]* '(' -name '*.fdc' -o -name '*~' \
				     -o -name '[@,#]*' -o -name '*.old' \
				     -o -name '*.orig' -o -name '*.rej' \
				     -o -name '*.bak' ')' \
				     -exec rm -f {} ';'

# Check that all symbols exported by libpython start with "Py" or "_Py"
.PHONY: smelly
smelly: all
	$(RUNSHARED) ./$(BUILDPYTHON) $(srcdir)/Tools/build/smelly.py

# Check if any unsupported C global variables have been added.
.PHONY: check-c-globals
check-c-globals:
	$(PYTHON_FOR_REGEN) $(srcdir)/Tools/c-analyzer/check-c-globals.py \
		--format summary \
		--traceback

# Find files with funny names
.PHONY: funny
funny:
	find $(SUBDIRS) $(SUBDIRSTOO) \
		-type d \
		-o -name '*.[chs]' \
		-o -name '*.py' \
		-o -name '*.pyw' \
		-o -name '*.dat' \
		-o -name '*.el' \
		-o -name '*.fd' \
		-o -name '*.in' \
		-o -name '*.gif' \
		-o -name '*.txt' \
		-o -name '*.xml' \
		-o -name '*.xbm' \
		-o -name '*.xpm' \
		-o -name '*.uue' \
		-o -name '*.decTest' \
		-o -name '*.tmCommand' \
		-o -name '*.tmSnippet' \
		-o -name 'Setup' \
		-o -name 'Setup.*' \
		-o -name README \
		-o -name NEWS \
		-o -name HISTORY \
		-o -name Makefile \
		-o -name ChangeLog \
		-o -name .hgignore \
		-o -name MANIFEST \
		-o -print

# Perform some verification checks on any modified files.
.PHONY: patchcheck
patchcheck: all
	$(RUNSHARED) ./$(BUILDPYTHON) $(srcdir)/Tools/patchcheck/patchcheck.py

.PHONY: check-limited-abi
check-limited-abi: all
	$(RUNSHARED) ./$(BUILDPYTHON) $(srcdir)/Tools/build/stable_abi.py --all $(srcdir)/Misc/stable_abi.toml

.PHONY: update-config
update-config:
	curl -sL -o config.guess 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD'
	curl -sL -o config.sub 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD'
	chmod +x config.guess config.sub

# Dependencies

Python/thread.o:  $(srcdir)/Python/thread_nt.h $(srcdir)/Python/thread_pthread.h $(srcdir)/Python/thread_pthread_stubs.h $(srcdir)/Python/condvar.h

##########################################################################
# Module dependencies and platform-specific files

# force rebuild when header file or module build flavor (static/shared) is changed
MODULE_DEPS_STATIC=Modules/config.c
MODULE_DEPS_SHARED=$(MODULE_DEPS_STATIC) $(EXPORTSYMS)

MODULE__CURSES_DEPS=$(srcdir)/Include/py_curses.h
MODULE__CURSES_PANEL_DEPS=$(srcdir)/Include/py_curses.h
MODULE__DATETIME_DEPS=$(srcdir)/Include/datetime.h
MODULE_CMATH_DEPS=$(srcdir)/Modules/_math.h
MODULE_MATH_DEPS=$(srcdir)/Modules/_math.h
MODULE_PYEXPAT_DEPS=$(LIBEXPAT_HEADERS) $(LIBEXPAT_A)
MODULE_UNICODEDATA_DEPS=$(srcdir)/Modules/unicodedata_db.h $(srcdir)/Modules/unicodename_db.h
MODULE__BLAKE2_DEPS=$(srcdir)/Modules/_blake2/impl/blake2-config.h $(srcdir)/Modules/_blake2/impl/blake2-impl.h $(srcdir)/Modules/_blake2/impl/blake2.h $(srcdir)/Modules/_blake2/impl/blake2b-load-sse2.h $(srcdir)/Modules/_blake2/impl/blake2b-load-sse41.h $(srcdir)/Modules/_blake2/impl/blake2b-ref.c $(srcdir)/Modules/_blake2/impl/blake2b-round.h $(srcdir)/Modules/_blake2/impl/blake2b.c $(srcdir)/Modules/_blake2/impl/blake2s-load-sse2.h $(srcdir)/Modules/_blake2/impl/blake2s-load-sse41.h $(srcdir)/Modules/_blake2/impl/blake2s-load-xop.h $(srcdir)/Modules/_blake2/impl/blake2s-ref.c $(srcdir)/Modules/_blake2/impl/blake2s-round.h $(srcdir)/Modules/_blake2/impl/blake2s.c $(srcdir)/Modules/_blake2/blake2module.h $(srcdir)/Modules/hashlib.h
MODULE__CTYPES_DEPS=$(srcdir)/Modules/_ctypes/ctypes.h
MODULE__CTYPES_MALLOC_CLOSURE=
MODULE__DECIMAL_DEPS=$(srcdir)/Modules/_decimal/docstrings.h $(LIBMPDEC_HEADERS) $(LIBMPDEC_A)
MODULE__ELEMENTTREE_DEPS=$(srcdir)/Modules/pyexpat.c $(LIBEXPAT_HEADERS) $(LIBEXPAT_A)
MODULE__HASHLIB_DEPS=$(srcdir)/Modules/hashlib.h
MODULE__IO_DEPS=$(srcdir)/Modules/_io/_iomodule.h
MODULE__MD5_DEPS=$(srcdir)/Modules/hashlib.h $(LIBHACL_HEADERS) Modules/_hacl/Hacl_Hash_MD5.h Modules/_hacl/Hacl_Hash_MD5.c
MODULE__SHA1_DEPS=$(srcdir)/Modules/hashlib.h $(LIBHACL_HEADERS) Modules/_hacl/Hacl_Hash_SHA1.h Modules/_hacl/Hacl_Hash_SHA1.c
MODULE__SHA2_DEPS=$(srcdir)/Modules/hashlib.h $(LIBHACL_SHA2_HEADERS) $(LIBHACL_SHA2_A)
MODULE__SHA3_DEPS=$(srcdir)/Modules/hashlib.h $(LIBHACL_HEADERS) Modules/_hacl/Hacl_Hash_SHA3.h Modules/_hacl/Hacl_Hash_SHA3.c
MODULE__SOCKET_DEPS=$(srcdir)/Modules/socketmodule.h $(srcdir)/Modules/addrinfo.h $(srcdir)/Modules/getaddrinfo.c $(srcdir)/Modules/getnameinfo.c
MODULE__SSL_DEPS=$(srcdir)/Modules/_ssl.h $(srcdir)/Modules/_ssl/cert.c $(srcdir)/Modules/_ssl/debughelpers.c $(srcdir)/Modules/_ssl/misc.c $(srcdir)/Modules/_ssl_data_111.h $(srcdir)/Modules/_ssl_data_300.h $(srcdir)/Modules/socketmodule.h
MODULE__TESTCAPI_DEPS=$(srcdir)/Modules/_testcapi/parts.h $(srcdir)/Modules/_testcapi/util.h
MODULE__TESTLIMITEDCAPI_DEPS=$(srcdir)/Modules/_testlimitedcapi/testcapi_long.h $(srcdir)/Modules/_testlimitedcapi/parts.h $(srcdir)/Modules/_testlimitedcapi/util.h
MODULE__TESTINTERNALCAPI_DEPS=$(srcdir)/Modules/_testinternalcapi/parts.h
MODULE__SQLITE3_DEPS=$(srcdir)/Modules/_sqlite/connection.h $(srcdir)/Modules/_sqlite/cursor.h $(srcdir)/Modules/_sqlite/microprotocols.h $(srcdir)/Modules/_sqlite/module.h $(srcdir)/Modules/_sqlite/prepare_protocol.h $(srcdir)/Modules/_sqlite/row.h $(srcdir)/Modules/_sqlite/util.h

CODECS_COMMON_HEADERS=$(srcdir)/Modules/cjkcodecs/multibytecodec.h $(srcdir)/Modules/cjkcodecs/cjkcodecs.h
MODULE__CODECS_CN_DEPS=$(srcdir)/Modules/cjkcodecs/mappings_cn.h $(CODECS_COMMON_HEADERS)
MODULE__CODECS_HK_DEPS=$(srcdir)/Modules/cjkcodecs/mappings_hk.h  $(CODECS_COMMON_HEADERS)
MODULE__CODECS_ISO2022_DEPS=$(srcdir)/Modules/cjkcodecs/mappings_jisx0213_pair.h $(srcdir)/Modules/cjkcodecs/alg_jisx0201.h $(srcdir)/Modules/cjkcodecs/emu_jisx0213_2000.h $(CODECS_COMMON_HEADERS)
MODULE__CODECS_JP_DEPS=$(srcdir)/Modules/cjkcodecs/mappings_jisx0213_pair.h $(srcdir)/Modules/cjkcodecs/alg_jisx0201.h $(srcdir)/Modules/cjkcodecs/emu_jisx0213_2000.h $(srcdir)/Modules/cjkcodecs/mappings_jp.h $(CODECS_COMMON_HEADERS)
MODULE__CODECS_KR_DEPS=$(srcdir)/Modules/cjkcodecs/mappings_kr.h $(CODECS_COMMON_HEADERS)
MODULE__CODECS_TW_DEPS=$(srcdir)/Modules/cjkcodecs/mappings_tw.h $(CODECS_COMMON_HEADERS)
MODULE__MULTIBYTECODEC_DEPS=$(srcdir)/Modules/cjkcodecs/multibytecodec.h

# IF YOU PUT ANYTHING HERE IT WILL GO AWAY
# Local Variables:
# mode: makefile
# End:

# Rules appended by makesetup

Modules/arraymodule.o: $(srcdir)/Modules/arraymodule.c $(MODULE_ARRAY_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_ARRAY_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/arraymodule.c -o Modules/arraymodule.o
Modules/array$(EXT_SUFFIX):  Modules/arraymodule.o; $(BLDSHARED)  Modules/arraymodule.o $(MODULE_ARRAY_LDFLAGS) $(LIBPYTHON) -o Modules/array$(EXT_SUFFIX)
Modules/_asynciomodule.o: $(srcdir)/Modules/_asynciomodule.c $(MODULE__ASYNCIO_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__ASYNCIO_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_asynciomodule.c -o Modules/_asynciomodule.o
Modules/_asyncio$(EXT_SUFFIX):  Modules/_asynciomodule.o; $(BLDSHARED)  Modules/_asynciomodule.o $(MODULE__ASYNCIO_LDFLAGS) $(LIBPYTHON) -o Modules/_asyncio$(EXT_SUFFIX)
Modules/_bisectmodule.o: $(srcdir)/Modules/_bisectmodule.c $(MODULE__BISECT_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__BISECT_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_bisectmodule.c -o Modules/_bisectmodule.o
Modules/_bisect$(EXT_SUFFIX):  Modules/_bisectmodule.o; $(BLDSHARED)  Modules/_bisectmodule.o $(MODULE__BISECT_LDFLAGS) $(LIBPYTHON) -o Modules/_bisect$(EXT_SUFFIX)
Modules/_contextvarsmodule.o: $(srcdir)/Modules/_contextvarsmodule.c $(MODULE__CONTEXTVARS_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CONTEXTVARS_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_contextvarsmodule.c -o Modules/_contextvarsmodule.o
Modules/_contextvars$(EXT_SUFFIX):  Modules/_contextvarsmodule.o; $(BLDSHARED)  Modules/_contextvarsmodule.o $(MODULE__CONTEXTVARS_LDFLAGS) $(LIBPYTHON) -o Modules/_contextvars$(EXT_SUFFIX)
Modules/_csv.o: $(srcdir)/Modules/_csv.c $(MODULE__CSV_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CSV_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_csv.c -o Modules/_csv.o
Modules/_csv$(EXT_SUFFIX):  Modules/_csv.o; $(BLDSHARED)  Modules/_csv.o $(MODULE__CSV_LDFLAGS) $(LIBPYTHON) -o Modules/_csv$(EXT_SUFFIX)
Modules/_heapqmodule.o: $(srcdir)/Modules/_heapqmodule.c $(MODULE__HEAPQ_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__HEAPQ_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_heapqmodule.c -o Modules/_heapqmodule.o
Modules/_heapq$(EXT_SUFFIX):  Modules/_heapqmodule.o; $(BLDSHARED)  Modules/_heapqmodule.o $(MODULE__HEAPQ_LDFLAGS) $(LIBPYTHON) -o Modules/_heapq$(EXT_SUFFIX)
Modules/_json.o: $(srcdir)/Modules/_json.c $(MODULE__JSON_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__JSON_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_json.c -o Modules/_json.o
Modules/_json$(EXT_SUFFIX):  Modules/_json.o; $(BLDSHARED)  Modules/_json.o $(MODULE__JSON_LDFLAGS) $(LIBPYTHON) -o Modules/_json$(EXT_SUFFIX)
Modules/_lsprof.o: $(srcdir)/Modules/_lsprof.c $(MODULE__LSPROF_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__LSPROF_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_lsprof.c -o Modules/_lsprof.o
Modules/rotatingtree.o: $(srcdir)/Modules/rotatingtree.c $(MODULE__LSPROF_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__LSPROF_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/rotatingtree.c -o Modules/rotatingtree.o
Modules/_lsprof$(EXT_SUFFIX):  Modules/_lsprof.o Modules/rotatingtree.o; $(BLDSHARED)  Modules/_lsprof.o Modules/rotatingtree.o $(MODULE__LSPROF_LDFLAGS) $(LIBPYTHON) -o Modules/_lsprof$(EXT_SUFFIX)
Modules/_opcode.o: $(srcdir)/Modules/_opcode.c $(MODULE__OPCODE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__OPCODE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_opcode.c -o Modules/_opcode.o
Modules/_opcode$(EXT_SUFFIX):  Modules/_opcode.o; $(BLDSHARED)  Modules/_opcode.o $(MODULE__OPCODE_LDFLAGS) $(LIBPYTHON) -o Modules/_opcode$(EXT_SUFFIX)
Modules/_pickle.o: $(srcdir)/Modules/_pickle.c $(MODULE__PICKLE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__PICKLE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_pickle.c -o Modules/_pickle.o
Modules/_pickle$(EXT_SUFFIX):  Modules/_pickle.o; $(BLDSHARED)  Modules/_pickle.o $(MODULE__PICKLE_LDFLAGS) $(LIBPYTHON) -o Modules/_pickle$(EXT_SUFFIX)
Modules/_queuemodule.o: $(srcdir)/Modules/_queuemodule.c $(MODULE__QUEUE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__QUEUE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_queuemodule.c -o Modules/_queuemodule.o
Modules/_queue$(EXT_SUFFIX):  Modules/_queuemodule.o; $(BLDSHARED)  Modules/_queuemodule.o $(MODULE__QUEUE_LDFLAGS) $(LIBPYTHON) -o Modules/_queue$(EXT_SUFFIX)
Modules/_randommodule.o: $(srcdir)/Modules/_randommodule.c $(MODULE__RANDOM_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__RANDOM_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_randommodule.c -o Modules/_randommodule.o
Modules/_random$(EXT_SUFFIX):  Modules/_randommodule.o; $(BLDSHARED)  Modules/_randommodule.o $(MODULE__RANDOM_LDFLAGS) $(LIBPYTHON) -o Modules/_random$(EXT_SUFFIX)
Modules/_struct.o: $(srcdir)/Modules/_struct.c $(MODULE__STRUCT_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__STRUCT_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_struct.c -o Modules/_struct.o
Modules/_struct$(EXT_SUFFIX):  Modules/_struct.o; $(BLDSHARED)  Modules/_struct.o $(MODULE__STRUCT_LDFLAGS) $(LIBPYTHON) -o Modules/_struct$(EXT_SUFFIX)
Modules/_interpretersmodule.o: $(srcdir)/Modules/_interpretersmodule.c $(MODULE__INTERPRETERS_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__INTERPRETERS_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_interpretersmodule.c -o Modules/_interpretersmodule.o
Modules/_interpreters$(EXT_SUFFIX):  Modules/_interpretersmodule.o; $(BLDSHARED)  Modules/_interpretersmodule.o $(MODULE__INTERPRETERS_LDFLAGS) $(LIBPYTHON) -o Modules/_interpreters$(EXT_SUFFIX)
Modules/_interpchannelsmodule.o: $(srcdir)/Modules/_interpchannelsmodule.c $(MODULE__INTERPCHANNELS_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__INTERPCHANNELS_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_interpchannelsmodule.c -o Modules/_interpchannelsmodule.o
Modules/_interpchannels$(EXT_SUFFIX):  Modules/_interpchannelsmodule.o; $(BLDSHARED)  Modules/_interpchannelsmodule.o $(MODULE__INTERPCHANNELS_LDFLAGS) $(LIBPYTHON) -o Modules/_interpchannels$(EXT_SUFFIX)
Modules/_interpqueuesmodule.o: $(srcdir)/Modules/_interpqueuesmodule.c $(MODULE__INTERPQUEUES_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__INTERPQUEUES_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_interpqueuesmodule.c -o Modules/_interpqueuesmodule.o
Modules/_interpqueues$(EXT_SUFFIX):  Modules/_interpqueuesmodule.o; $(BLDSHARED)  Modules/_interpqueuesmodule.o $(MODULE__INTERPQUEUES_LDFLAGS) $(LIBPYTHON) -o Modules/_interpqueues$(EXT_SUFFIX)
Modules/_zoneinfo.o: $(srcdir)/Modules/_zoneinfo.c $(MODULE__ZONEINFO_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__ZONEINFO_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_zoneinfo.c -o Modules/_zoneinfo.o
Modules/_zoneinfo$(EXT_SUFFIX):  Modules/_zoneinfo.o; $(BLDSHARED)  Modules/_zoneinfo.o $(MODULE__ZONEINFO_LDFLAGS) $(LIBPYTHON) -o Modules/_zoneinfo$(EXT_SUFFIX)
Modules/mathmodule.o: $(srcdir)/Modules/mathmodule.c $(MODULE_MATH_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_MATH_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/mathmodule.c -o Modules/mathmodule.o
Modules/math$(EXT_SUFFIX):  Modules/mathmodule.o; $(BLDSHARED)  Modules/mathmodule.o $(MODULE_MATH_LDFLAGS) $(LIBPYTHON) -o Modules/math$(EXT_SUFFIX)
Modules/cmathmodule.o: $(srcdir)/Modules/cmathmodule.c $(MODULE_CMATH_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_CMATH_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/cmathmodule.c -o Modules/cmathmodule.o
Modules/cmath$(EXT_SUFFIX):  Modules/cmathmodule.o; $(BLDSHARED)  Modules/cmathmodule.o $(MODULE_CMATH_LDFLAGS) $(LIBPYTHON) -o Modules/cmath$(EXT_SUFFIX)
Modules/_statisticsmodule.o: $(srcdir)/Modules/_statisticsmodule.c $(MODULE__STATISTICS_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__STATISTICS_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_statisticsmodule.c -o Modules/_statisticsmodule.o
Modules/_statistics$(EXT_SUFFIX):  Modules/_statisticsmodule.o; $(BLDSHARED)  Modules/_statisticsmodule.o $(MODULE__STATISTICS_LDFLAGS) $(LIBPYTHON) -o Modules/_statistics$(EXT_SUFFIX)
Modules/_datetimemodule.o: $(srcdir)/Modules/_datetimemodule.c $(MODULE__DATETIME_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__DATETIME_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_datetimemodule.c -o Modules/_datetimemodule.o
Modules/_datetime$(EXT_SUFFIX):  Modules/_datetimemodule.o; $(BLDSHARED)  Modules/_datetimemodule.o $(MODULE__DATETIME_LDFLAGS) $(LIBPYTHON) -o Modules/_datetime$(EXT_SUFFIX)
Modules/_decimal/_decimal.o: $(srcdir)/Modules/_decimal/_decimal.c $(MODULE__DECIMAL_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__DECIMAL_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_decimal/_decimal.c -o Modules/_decimal/_decimal.o
Modules/_decimal$(EXT_SUFFIX):  Modules/_decimal/_decimal.o; $(BLDSHARED)  Modules/_decimal/_decimal.o $(MODULE__DECIMAL_LDFLAGS) $(LIBPYTHON) -o Modules/_decimal$(EXT_SUFFIX)
Modules/binascii.o: $(srcdir)/Modules/binascii.c $(MODULE_BINASCII_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_BINASCII_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/binascii.c -o Modules/binascii.o
Modules/binascii$(EXT_SUFFIX):  Modules/binascii.o; $(BLDSHARED)  Modules/binascii.o $(MODULE_BINASCII_LDFLAGS) $(LIBPYTHON) -o Modules/binascii$(EXT_SUFFIX)
Modules/_bz2module.o: $(srcdir)/Modules/_bz2module.c $(MODULE__BZ2_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__BZ2_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_bz2module.c -o Modules/_bz2module.o
Modules/_bz2$(EXT_SUFFIX):  Modules/_bz2module.o; $(BLDSHARED)  Modules/_bz2module.o $(MODULE__BZ2_LDFLAGS) $(LIBPYTHON) -o Modules/_bz2$(EXT_SUFFIX)
Modules/_lzmamodule.o: $(srcdir)/Modules/_lzmamodule.c $(MODULE__LZMA_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__LZMA_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_lzmamodule.c -o Modules/_lzmamodule.o
Modules/_lzma$(EXT_SUFFIX):  Modules/_lzmamodule.o; $(BLDSHARED)  Modules/_lzmamodule.o $(MODULE__LZMA_LDFLAGS) $(LIBPYTHON) -o Modules/_lzma$(EXT_SUFFIX)
Modules/zlibmodule.o: $(srcdir)/Modules/zlibmodule.c $(MODULE_ZLIB_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_ZLIB_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/zlibmodule.c -o Modules/zlibmodule.o
Modules/zlib$(EXT_SUFFIX):  Modules/zlibmodule.o; $(BLDSHARED)  Modules/zlibmodule.o $(MODULE_ZLIB_LDFLAGS) $(LIBPYTHON) -o Modules/zlib$(EXT_SUFFIX)
Modules/_dbmmodule.o: $(srcdir)/Modules/_dbmmodule.c $(MODULE__DBM_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__DBM_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_dbmmodule.c -o Modules/_dbmmodule.o
Modules/_dbm$(EXT_SUFFIX):  Modules/_dbmmodule.o; $(BLDSHARED)  Modules/_dbmmodule.o $(MODULE__DBM_LDFLAGS) $(LIBPYTHON) -o Modules/_dbm$(EXT_SUFFIX)
Modules/_gdbmmodule.o: $(srcdir)/Modules/_gdbmmodule.c $(MODULE__GDBM_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__GDBM_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_gdbmmodule.c -o Modules/_gdbmmodule.o
Modules/_gdbm$(EXT_SUFFIX):  Modules/_gdbmmodule.o; $(BLDSHARED)  Modules/_gdbmmodule.o $(MODULE__GDBM_LDFLAGS) $(LIBPYTHON) -o Modules/_gdbm$(EXT_SUFFIX)
Modules/readline.o: $(srcdir)/Modules/readline.c $(MODULE_READLINE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_READLINE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/readline.c -o Modules/readline.o
Modules/readline$(EXT_SUFFIX):  Modules/readline.o; $(BLDSHARED)  Modules/readline.o $(MODULE_READLINE_LDFLAGS) $(LIBPYTHON) -o Modules/readline$(EXT_SUFFIX)
Modules/md5module.o: $(srcdir)/Modules/md5module.c $(MODULE__MD5_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC)  -I$(srcdir)/Modules/_hacl/include -D_BSD_SOURCE -D_DEFAULT_SOURCE $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/md5module.c -o Modules/md5module.o
Modules/_hacl/Hacl_Hash_MD5.o: $(srcdir)/Modules/_hacl/Hacl_Hash_MD5.c $(MODULE__MD5_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC)  -I$(srcdir)/Modules/_hacl/include -D_BSD_SOURCE -D_DEFAULT_SOURCE $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_hacl/Hacl_Hash_MD5.c -o Modules/_hacl/Hacl_Hash_MD5.o
Modules/_md5$(EXT_SUFFIX):  Modules/md5module.o Modules/_hacl/Hacl_Hash_MD5.o; $(BLDSHARED)  Modules/md5module.o Modules/_hacl/Hacl_Hash_MD5.o  $(LIBPYTHON) -o Modules/_md5$(EXT_SUFFIX)
Modules/sha1module.o: $(srcdir)/Modules/sha1module.c $(MODULE__SHA1_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC)  -I$(srcdir)/Modules/_hacl/include -D_BSD_SOURCE -D_DEFAULT_SOURCE $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/sha1module.c -o Modules/sha1module.o
Modules/_hacl/Hacl_Hash_SHA1.o: $(srcdir)/Modules/_hacl/Hacl_Hash_SHA1.c $(MODULE__SHA1_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC)  -I$(srcdir)/Modules/_hacl/include -D_BSD_SOURCE -D_DEFAULT_SOURCE $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_hacl/Hacl_Hash_SHA1.c -o Modules/_hacl/Hacl_Hash_SHA1.o
Modules/_sha1$(EXT_SUFFIX):  Modules/sha1module.o Modules/_hacl/Hacl_Hash_SHA1.o; $(BLDSHARED)  Modules/sha1module.o Modules/_hacl/Hacl_Hash_SHA1.o  $(LIBPYTHON) -o Modules/_sha1$(EXT_SUFFIX)
Modules/sha2module.o: $(srcdir)/Modules/sha2module.c $(MODULE__SHA2_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC)  -I$(srcdir)/Modules/_hacl/include $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/sha2module.c -o Modules/sha2module.o
Modules/_sha2$(EXT_SUFFIX):  Modules/sha2module.o; $(BLDSHARED)  Modules/sha2module.o  Modules/_hacl/libHacl_Hash_SHA2.a $(LIBPYTHON) -o Modules/_sha2$(EXT_SUFFIX)
Modules/sha3module.o: $(srcdir)/Modules/sha3module.c $(MODULE__SHA3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC)  -I$(srcdir)/Modules/_hacl/include -D_BSD_SOURCE -D_DEFAULT_SOURCE $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/sha3module.c -o Modules/sha3module.o
Modules/_hacl/Hacl_Hash_SHA3.o: $(srcdir)/Modules/_hacl/Hacl_Hash_SHA3.c $(MODULE__SHA3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC)  -I$(srcdir)/Modules/_hacl/include -D_BSD_SOURCE -D_DEFAULT_SOURCE $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_hacl/Hacl_Hash_SHA3.c -o Modules/_hacl/Hacl_Hash_SHA3.o
Modules/_sha3$(EXT_SUFFIX):  Modules/sha3module.o Modules/_hacl/Hacl_Hash_SHA3.o; $(BLDSHARED)  Modules/sha3module.o Modules/_hacl/Hacl_Hash_SHA3.o  $(LIBPYTHON) -o Modules/_sha3$(EXT_SUFFIX)
Modules/_blake2/blake2module.o: $(srcdir)/Modules/_blake2/blake2module.c $(MODULE__BLAKE2_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__BLAKE2_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_blake2/blake2module.c -o Modules/_blake2/blake2module.o
Modules/_blake2/blake2b_impl.o: $(srcdir)/Modules/_blake2/blake2b_impl.c $(MODULE__BLAKE2_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__BLAKE2_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_blake2/blake2b_impl.c -o Modules/_blake2/blake2b_impl.o
Modules/_blake2/blake2s_impl.o: $(srcdir)/Modules/_blake2/blake2s_impl.c $(MODULE__BLAKE2_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__BLAKE2_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_blake2/blake2s_impl.c -o Modules/_blake2/blake2s_impl.o
Modules/_blake2$(EXT_SUFFIX):  Modules/_blake2/blake2module.o Modules/_blake2/blake2b_impl.o Modules/_blake2/blake2s_impl.o; $(BLDSHARED)  Modules/_blake2/blake2module.o Modules/_blake2/blake2b_impl.o Modules/_blake2/blake2s_impl.o $(MODULE__BLAKE2_LDFLAGS) $(LIBPYTHON) -o Modules/_blake2$(EXT_SUFFIX)
Modules/pyexpat.o: $(srcdir)/Modules/pyexpat.c $(MODULE_PYEXPAT_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_PYEXPAT_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/pyexpat.c -o Modules/pyexpat.o
Modules/pyexpat$(EXT_SUFFIX):  Modules/pyexpat.o; $(BLDSHARED)  Modules/pyexpat.o $(MODULE_PYEXPAT_LDFLAGS) $(LIBPYTHON) -o Modules/pyexpat$(EXT_SUFFIX)
Modules/_elementtree.o: $(srcdir)/Modules/_elementtree.c $(MODULE__ELEMENTTREE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__ELEMENTTREE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_elementtree.c -o Modules/_elementtree.o
Modules/_elementtree$(EXT_SUFFIX):  Modules/_elementtree.o; $(BLDSHARED)  Modules/_elementtree.o $(MODULE__ELEMENTTREE_LDFLAGS) $(LIBPYTHON) -o Modules/_elementtree$(EXT_SUFFIX)
Modules/cjkcodecs/_codecs_cn.o: $(srcdir)/Modules/cjkcodecs/_codecs_cn.c $(MODULE__CODECS_CN_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CODECS_CN_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/cjkcodecs/_codecs_cn.c -o Modules/cjkcodecs/_codecs_cn.o
Modules/_codecs_cn$(EXT_SUFFIX):  Modules/cjkcodecs/_codecs_cn.o; $(BLDSHARED)  Modules/cjkcodecs/_codecs_cn.o $(MODULE__CODECS_CN_LDFLAGS) $(LIBPYTHON) -o Modules/_codecs_cn$(EXT_SUFFIX)
Modules/cjkcodecs/_codecs_hk.o: $(srcdir)/Modules/cjkcodecs/_codecs_hk.c $(MODULE__CODECS_HK_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CODECS_HK_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/cjkcodecs/_codecs_hk.c -o Modules/cjkcodecs/_codecs_hk.o
Modules/_codecs_hk$(EXT_SUFFIX):  Modules/cjkcodecs/_codecs_hk.o; $(BLDSHARED)  Modules/cjkcodecs/_codecs_hk.o $(MODULE__CODECS_HK_LDFLAGS) $(LIBPYTHON) -o Modules/_codecs_hk$(EXT_SUFFIX)
Modules/cjkcodecs/_codecs_iso2022.o: $(srcdir)/Modules/cjkcodecs/_codecs_iso2022.c $(MODULE__CODECS_ISO2022_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CODECS_ISO2022_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/cjkcodecs/_codecs_iso2022.c -o Modules/cjkcodecs/_codecs_iso2022.o
Modules/_codecs_iso2022$(EXT_SUFFIX):  Modules/cjkcodecs/_codecs_iso2022.o; $(BLDSHARED)  Modules/cjkcodecs/_codecs_iso2022.o $(MODULE__CODECS_ISO2022_LDFLAGS) $(LIBPYTHON) -o Modules/_codecs_iso2022$(EXT_SUFFIX)
Modules/cjkcodecs/_codecs_jp.o: $(srcdir)/Modules/cjkcodecs/_codecs_jp.c $(MODULE__CODECS_JP_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CODECS_JP_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/cjkcodecs/_codecs_jp.c -o Modules/cjkcodecs/_codecs_jp.o
Modules/_codecs_jp$(EXT_SUFFIX):  Modules/cjkcodecs/_codecs_jp.o; $(BLDSHARED)  Modules/cjkcodecs/_codecs_jp.o $(MODULE__CODECS_JP_LDFLAGS) $(LIBPYTHON) -o Modules/_codecs_jp$(EXT_SUFFIX)
Modules/cjkcodecs/_codecs_kr.o: $(srcdir)/Modules/cjkcodecs/_codecs_kr.c $(MODULE__CODECS_KR_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CODECS_KR_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/cjkcodecs/_codecs_kr.c -o Modules/cjkcodecs/_codecs_kr.o
Modules/_codecs_kr$(EXT_SUFFIX):  Modules/cjkcodecs/_codecs_kr.o; $(BLDSHARED)  Modules/cjkcodecs/_codecs_kr.o $(MODULE__CODECS_KR_LDFLAGS) $(LIBPYTHON) -o Modules/_codecs_kr$(EXT_SUFFIX)
Modules/cjkcodecs/_codecs_tw.o: $(srcdir)/Modules/cjkcodecs/_codecs_tw.c $(MODULE__CODECS_TW_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CODECS_TW_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/cjkcodecs/_codecs_tw.c -o Modules/cjkcodecs/_codecs_tw.o
Modules/_codecs_tw$(EXT_SUFFIX):  Modules/cjkcodecs/_codecs_tw.o; $(BLDSHARED)  Modules/cjkcodecs/_codecs_tw.o $(MODULE__CODECS_TW_LDFLAGS) $(LIBPYTHON) -o Modules/_codecs_tw$(EXT_SUFFIX)
Modules/cjkcodecs/multibytecodec.o: $(srcdir)/Modules/cjkcodecs/multibytecodec.c $(MODULE__MULTIBYTECODEC_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__MULTIBYTECODEC_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/cjkcodecs/multibytecodec.c -o Modules/cjkcodecs/multibytecodec.o
Modules/_multibytecodec$(EXT_SUFFIX):  Modules/cjkcodecs/multibytecodec.o; $(BLDSHARED)  Modules/cjkcodecs/multibytecodec.o $(MODULE__MULTIBYTECODEC_LDFLAGS) $(LIBPYTHON) -o Modules/_multibytecodec$(EXT_SUFFIX)
Modules/unicodedata.o: $(srcdir)/Modules/unicodedata.c $(MODULE_UNICODEDATA_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_UNICODEDATA_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/unicodedata.c -o Modules/unicodedata.o
Modules/unicodedata$(EXT_SUFFIX):  Modules/unicodedata.o; $(BLDSHARED)  Modules/unicodedata.o $(MODULE_UNICODEDATA_LDFLAGS) $(LIBPYTHON) -o Modules/unicodedata$(EXT_SUFFIX)
Modules/fcntlmodule.o: $(srcdir)/Modules/fcntlmodule.c $(MODULE_FCNTL_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_FCNTL_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/fcntlmodule.c -o Modules/fcntlmodule.o
Modules/fcntl$(EXT_SUFFIX):  Modules/fcntlmodule.o; $(BLDSHARED)  Modules/fcntlmodule.o $(MODULE_FCNTL_LDFLAGS) $(LIBPYTHON) -o Modules/fcntl$(EXT_SUFFIX)
Modules/grpmodule.o: $(srcdir)/Modules/grpmodule.c $(MODULE_GRP_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_GRP_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/grpmodule.c -o Modules/grpmodule.o
Modules/grp$(EXT_SUFFIX):  Modules/grpmodule.o; $(BLDSHARED)  Modules/grpmodule.o $(MODULE_GRP_LDFLAGS) $(LIBPYTHON) -o Modules/grp$(EXT_SUFFIX)
Modules/mmapmodule.o: $(srcdir)/Modules/mmapmodule.c $(MODULE_MMAP_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_MMAP_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/mmapmodule.c -o Modules/mmapmodule.o
Modules/mmap$(EXT_SUFFIX):  Modules/mmapmodule.o; $(BLDSHARED)  Modules/mmapmodule.o $(MODULE_MMAP_LDFLAGS) $(LIBPYTHON) -o Modules/mmap$(EXT_SUFFIX)
Modules/_posixsubprocess.o: $(srcdir)/Modules/_posixsubprocess.c $(MODULE__POSIXSUBPROCESS_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__POSIXSUBPROCESS_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_posixsubprocess.c -o Modules/_posixsubprocess.o
Modules/_posixsubprocess$(EXT_SUFFIX):  Modules/_posixsubprocess.o; $(BLDSHARED)  Modules/_posixsubprocess.o $(MODULE__POSIXSUBPROCESS_LDFLAGS) $(LIBPYTHON) -o Modules/_posixsubprocess$(EXT_SUFFIX)
Modules/resource.o: $(srcdir)/Modules/resource.c $(MODULE_RESOURCE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_RESOURCE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/resource.c -o Modules/resource.o
Modules/resource$(EXT_SUFFIX):  Modules/resource.o; $(BLDSHARED)  Modules/resource.o $(MODULE_RESOURCE_LDFLAGS) $(LIBPYTHON) -o Modules/resource$(EXT_SUFFIX)
Modules/selectmodule.o: $(srcdir)/Modules/selectmodule.c $(MODULE_SELECT_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_SELECT_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/selectmodule.c -o Modules/selectmodule.o
Modules/select$(EXT_SUFFIX):  Modules/selectmodule.o; $(BLDSHARED)  Modules/selectmodule.o $(MODULE_SELECT_LDFLAGS) $(LIBPYTHON) -o Modules/select$(EXT_SUFFIX)
Modules/socketmodule.o: $(srcdir)/Modules/socketmodule.c $(MODULE__SOCKET_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SOCKET_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/socketmodule.c -o Modules/socketmodule.o
Modules/_socket$(EXT_SUFFIX):  Modules/socketmodule.o; $(BLDSHARED)  Modules/socketmodule.o $(MODULE__SOCKET_LDFLAGS) $(LIBPYTHON) -o Modules/_socket$(EXT_SUFFIX)
Modules/syslogmodule.o: $(srcdir)/Modules/syslogmodule.c $(MODULE_SYSLOG_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_SYSLOG_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/syslogmodule.c -o Modules/syslogmodule.o
Modules/syslog$(EXT_SUFFIX):  Modules/syslogmodule.o; $(BLDSHARED)  Modules/syslogmodule.o $(MODULE_SYSLOG_LDFLAGS) $(LIBPYTHON) -o Modules/syslog$(EXT_SUFFIX)
Modules/termios.o: $(srcdir)/Modules/termios.c $(MODULE_TERMIOS_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_TERMIOS_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/termios.c -o Modules/termios.o
Modules/termios$(EXT_SUFFIX):  Modules/termios.o; $(BLDSHARED)  Modules/termios.o $(MODULE_TERMIOS_LDFLAGS) $(LIBPYTHON) -o Modules/termios$(EXT_SUFFIX)
Modules/_multiprocessing/posixshmem.o: $(srcdir)/Modules/_multiprocessing/posixshmem.c $(MODULE__POSIXSHMEM_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__POSIXSHMEM_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_multiprocessing/posixshmem.c -o Modules/_multiprocessing/posixshmem.o
Modules/_posixshmem$(EXT_SUFFIX):  Modules/_multiprocessing/posixshmem.o; $(BLDSHARED)  Modules/_multiprocessing/posixshmem.o $(MODULE__POSIXSHMEM_LDFLAGS) $(LIBPYTHON) -o Modules/_posixshmem$(EXT_SUFFIX)
Modules/_multiprocessing/multiprocessing.o: $(srcdir)/Modules/_multiprocessing/multiprocessing.c $(MODULE__MULTIPROCESSING_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__MULTIPROCESSING_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_multiprocessing/multiprocessing.c -o Modules/_multiprocessing/multiprocessing.o
Modules/_multiprocessing/semaphore.o: $(srcdir)/Modules/_multiprocessing/semaphore.c $(MODULE__MULTIPROCESSING_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__MULTIPROCESSING_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_multiprocessing/semaphore.c -o Modules/_multiprocessing/semaphore.o
Modules/_multiprocessing$(EXT_SUFFIX):  Modules/_multiprocessing/multiprocessing.o Modules/_multiprocessing/semaphore.o; $(BLDSHARED)  Modules/_multiprocessing/multiprocessing.o Modules/_multiprocessing/semaphore.o $(MODULE__MULTIPROCESSING_LDFLAGS) $(LIBPYTHON) -o Modules/_multiprocessing$(EXT_SUFFIX)
Modules/_ctypes/_ctypes.o: $(srcdir)/Modules/_ctypes/_ctypes.c $(MODULE__CTYPES_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CTYPES_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_ctypes/_ctypes.c -o Modules/_ctypes/_ctypes.o
Modules/_ctypes/callbacks.o: $(srcdir)/Modules/_ctypes/callbacks.c $(MODULE__CTYPES_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CTYPES_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_ctypes/callbacks.c -o Modules/_ctypes/callbacks.o
Modules/_ctypes/callproc.o: $(srcdir)/Modules/_ctypes/callproc.c $(MODULE__CTYPES_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CTYPES_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_ctypes/callproc.c -o Modules/_ctypes/callproc.o
Modules/_ctypes/stgdict.o: $(srcdir)/Modules/_ctypes/stgdict.c $(MODULE__CTYPES_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CTYPES_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_ctypes/stgdict.c -o Modules/_ctypes/stgdict.o
Modules/_ctypes/cfield.o: $(srcdir)/Modules/_ctypes/cfield.c $(MODULE__CTYPES_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CTYPES_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_ctypes/cfield.c -o Modules/_ctypes/cfield.o
Modules/_ctypes$(EXT_SUFFIX):  Modules/_ctypes/_ctypes.o Modules/_ctypes/callbacks.o Modules/_ctypes/callproc.o Modules/_ctypes/stgdict.o Modules/_ctypes/cfield.o; $(BLDSHARED)  Modules/_ctypes/_ctypes.o Modules/_ctypes/callbacks.o Modules/_ctypes/callproc.o Modules/_ctypes/stgdict.o Modules/_ctypes/cfield.o $(MODULE__CTYPES_LDFLAGS) $(LIBPYTHON) -o Modules/_ctypes$(EXT_SUFFIX)
Modules/_cursesmodule.o: $(srcdir)/Modules/_cursesmodule.c $(MODULE__CURSES_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CURSES_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_cursesmodule.c -o Modules/_cursesmodule.o
Modules/_curses$(EXT_SUFFIX):  Modules/_cursesmodule.o; $(BLDSHARED)  Modules/_cursesmodule.o $(MODULE__CURSES_LDFLAGS) $(LIBPYTHON) -o Modules/_curses$(EXT_SUFFIX)
Modules/_curses_panel.o: $(srcdir)/Modules/_curses_panel.c $(MODULE__CURSES_PANEL_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CURSES_PANEL_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_curses_panel.c -o Modules/_curses_panel.o
Modules/_curses_panel$(EXT_SUFFIX):  Modules/_curses_panel.o; $(BLDSHARED)  Modules/_curses_panel.o $(MODULE__CURSES_PANEL_LDFLAGS) $(LIBPYTHON) -o Modules/_curses_panel$(EXT_SUFFIX)
Modules/_sqlite/blob.o: $(srcdir)/Modules/_sqlite/blob.c $(MODULE__SQLITE3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SQLITE3_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_sqlite/blob.c -o Modules/_sqlite/blob.o
Modules/_sqlite/connection.o: $(srcdir)/Modules/_sqlite/connection.c $(MODULE__SQLITE3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SQLITE3_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_sqlite/connection.c -o Modules/_sqlite/connection.o
Modules/_sqlite/cursor.o: $(srcdir)/Modules/_sqlite/cursor.c $(MODULE__SQLITE3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SQLITE3_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_sqlite/cursor.c -o Modules/_sqlite/cursor.o
Modules/_sqlite/microprotocols.o: $(srcdir)/Modules/_sqlite/microprotocols.c $(MODULE__SQLITE3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SQLITE3_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_sqlite/microprotocols.c -o Modules/_sqlite/microprotocols.o
Modules/_sqlite/module.o: $(srcdir)/Modules/_sqlite/module.c $(MODULE__SQLITE3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SQLITE3_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_sqlite/module.c -o Modules/_sqlite/module.o
Modules/_sqlite/prepare_protocol.o: $(srcdir)/Modules/_sqlite/prepare_protocol.c $(MODULE__SQLITE3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SQLITE3_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_sqlite/prepare_protocol.c -o Modules/_sqlite/prepare_protocol.o
Modules/_sqlite/row.o: $(srcdir)/Modules/_sqlite/row.c $(MODULE__SQLITE3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SQLITE3_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_sqlite/row.c -o Modules/_sqlite/row.o
Modules/_sqlite/statement.o: $(srcdir)/Modules/_sqlite/statement.c $(MODULE__SQLITE3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SQLITE3_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_sqlite/statement.c -o Modules/_sqlite/statement.o
Modules/_sqlite/util.o: $(srcdir)/Modules/_sqlite/util.c $(MODULE__SQLITE3_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__SQLITE3_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_sqlite/util.c -o Modules/_sqlite/util.o
Modules/_sqlite3$(EXT_SUFFIX):  Modules/_sqlite/blob.o Modules/_sqlite/connection.o Modules/_sqlite/cursor.o Modules/_sqlite/microprotocols.o Modules/_sqlite/module.o Modules/_sqlite/prepare_protocol.o Modules/_sqlite/row.o Modules/_sqlite/statement.o Modules/_sqlite/util.o; $(BLDSHARED)  Modules/_sqlite/blob.o Modules/_sqlite/connection.o Modules/_sqlite/cursor.o Modules/_sqlite/microprotocols.o Modules/_sqlite/module.o Modules/_sqlite/prepare_protocol.o Modules/_sqlite/row.o Modules/_sqlite/statement.o Modules/_sqlite/util.o $(MODULE__SQLITE3_LDFLAGS) $(LIBPYTHON) -o Modules/_sqlite3$(EXT_SUFFIX)
Modules/_uuidmodule.o: $(srcdir)/Modules/_uuidmodule.c $(MODULE__UUID_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__UUID_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_uuidmodule.c -o Modules/_uuidmodule.o
Modules/_uuid$(EXT_SUFFIX):  Modules/_uuidmodule.o; $(BLDSHARED)  Modules/_uuidmodule.o $(MODULE__UUID_LDFLAGS) $(LIBPYTHON) -o Modules/_uuid$(EXT_SUFFIX)
Modules/_tkinter.o: $(srcdir)/Modules/_tkinter.c $(MODULE__TKINTER_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TKINTER_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_tkinter.c -o Modules/_tkinter.o
Modules/tkappinit.o: $(srcdir)/Modules/tkappinit.c $(MODULE__TKINTER_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TKINTER_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/tkappinit.c -o Modules/tkappinit.o
Modules/_tkinter$(EXT_SUFFIX):  Modules/_tkinter.o Modules/tkappinit.o; $(BLDSHARED)  Modules/_tkinter.o Modules/tkappinit.o $(MODULE__TKINTER_LDFLAGS) $(LIBPYTHON) -o Modules/_tkinter$(EXT_SUFFIX)
Modules/xxsubtype.o: $(srcdir)/Modules/xxsubtype.c $(MODULE_XXSUBTYPE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_XXSUBTYPE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/xxsubtype.c -o Modules/xxsubtype.o
Modules/xxsubtype$(EXT_SUFFIX):  Modules/xxsubtype.o; $(BLDSHARED)  Modules/xxsubtype.o $(MODULE_XXSUBTYPE_LDFLAGS) $(LIBPYTHON) -o Modules/xxsubtype$(EXT_SUFFIX)
Modules/_xxtestfuzz/_xxtestfuzz.o: $(srcdir)/Modules/_xxtestfuzz/_xxtestfuzz.c $(MODULE__XXTESTFUZZ_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__XXTESTFUZZ_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_xxtestfuzz/_xxtestfuzz.c -o Modules/_xxtestfuzz/_xxtestfuzz.o
Modules/_xxtestfuzz/fuzzer.o: $(srcdir)/Modules/_xxtestfuzz/fuzzer.c $(MODULE__XXTESTFUZZ_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__XXTESTFUZZ_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_xxtestfuzz/fuzzer.c -o Modules/_xxtestfuzz/fuzzer.o
Modules/_xxtestfuzz$(EXT_SUFFIX):  Modules/_xxtestfuzz/_xxtestfuzz.o Modules/_xxtestfuzz/fuzzer.o; $(BLDSHARED)  Modules/_xxtestfuzz/_xxtestfuzz.o Modules/_xxtestfuzz/fuzzer.o $(MODULE__XXTESTFUZZ_LDFLAGS) $(LIBPYTHON) -o Modules/_xxtestfuzz$(EXT_SUFFIX)
Modules/_testbuffer.o: $(srcdir)/Modules/_testbuffer.c $(MODULE__TESTBUFFER_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTBUFFER_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testbuffer.c -o Modules/_testbuffer.o
Modules/_testbuffer$(EXT_SUFFIX):  Modules/_testbuffer.o; $(BLDSHARED)  Modules/_testbuffer.o $(MODULE__TESTBUFFER_LDFLAGS) $(LIBPYTHON) -o Modules/_testbuffer$(EXT_SUFFIX)
Modules/_testinternalcapi.o: $(srcdir)/Modules/_testinternalcapi.c $(MODULE__TESTINTERNALCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTINTERNALCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testinternalcapi.c -o Modules/_testinternalcapi.o
Modules/_testinternalcapi/test_lock.o: $(srcdir)/Modules/_testinternalcapi/test_lock.c $(MODULE__TESTINTERNALCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTINTERNALCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testinternalcapi/test_lock.c -o Modules/_testinternalcapi/test_lock.o
Modules/_testinternalcapi/pytime.o: $(srcdir)/Modules/_testinternalcapi/pytime.c $(MODULE__TESTINTERNALCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTINTERNALCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testinternalcapi/pytime.c -o Modules/_testinternalcapi/pytime.o
Modules/_testinternalcapi/set.o: $(srcdir)/Modules/_testinternalcapi/set.c $(MODULE__TESTINTERNALCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTINTERNALCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testinternalcapi/set.c -o Modules/_testinternalcapi/set.o
Modules/_testinternalcapi/test_critical_sections.o: $(srcdir)/Modules/_testinternalcapi/test_critical_sections.c $(MODULE__TESTINTERNALCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTINTERNALCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testinternalcapi/test_critical_sections.c -o Modules/_testinternalcapi/test_critical_sections.o
Modules/_testinternalcapi$(EXT_SUFFIX):  Modules/_testinternalcapi.o Modules/_testinternalcapi/test_lock.o Modules/_testinternalcapi/pytime.o Modules/_testinternalcapi/set.o Modules/_testinternalcapi/test_critical_sections.o; $(BLDSHARED)  Modules/_testinternalcapi.o Modules/_testinternalcapi/test_lock.o Modules/_testinternalcapi/pytime.o Modules/_testinternalcapi/set.o Modules/_testinternalcapi/test_critical_sections.o $(MODULE__TESTINTERNALCAPI_LDFLAGS) $(LIBPYTHON) -o Modules/_testinternalcapi$(EXT_SUFFIX)
Modules/_testcapimodule.o: $(srcdir)/Modules/_testcapimodule.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapimodule.c -o Modules/_testcapimodule.o
Modules/_testcapi/vectorcall.o: $(srcdir)/Modules/_testcapi/vectorcall.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/vectorcall.c -o Modules/_testcapi/vectorcall.o
Modules/_testcapi/heaptype.o: $(srcdir)/Modules/_testcapi/heaptype.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/heaptype.c -o Modules/_testcapi/heaptype.o
Modules/_testcapi/abstract.o: $(srcdir)/Modules/_testcapi/abstract.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/abstract.c -o Modules/_testcapi/abstract.o
Modules/_testcapi/unicode.o: $(srcdir)/Modules/_testcapi/unicode.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/unicode.c -o Modules/_testcapi/unicode.o
Modules/_testcapi/dict.o: $(srcdir)/Modules/_testcapi/dict.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/dict.c -o Modules/_testcapi/dict.o
Modules/_testcapi/set.o: $(srcdir)/Modules/_testcapi/set.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/set.c -o Modules/_testcapi/set.o
Modules/_testcapi/list.o: $(srcdir)/Modules/_testcapi/list.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/list.c -o Modules/_testcapi/list.o
Modules/_testcapi/tuple.o: $(srcdir)/Modules/_testcapi/tuple.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/tuple.c -o Modules/_testcapi/tuple.o
Modules/_testcapi/getargs.o: $(srcdir)/Modules/_testcapi/getargs.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/getargs.c -o Modules/_testcapi/getargs.o
Modules/_testcapi/datetime.o: $(srcdir)/Modules/_testcapi/datetime.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/datetime.c -o Modules/_testcapi/datetime.o
Modules/_testcapi/docstring.o: $(srcdir)/Modules/_testcapi/docstring.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/docstring.c -o Modules/_testcapi/docstring.o
Modules/_testcapi/mem.o: $(srcdir)/Modules/_testcapi/mem.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/mem.c -o Modules/_testcapi/mem.o
Modules/_testcapi/watchers.o: $(srcdir)/Modules/_testcapi/watchers.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/watchers.c -o Modules/_testcapi/watchers.o
Modules/_testcapi/long.o: $(srcdir)/Modules/_testcapi/long.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/long.c -o Modules/_testcapi/long.o
Modules/_testcapi/float.o: $(srcdir)/Modules/_testcapi/float.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/float.c -o Modules/_testcapi/float.o
Modules/_testcapi/complex.o: $(srcdir)/Modules/_testcapi/complex.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/complex.c -o Modules/_testcapi/complex.o
Modules/_testcapi/numbers.o: $(srcdir)/Modules/_testcapi/numbers.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/numbers.c -o Modules/_testcapi/numbers.o
Modules/_testcapi/structmember.o: $(srcdir)/Modules/_testcapi/structmember.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/structmember.c -o Modules/_testcapi/structmember.o
Modules/_testcapi/exceptions.o: $(srcdir)/Modules/_testcapi/exceptions.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/exceptions.c -o Modules/_testcapi/exceptions.o
Modules/_testcapi/code.o: $(srcdir)/Modules/_testcapi/code.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/code.c -o Modules/_testcapi/code.o
Modules/_testcapi/buffer.o: $(srcdir)/Modules/_testcapi/buffer.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/buffer.c -o Modules/_testcapi/buffer.o
Modules/_testcapi/pyatomic.o: $(srcdir)/Modules/_testcapi/pyatomic.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/pyatomic.c -o Modules/_testcapi/pyatomic.o
Modules/_testcapi/run.o: $(srcdir)/Modules/_testcapi/run.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/run.c -o Modules/_testcapi/run.o
Modules/_testcapi/file.o: $(srcdir)/Modules/_testcapi/file.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/file.c -o Modules/_testcapi/file.o
Modules/_testcapi/codec.o: $(srcdir)/Modules/_testcapi/codec.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/codec.c -o Modules/_testcapi/codec.o
Modules/_testcapi/immortal.o: $(srcdir)/Modules/_testcapi/immortal.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/immortal.c -o Modules/_testcapi/immortal.o
Modules/_testcapi/gc.o: $(srcdir)/Modules/_testcapi/gc.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/gc.c -o Modules/_testcapi/gc.o
Modules/_testcapi/hash.o: $(srcdir)/Modules/_testcapi/hash.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/hash.c -o Modules/_testcapi/hash.o
Modules/_testcapi/time.o: $(srcdir)/Modules/_testcapi/time.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/time.c -o Modules/_testcapi/time.o
Modules/_testcapi/bytes.o: $(srcdir)/Modules/_testcapi/bytes.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/bytes.c -o Modules/_testcapi/bytes.o
Modules/_testcapi/object.o: $(srcdir)/Modules/_testcapi/object.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/object.c -o Modules/_testcapi/object.o
Modules/_testcapi/monitoring.o: $(srcdir)/Modules/_testcapi/monitoring.c $(MODULE__TESTCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testcapi/monitoring.c -o Modules/_testcapi/monitoring.o
Modules/_testcapi$(EXT_SUFFIX):  Modules/_testcapimodule.o Modules/_testcapi/vectorcall.o Modules/_testcapi/heaptype.o Modules/_testcapi/abstract.o Modules/_testcapi/unicode.o Modules/_testcapi/dict.o Modules/_testcapi/set.o Modules/_testcapi/list.o Modules/_testcapi/tuple.o Modules/_testcapi/getargs.o Modules/_testcapi/datetime.o Modules/_testcapi/docstring.o Modules/_testcapi/mem.o Modules/_testcapi/watchers.o Modules/_testcapi/long.o Modules/_testcapi/float.o Modules/_testcapi/complex.o Modules/_testcapi/numbers.o Modules/_testcapi/structmember.o Modules/_testcapi/exceptions.o Modules/_testcapi/code.o Modules/_testcapi/buffer.o Modules/_testcapi/pyatomic.o Modules/_testcapi/run.o Modules/_testcapi/file.o Modules/_testcapi/codec.o Modules/_testcapi/immortal.o Modules/_testcapi/gc.o Modules/_testcapi/hash.o Modules/_testcapi/time.o Modules/_testcapi/bytes.o Modules/_testcapi/object.o Modules/_testcapi/monitoring.o; $(BLDSHARED)  Modules/_testcapimodule.o Modules/_testcapi/vectorcall.o Modules/_testcapi/heaptype.o Modules/_testcapi/abstract.o Modules/_testcapi/unicode.o Modules/_testcapi/dict.o Modules/_testcapi/set.o Modules/_testcapi/list.o Modules/_testcapi/tuple.o Modules/_testcapi/getargs.o Modules/_testcapi/datetime.o Modules/_testcapi/docstring.o Modules/_testcapi/mem.o Modules/_testcapi/watchers.o Modules/_testcapi/long.o Modules/_testcapi/float.o Modules/_testcapi/complex.o Modules/_testcapi/numbers.o Modules/_testcapi/structmember.o Modules/_testcapi/exceptions.o Modules/_testcapi/code.o Modules/_testcapi/buffer.o Modules/_testcapi/pyatomic.o Modules/_testcapi/run.o Modules/_testcapi/file.o Modules/_testcapi/codec.o Modules/_testcapi/immortal.o Modules/_testcapi/gc.o Modules/_testcapi/hash.o Modules/_testcapi/time.o Modules/_testcapi/bytes.o Modules/_testcapi/object.o Modules/_testcapi/monitoring.o $(MODULE__TESTCAPI_LDFLAGS) $(LIBPYTHON) -o Modules/_testcapi$(EXT_SUFFIX)
Modules/_testlimitedcapi.o: $(srcdir)/Modules/_testlimitedcapi.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi.c -o Modules/_testlimitedcapi.o
Modules/_testlimitedcapi/abstract.o: $(srcdir)/Modules/_testlimitedcapi/abstract.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/abstract.c -o Modules/_testlimitedcapi/abstract.o
Modules/_testlimitedcapi/bytearray.o: $(srcdir)/Modules/_testlimitedcapi/bytearray.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/bytearray.c -o Modules/_testlimitedcapi/bytearray.o
Modules/_testlimitedcapi/bytes.o: $(srcdir)/Modules/_testlimitedcapi/bytes.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/bytes.c -o Modules/_testlimitedcapi/bytes.o
Modules/_testlimitedcapi/complex.o: $(srcdir)/Modules/_testlimitedcapi/complex.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/complex.c -o Modules/_testlimitedcapi/complex.o
Modules/_testlimitedcapi/dict.o: $(srcdir)/Modules/_testlimitedcapi/dict.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/dict.c -o Modules/_testlimitedcapi/dict.o
Modules/_testlimitedcapi/float.o: $(srcdir)/Modules/_testlimitedcapi/float.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/float.c -o Modules/_testlimitedcapi/float.o
Modules/_testlimitedcapi/heaptype_relative.o: $(srcdir)/Modules/_testlimitedcapi/heaptype_relative.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/heaptype_relative.c -o Modules/_testlimitedcapi/heaptype_relative.o
Modules/_testlimitedcapi/list.o: $(srcdir)/Modules/_testlimitedcapi/list.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/list.c -o Modules/_testlimitedcapi/list.o
Modules/_testlimitedcapi/long.o: $(srcdir)/Modules/_testlimitedcapi/long.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/long.c -o Modules/_testlimitedcapi/long.o
Modules/_testlimitedcapi/object.o: $(srcdir)/Modules/_testlimitedcapi/object.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/object.c -o Modules/_testlimitedcapi/object.o
Modules/_testlimitedcapi/pyos.o: $(srcdir)/Modules/_testlimitedcapi/pyos.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/pyos.c -o Modules/_testlimitedcapi/pyos.o
Modules/_testlimitedcapi/set.o: $(srcdir)/Modules/_testlimitedcapi/set.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/set.c -o Modules/_testlimitedcapi/set.o
Modules/_testlimitedcapi/sys.o: $(srcdir)/Modules/_testlimitedcapi/sys.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/sys.c -o Modules/_testlimitedcapi/sys.o
Modules/_testlimitedcapi/tuple.o: $(srcdir)/Modules/_testlimitedcapi/tuple.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/tuple.c -o Modules/_testlimitedcapi/tuple.o
Modules/_testlimitedcapi/unicode.o: $(srcdir)/Modules/_testlimitedcapi/unicode.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/unicode.c -o Modules/_testlimitedcapi/unicode.o
Modules/_testlimitedcapi/vectorcall_limited.o: $(srcdir)/Modules/_testlimitedcapi/vectorcall_limited.c $(MODULE__TESTLIMITEDCAPI_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTLIMITEDCAPI_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testlimitedcapi/vectorcall_limited.c -o Modules/_testlimitedcapi/vectorcall_limited.o
Modules/_testlimitedcapi$(EXT_SUFFIX):  Modules/_testlimitedcapi.o Modules/_testlimitedcapi/abstract.o Modules/_testlimitedcapi/bytearray.o Modules/_testlimitedcapi/bytes.o Modules/_testlimitedcapi/complex.o Modules/_testlimitedcapi/dict.o Modules/_testlimitedcapi/float.o Modules/_testlimitedcapi/heaptype_relative.o Modules/_testlimitedcapi/list.o Modules/_testlimitedcapi/long.o Modules/_testlimitedcapi/object.o Modules/_testlimitedcapi/pyos.o Modules/_testlimitedcapi/set.o Modules/_testlimitedcapi/sys.o Modules/_testlimitedcapi/tuple.o Modules/_testlimitedcapi/unicode.o Modules/_testlimitedcapi/vectorcall_limited.o; $(BLDSHARED)  Modules/_testlimitedcapi.o Modules/_testlimitedcapi/abstract.o Modules/_testlimitedcapi/bytearray.o Modules/_testlimitedcapi/bytes.o Modules/_testlimitedcapi/complex.o Modules/_testlimitedcapi/dict.o Modules/_testlimitedcapi/float.o Modules/_testlimitedcapi/heaptype_relative.o Modules/_testlimitedcapi/list.o Modules/_testlimitedcapi/long.o Modules/_testlimitedcapi/object.o Modules/_testlimitedcapi/pyos.o Modules/_testlimitedcapi/set.o Modules/_testlimitedcapi/sys.o Modules/_testlimitedcapi/tuple.o Modules/_testlimitedcapi/unicode.o Modules/_testlimitedcapi/vectorcall_limited.o $(MODULE__TESTLIMITEDCAPI_LDFLAGS) $(LIBPYTHON) -o Modules/_testlimitedcapi$(EXT_SUFFIX)
Modules/_testclinic.o: $(srcdir)/Modules/_testclinic.c $(MODULE__TESTCLINIC_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCLINIC_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testclinic.c -o Modules/_testclinic.o
Modules/_testclinic$(EXT_SUFFIX):  Modules/_testclinic.o; $(BLDSHARED)  Modules/_testclinic.o $(MODULE__TESTCLINIC_LDFLAGS) $(LIBPYTHON) -o Modules/_testclinic$(EXT_SUFFIX)
Modules/_testclinic_limited.o: $(srcdir)/Modules/_testclinic_limited.c $(MODULE__TESTCLINIC_LIMITED_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTCLINIC_LIMITED_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testclinic_limited.c -o Modules/_testclinic_limited.o
Modules/_testclinic_limited$(EXT_SUFFIX):  Modules/_testclinic_limited.o; $(BLDSHARED)  Modules/_testclinic_limited.o $(MODULE__TESTCLINIC_LIMITED_LDFLAGS) $(LIBPYTHON) -o Modules/_testclinic_limited$(EXT_SUFFIX)
Modules/_testimportmultiple.o: $(srcdir)/Modules/_testimportmultiple.c $(MODULE__TESTIMPORTMULTIPLE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTIMPORTMULTIPLE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testimportmultiple.c -o Modules/_testimportmultiple.o
Modules/_testimportmultiple$(EXT_SUFFIX):  Modules/_testimportmultiple.o; $(BLDSHARED)  Modules/_testimportmultiple.o $(MODULE__TESTIMPORTMULTIPLE_LDFLAGS) $(LIBPYTHON) -o Modules/_testimportmultiple$(EXT_SUFFIX)
Modules/_testmultiphase.o: $(srcdir)/Modules/_testmultiphase.c $(MODULE__TESTMULTIPHASE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTMULTIPHASE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testmultiphase.c -o Modules/_testmultiphase.o
Modules/_testmultiphase$(EXT_SUFFIX):  Modules/_testmultiphase.o; $(BLDSHARED)  Modules/_testmultiphase.o $(MODULE__TESTMULTIPHASE_LDFLAGS) $(LIBPYTHON) -o Modules/_testmultiphase$(EXT_SUFFIX)
Modules/_testsinglephase.o: $(srcdir)/Modules/_testsinglephase.c $(MODULE__TESTSINGLEPHASE_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTSINGLEPHASE_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testsinglephase.c -o Modules/_testsinglephase.o
Modules/_testsinglephase$(EXT_SUFFIX):  Modules/_testsinglephase.o; $(BLDSHARED)  Modules/_testsinglephase.o $(MODULE__TESTSINGLEPHASE_LDFLAGS) $(LIBPYTHON) -o Modules/_testsinglephase$(EXT_SUFFIX)
Modules/_testexternalinspection.o: $(srcdir)/Modules/_testexternalinspection.c $(MODULE__TESTEXTERNALINSPECTION_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__TESTEXTERNALINSPECTION_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_testexternalinspection.c -o Modules/_testexternalinspection.o
Modules/_testexternalinspection$(EXT_SUFFIX):  Modules/_testexternalinspection.o; $(BLDSHARED)  Modules/_testexternalinspection.o $(MODULE__TESTEXTERNALINSPECTION_LDFLAGS) $(LIBPYTHON) -o Modules/_testexternalinspection$(EXT_SUFFIX)
Modules/_ctypes/_ctypes_test.o: $(srcdir)/Modules/_ctypes/_ctypes_test.c $(MODULE__CTYPES_TEST_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE__CTYPES_TEST_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/_ctypes/_ctypes_test.c -o Modules/_ctypes/_ctypes_test.o
Modules/_ctypes_test$(EXT_SUFFIX):  Modules/_ctypes/_ctypes_test.o; $(BLDSHARED)  Modules/_ctypes/_ctypes_test.o $(MODULE__CTYPES_TEST_LDFLAGS) $(LIBPYTHON) -o Modules/_ctypes_test$(EXT_SUFFIX)
Modules/xxlimited.o: $(srcdir)/Modules/xxlimited.c $(MODULE_XXLIMITED_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_XXLIMITED_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/xxlimited.c -o Modules/xxlimited.o
Modules/xxlimited$(EXT_SUFFIX):  Modules/xxlimited.o; $(BLDSHARED)  Modules/xxlimited.o $(MODULE_XXLIMITED_LDFLAGS) $(LIBPYTHON) -o Modules/xxlimited$(EXT_SUFFIX)
Modules/xxlimited_35.o: $(srcdir)/Modules/xxlimited_35.c $(MODULE_XXLIMITED_35_DEPS) $(MODULE_DEPS_SHARED) $(PYTHON_HEADERS); $(CC) $(MODULE_XXLIMITED_35_CFLAGS) $(PY_STDMODULE_CFLAGS) $(CCSHARED) -c $(srcdir)/Modules/xxlimited_35.c -o Modules/xxlimited_35.o
Modules/xxlimited_35$(EXT_SUFFIX):  Modules/xxlimited_35.o; $(BLDSHARED)  Modules/xxlimited_35.o $(MODULE_XXLIMITED_35_LDFLAGS) $(LIBPYTHON) -o Modules/xxlimited_35$(EXT_SUFFIX)
Modules/atexitmodule.o: $(srcdir)/Modules/atexitmodule.c $(MODULE_ATEXIT_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE_ATEXIT_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/atexitmodule.c -o Modules/atexitmodule.o
Modules/atexit$(EXT_SUFFIX):  Modules/atexitmodule.o; $(BLDSHARED)  Modules/atexitmodule.o $(MODULE_ATEXIT_LDFLAGS) $(LIBPYTHON) -o Modules/atexit$(EXT_SUFFIX)
Modules/faulthandler.o: $(srcdir)/Modules/faulthandler.c $(MODULE_FAULTHANDLER_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE_FAULTHANDLER_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/faulthandler.c -o Modules/faulthandler.o
Modules/faulthandler$(EXT_SUFFIX):  Modules/faulthandler.o; $(BLDSHARED)  Modules/faulthandler.o $(MODULE_FAULTHANDLER_LDFLAGS) $(LIBPYTHON) -o Modules/faulthandler$(EXT_SUFFIX)
Modules/posixmodule.o: $(srcdir)/Modules/posixmodule.c $(MODULE_POSIX_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE_POSIX_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/posixmodule.c -o Modules/posixmodule.o
Modules/posix$(EXT_SUFFIX):  Modules/posixmodule.o; $(BLDSHARED)  Modules/posixmodule.o $(MODULE_POSIX_LDFLAGS) $(LIBPYTHON) -o Modules/posix$(EXT_SUFFIX)
Modules/signalmodule.o: $(srcdir)/Modules/signalmodule.c $(MODULE__SIGNAL_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__SIGNAL_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/signalmodule.c -o Modules/signalmodule.o
Modules/_signal$(EXT_SUFFIX):  Modules/signalmodule.o; $(BLDSHARED)  Modules/signalmodule.o $(MODULE__SIGNAL_LDFLAGS) $(LIBPYTHON) -o Modules/_signal$(EXT_SUFFIX)
Modules/_tracemalloc.o: $(srcdir)/Modules/_tracemalloc.c $(MODULE__TRACEMALLOC_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__TRACEMALLOC_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_tracemalloc.c -o Modules/_tracemalloc.o
Modules/_tracemalloc$(EXT_SUFFIX):  Modules/_tracemalloc.o; $(BLDSHARED)  Modules/_tracemalloc.o $(MODULE__TRACEMALLOC_LDFLAGS) $(LIBPYTHON) -o Modules/_tracemalloc$(EXT_SUFFIX)
Modules/_suggestions.o: $(srcdir)/Modules/_suggestions.c $(MODULE__SUGGESTIONS_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__SUGGESTIONS_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_suggestions.c -o Modules/_suggestions.o
Modules/_suggestions$(EXT_SUFFIX):  Modules/_suggestions.o; $(BLDSHARED)  Modules/_suggestions.o $(MODULE__SUGGESTIONS_LDFLAGS) $(LIBPYTHON) -o Modules/_suggestions$(EXT_SUFFIX)
Modules/_codecsmodule.o: $(srcdir)/Modules/_codecsmodule.c $(MODULE__CODECS_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__CODECS_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_codecsmodule.c -o Modules/_codecsmodule.o
Modules/_codecs$(EXT_SUFFIX):  Modules/_codecsmodule.o; $(BLDSHARED)  Modules/_codecsmodule.o $(MODULE__CODECS_LDFLAGS) $(LIBPYTHON) -o Modules/_codecs$(EXT_SUFFIX)
Modules/_collectionsmodule.o: $(srcdir)/Modules/_collectionsmodule.c $(MODULE__COLLECTIONS_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__COLLECTIONS_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_collectionsmodule.c -o Modules/_collectionsmodule.o
Modules/_collections$(EXT_SUFFIX):  Modules/_collectionsmodule.o; $(BLDSHARED)  Modules/_collectionsmodule.o $(MODULE__COLLECTIONS_LDFLAGS) $(LIBPYTHON) -o Modules/_collections$(EXT_SUFFIX)
Modules/errnomodule.o: $(srcdir)/Modules/errnomodule.c $(MODULE_ERRNO_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE_ERRNO_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/errnomodule.c -o Modules/errnomodule.o
Modules/errno$(EXT_SUFFIX):  Modules/errnomodule.o; $(BLDSHARED)  Modules/errnomodule.o $(MODULE_ERRNO_LDFLAGS) $(LIBPYTHON) -o Modules/errno$(EXT_SUFFIX)
Modules/_io/_iomodule.o: $(srcdir)/Modules/_io/_iomodule.c $(MODULE__IO_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__IO_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_io/_iomodule.c -o Modules/_io/_iomodule.o
Modules/_io/iobase.o: $(srcdir)/Modules/_io/iobase.c $(MODULE__IO_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__IO_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_io/iobase.c -o Modules/_io/iobase.o
Modules/_io/fileio.o: $(srcdir)/Modules/_io/fileio.c $(MODULE__IO_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__IO_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_io/fileio.c -o Modules/_io/fileio.o
Modules/_io/bytesio.o: $(srcdir)/Modules/_io/bytesio.c $(MODULE__IO_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__IO_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_io/bytesio.c -o Modules/_io/bytesio.o
Modules/_io/bufferedio.o: $(srcdir)/Modules/_io/bufferedio.c $(MODULE__IO_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__IO_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_io/bufferedio.c -o Modules/_io/bufferedio.o
Modules/_io/textio.o: $(srcdir)/Modules/_io/textio.c $(MODULE__IO_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__IO_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_io/textio.c -o Modules/_io/textio.o
Modules/_io/stringio.o: $(srcdir)/Modules/_io/stringio.c $(MODULE__IO_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__IO_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_io/stringio.c -o Modules/_io/stringio.o
Modules/_io$(EXT_SUFFIX):  Modules/_io/_iomodule.o Modules/_io/iobase.o Modules/_io/fileio.o Modules/_io/bytesio.o Modules/_io/bufferedio.o Modules/_io/textio.o Modules/_io/stringio.o; $(BLDSHARED)  Modules/_io/_iomodule.o Modules/_io/iobase.o Modules/_io/fileio.o Modules/_io/bytesio.o Modules/_io/bufferedio.o Modules/_io/textio.o Modules/_io/stringio.o $(MODULE__IO_LDFLAGS) $(LIBPYTHON) -o Modules/_io$(EXT_SUFFIX)
Modules/itertoolsmodule.o: $(srcdir)/Modules/itertoolsmodule.c $(MODULE_ITERTOOLS_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE_ITERTOOLS_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/itertoolsmodule.c -o Modules/itertoolsmodule.o
Modules/itertools$(EXT_SUFFIX):  Modules/itertoolsmodule.o; $(BLDSHARED)  Modules/itertoolsmodule.o $(MODULE_ITERTOOLS_LDFLAGS) $(LIBPYTHON) -o Modules/itertools$(EXT_SUFFIX)
Modules/_sre/sre.o: $(srcdir)/Modules/_sre/sre.c $(MODULE__SRE_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__SRE_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_sre/sre.c -o Modules/_sre/sre.o
Modules/_sre$(EXT_SUFFIX):  Modules/_sre/sre.o; $(BLDSHARED)  Modules/_sre/sre.o $(MODULE__SRE_LDFLAGS) $(LIBPYTHON) -o Modules/_sre$(EXT_SUFFIX)
Modules/_sysconfig.o: $(srcdir)/Modules/_sysconfig.c $(MODULE__SYSCONFIG_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__SYSCONFIG_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_sysconfig.c -o Modules/_sysconfig.o
Modules/_sysconfig$(EXT_SUFFIX):  Modules/_sysconfig.o; $(BLDSHARED)  Modules/_sysconfig.o $(MODULE__SYSCONFIG_LDFLAGS) $(LIBPYTHON) -o Modules/_sysconfig$(EXT_SUFFIX)
Modules/_threadmodule.o: $(srcdir)/Modules/_threadmodule.c $(MODULE__THREAD_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__THREAD_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_threadmodule.c -o Modules/_threadmodule.o
Modules/_thread$(EXT_SUFFIX):  Modules/_threadmodule.o; $(BLDSHARED)  Modules/_threadmodule.o $(MODULE__THREAD_LDFLAGS) $(LIBPYTHON) -o Modules/_thread$(EXT_SUFFIX)
Modules/timemodule.o: $(srcdir)/Modules/timemodule.c $(MODULE_TIME_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE_TIME_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/timemodule.c -o Modules/timemodule.o
Modules/time$(EXT_SUFFIX):  Modules/timemodule.o; $(BLDSHARED)  Modules/timemodule.o $(MODULE_TIME_LDFLAGS) $(LIBPYTHON) -o Modules/time$(EXT_SUFFIX)
Modules/_typingmodule.o: $(srcdir)/Modules/_typingmodule.c $(MODULE__TYPING_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__TYPING_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_typingmodule.c -o Modules/_typingmodule.o
Modules/_typing$(EXT_SUFFIX):  Modules/_typingmodule.o; $(BLDSHARED)  Modules/_typingmodule.o $(MODULE__TYPING_LDFLAGS) $(LIBPYTHON) -o Modules/_typing$(EXT_SUFFIX)
Modules/_weakref.o: $(srcdir)/Modules/_weakref.c $(MODULE__WEAKREF_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__WEAKREF_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_weakref.c -o Modules/_weakref.o
Modules/_weakref$(EXT_SUFFIX):  Modules/_weakref.o; $(BLDSHARED)  Modules/_weakref.o $(MODULE__WEAKREF_LDFLAGS) $(LIBPYTHON) -o Modules/_weakref$(EXT_SUFFIX)
Modules/_abc.o: $(srcdir)/Modules/_abc.c $(MODULE__ABC_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__ABC_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_abc.c -o Modules/_abc.o
Modules/_abc$(EXT_SUFFIX):  Modules/_abc.o; $(BLDSHARED)  Modules/_abc.o $(MODULE__ABC_LDFLAGS) $(LIBPYTHON) -o Modules/_abc$(EXT_SUFFIX)
Modules/_functoolsmodule.o: $(srcdir)/Modules/_functoolsmodule.c $(MODULE__FUNCTOOLS_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__FUNCTOOLS_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_functoolsmodule.c -o Modules/_functoolsmodule.o
Modules/_functools$(EXT_SUFFIX):  Modules/_functoolsmodule.o; $(BLDSHARED)  Modules/_functoolsmodule.o $(MODULE__FUNCTOOLS_LDFLAGS) $(LIBPYTHON) -o Modules/_functools$(EXT_SUFFIX)
Modules/_localemodule.o: $(srcdir)/Modules/_localemodule.c $(MODULE__LOCALE_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__LOCALE_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_localemodule.c -o Modules/_localemodule.o
Modules/_locale$(EXT_SUFFIX):  Modules/_localemodule.o; $(BLDSHARED)  Modules/_localemodule.o $(MODULE__LOCALE_LDFLAGS) $(LIBPYTHON) -o Modules/_locale$(EXT_SUFFIX)
Modules/_operator.o: $(srcdir)/Modules/_operator.c $(MODULE__OPERATOR_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__OPERATOR_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_operator.c -o Modules/_operator.o
Modules/_operator$(EXT_SUFFIX):  Modules/_operator.o; $(BLDSHARED)  Modules/_operator.o $(MODULE__OPERATOR_LDFLAGS) $(LIBPYTHON) -o Modules/_operator$(EXT_SUFFIX)
Modules/_stat.o: $(srcdir)/Modules/_stat.c $(MODULE__STAT_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__STAT_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/_stat.c -o Modules/_stat.o
Modules/_stat$(EXT_SUFFIX):  Modules/_stat.o; $(BLDSHARED)  Modules/_stat.o $(MODULE__STAT_LDFLAGS) $(LIBPYTHON) -o Modules/_stat$(EXT_SUFFIX)
Modules/symtablemodule.o: $(srcdir)/Modules/symtablemodule.c $(MODULE__SYMTABLE_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE__SYMTABLE_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/symtablemodule.c -o Modules/symtablemodule.o
Modules/_symtable$(EXT_SUFFIX):  Modules/symtablemodule.o; $(BLDSHARED)  Modules/symtablemodule.o $(MODULE__SYMTABLE_LDFLAGS) $(LIBPYTHON) -o Modules/_symtable$(EXT_SUFFIX)
Modules/pwdmodule.o: $(srcdir)/Modules/pwdmodule.c $(MODULE_PWD_DEPS) $(MODULE_DEPS_STATIC) $(PYTHON_HEADERS); $(CC) $(MODULE_PWD_CFLAGS) $(PY_BUILTIN_MODULE_CFLAGS) -c $(srcdir)/Modules/pwdmodule.c -o Modules/pwdmodule.o
Modules/pwd$(EXT_SUFFIX):  Modules/pwdmodule.o; $(BLDSHARED)  Modules/pwdmodule.o $(MODULE_PWD_LDFLAGS) $(LIBPYTHON) -o Modules/pwd$(EXT_SUFFIX)