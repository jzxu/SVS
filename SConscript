import sys

Import('env')

inc = [env.Dir(d).srcnode() for d in 'src src/algorithms src/models eigen ccd'.split()]

# svs viewer
viewer_src = Glob('viewer/*.c')
viewer_libs = [ 'SDL', 'GL', 'GLU', 'm' ]
viewer_env = env.Clone()
viewer_env['LIBS'] = []
config = Configure(viewer_env)
missing_libs = [ l for l in viewer_libs if not config.CheckLib(l)]
viewer_env = config.Finish()
if not missing_libs:
	viewer_prog = viewer_env.Program('svs_viewer', viewer_src)
	viewer_install = viewer_env.Alias('svs_viewer', viewer_env.Install('$OUT_DIR', viewer_prog))
else:
	print 'Cannot find %s, not building svs_viewer' % ', '.join(missing_libs)

# svs library objects
svs_env = env.Clone()
svs_env['LIBS'] = []
svs_env.Prepend(
	CPPPATH = inc,
	CPPFLAGS = [
		# By default Eigen will try to align all fixed size vectors to 128-bit
		# boundaries to enable SIMD instructions on hardware such as SSE. However,
		# this requires that you modify every class that has such vectors as members
		# so that they are correctly allocated. This seems like more trouble than
		# it's worth at the moment, so I'm disabling it.
		'-DEIGEN_DONT_ALIGN',
		'-Wno-enum-compare',
	],
)

ccd_env = env.Clone()
ccd_env['LIBS'] = []
ccd_env['CPPPATH'] = [env.Dir('ccd').srcnode()]
ccd_src = Glob('ccd/*.c')

src = []
for d in ['src', 'src/filters', 'src/commands', 'src/models', 'src/algorithms']:
	src += Glob(d + '/*.cpp')

if GetOption('static'):
	svs_objs = svs_env.Object(src) + ccd_env.Object(ccd_src)
else:
	svs_objs = svs_env.SharedObject(src) + ccd_env.SharedObject(ccd_src)

svs_inc = ['SVS/src']
Return('svs_objs', 'svs_inc')
