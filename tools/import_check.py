import sys, importlib
sys.path.insert(0, 'src')
importlib.invalidate_caches()
importlib.import_module('world.objects.wall_tile')
print('import ok')
