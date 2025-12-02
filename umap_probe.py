import traceback
try:
    import umap
    print('umap file=', getattr(umap, '__file__', None))
    members = [a for a in dir(umap) if not a.startswith('__')]
    print('umap members=', members)
    print('top-level has UMAP:', hasattr(umap, 'UMAP'))
    try:
        import umap.umap_
        print('umap.umap_ file=', getattr(umap.umap_, '__file__', None))
        print('umap.umap_.UMAP:', hasattr(umap.umap_, 'UMAP'))
    except Exception as e:
        print('cannot import umap.umap_:', type(e).__name__, e)
    try:
        import umap._umap as _u
        print('umap._umap file=', getattr(_u, '__file__', None))
        print('umap._umap has UMAP:', hasattr(_u, 'UMAP'))
    except Exception as e:
        print('cannot import umap._umap:', type(e).__name__, e)
except Exception as e:
    print('import umap failed:', type(e).__name__, e)
    traceback.print_exc()
