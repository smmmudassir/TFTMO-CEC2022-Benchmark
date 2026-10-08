import ctypes,os,numpy as np
class M(ctypes.Structure):
 _fields_=[('score',ctypes.c_double),('j_loss',ctypes.c_double),('eta_w',ctypes.c_double),('zvs_cov',ctypes.c_double),('max_irms',ctypes.c_double),('max_ipk',ctypes.c_double),('max_tj',ctypes.c_double),('max_ripple',ctypes.c_double),('cv',ctypes.c_double),('max_track',ctypes.c_double),('min_zvs_high',ctypes.c_double),('bpk',ctypes.c_double),('feasible',ctypes.c_int)]
_lib=ctypes.CDLL(os.path.join(os.path.dirname(__file__),'libdab.so'));_lib.dab_eval.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int,ctypes.POINTER(M)];_lib.dab_eval.restype=ctypes.c_double
def evaluate(x,metrics=False):
 a=np.ascontiguousarray(x,dtype=np.float64);m=M();s=_lib.dab_eval(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(a),ctypes.byref(m));return {k:getattr(m,k) for k,_ in M._fields_} if metrics else float(s)
