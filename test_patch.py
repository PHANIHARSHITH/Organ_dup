import sklearn.compose._column_transformer as c
class _RemainderColsList(list): pass
c._RemainderColsList = _RemainderColsList
import pickle
print("Testing load with patch...")
try:
    with open('pickel_files/lung_transplant.sav', 'rb') as f:
        m = pickle.load(f)
    print("Success!")
except Exception as e:
    print("Error:", e)
