import main as m
import filter as f


I = m.ImportAsPng("export/Rebuild.png")

P = f.NiblackParam(I, 0.2, 35)

print(P)
