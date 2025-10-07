# %%
from tsfm import Model, generator

df = generator()

mdl = Model.build(name="armodel")
yhs = mdl.pred(df, y="y", horizon=3, oos_start="2005-01-31")
print(yhs.head(10))
