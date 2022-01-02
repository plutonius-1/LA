from ticker_c import *
import matplotlib.pyplot as plt

tick = Ticker_c()
main_df = pd.read_csv("./HistoricalData.csv")
tick.set_main_df(main_df)
gs = [(1,"r--"), (2, "bs"), (3, "g^")]
ress = []
for pair in gs:
    K = tick.parse_kinetic_energy(main_df, pair[0])
    index = K.index.tolist()
    vals  = [i[0] for i in K.values.tolist()]
    ress.append(vals)

plt.plot(index, ress[0], index, ress[1], index, ress[2])
plt.show()

