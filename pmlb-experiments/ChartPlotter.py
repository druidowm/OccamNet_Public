from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import math
import string
from matplotlib.backends.backend_pdf import PdfPages

titlesize = 18
labelsize = 15

def plot(ax, number, occam, other, title, x_label = False, y_label = False, legend = False):
    ax.text(0, 1.03, string.ascii_lowercase[number]+")", transform=ax.transAxes, size=titlesize, weight='bold')
    occam = occam.to_list()
    other = other.to_list()

    occamBetterName = []
    occamBetterX = []
    occamBetter = []
    otherBetterName = []
    otherBetterX = []
    otherBetter = []

    #print(occam)
    #print(other)

    j = 0
    for i in range(len(occam)):
        if not math.isnan(other[i]):
            if occam[i]<other[i]:
                occamBetterName.append(i+1)
                occamBetter.append(other[i]/occam[i])
                occamBetterX.append(j)
            else:
                otherBetterName.append(i+1)
                otherBetter.append(occam[i]/other[i])
                otherBetterX.append(j)
            j+=1
        else:
            print("here")
            pass

    #print(occamBetter)

    maxHeight = 1.1*max(max(occamBetter),max(otherBetter))

    ax.set_ylim(1,maxHeight)

    rects1 = ax.bar(occamBetterX, occamBetter, 0.7, color = "tab:green", label = "OccamNet Performs Better")
    for i, v in enumerate(occamBetter):
        if v >= 10:
            ax.text(occamBetterX[i], v + 0.01*(maxHeight-1), str(round(v,1)), size = 8, color = "tab:green", ha='center')
        else:
            ax.text(occamBetterX[i], v + 0.01*(maxHeight-1), str(round(v,2)), size = 8, color = "tab:green", ha='center')

    rects2 = ax.bar(otherBetterX, otherBetter, 0.7, color = "tab:red", label = "Comparison Performs Better")
    for i, v in enumerate(otherBetter):
        if v >= 10:
            ax.text(otherBetterX[i], v + 0.01*(maxHeight-1), str(round(v,1)), size = 8, color = "tab:red", ha='center')
        else:
            ax.text(otherBetterX[i], v + 0.01*(maxHeight-1), str(round(v,2)), size = 8, color = "tab:red", ha='center')

    tickLabels = (occamBetterName+otherBetterName)
    tickLabels.sort()

    if x_label:
        ax.set_xlabel("Dataset", size = labelsize)
    if y_label:
        ax.set_ylabel("Higher/Lower", size = labelsize)
    ax.set_title(title, size = titlesize)
    ax.set_xticks(range(len(occamBetter)+len(otherBetter)))
    ax.set_xticklabels(tickLabels, size = labelsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(labelsize) 

    #ax.set_yscale("log")

    if legend:
        ax.legend(loc = 2, fontsize = labelsize)

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)


pp = PdfPages('PMLBResults.pdf')

data = pd.read_csv("../../../Desktop/TrainTestValResults/CSVData2.csv")

print(data)

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, figsize = (18,7))

plot(ax1, 0, data["OccamNet1M Train"], data["Eplex1M Train"], "OccamNet v. Eplex, Training", y_label = True, legend = True)
plot(ax2, 1, data["OccamNet1M Val"], data["Eplex1M Val"], "OccamNet v. Eplex, Validation")
plot(ax3, 2, data["OccamNet1M Test"], data["Eplex1M Test"], "OccamNet v. Eplex, Testing")
plot(ax4, 3, data["OccamNet1M Time"], data["Eplex1M Time"], "OccamNet v. Eplex, Timing", x_label = True, y_label = True)
plot(ax5, 4, data["OccamNet V100 Test"], data["Eplex1M Test"], "OccamNetV100 v. Eplex, Testing", x_label = True)
plot(ax6, 5, data["OccamNet1M Test"], data["Feynman Test"], "OccamNet v. AIF, Testing", x_label = True)
#plt.show()
fig.tight_layout()
pp.savefig()
pp.close()